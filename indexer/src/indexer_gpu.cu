#ifndef __USE_XOPEN
#define __USE_XOPEN // for <cmath> M_PI
#endif
#include <iostream>
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <map>
#include <memory>
#include <chrono>
#include <cmath>    // for M_PI    
#include "exception.h"
#include "log.h"
#include "indexer_gpu.h"
#include "cuda_runtime.h"
#include <cub/cub.cuh>

using logger::stanza;

namespace {

    constexpr char INDEXER_GPU_DEVICE[] = "INDEXER_GPU_DEVICE";
    constexpr unsigned n_threads = 512; // num cuda threads per block for find_candidates kernel

    template<typename float_type>
    struct constant final {
        static constexpr float_type pi = M_PI;
        static constexpr float_type pi2 = M_PI_2;
    };

    // Cuda device infos
    struct gpu_device final {
        int id;
        cudaDeviceProp prop;

        // Device list
        static std::vector<gpu_device> list;
    };

    std::vector<gpu_device> gpu_device::list;

    int gpu_device_number;  // Used gpu device, TODO: allow multiple GPU devices

    // Exception for cuda related errors
    struct cuda_exception final : public fast_feedback::exception {

        inline cuda_exception(cudaError_t error, const char* file, unsigned line)
            : exception(std::string{}, file, line)
        {
            const char* error_name = cudaGetErrorName(error);
            const char* error_string = cudaGetErrorString(error);
            *this << '(' << error_name << ") " << error_string;
        }

    };

    #define CU_EXCEPTION(err) cuda_exception(err, __FILE__, __LINE__)

    #define CU_CHECK(err) {        \
        cudaError_t error = (err); \
        if (error != cudaSuccess)  \
            CU_EXCEPTION(error);   \
    }

    // Deleter for device smart pointers
    template<typename T>
    struct gpu_deleter final {
        void operator()(T* ptr) const
        {
            CU_CHECK(cudaFree(ptr));
        }
    };

    // Device smart pointer
    template<typename T>
    using gpu_pointer = std::unique_ptr<T, gpu_deleter<T>>;

    // Is the cuda runtime initialized?
    std::atomic_bool cuda_initialized = false;

    // Protect cuda runtime initalisation
    std::mutex cuda_init_lock;

    // Init CUDA runtime environment
    void cuda_init()
    {
        if (cuda_initialized.load())
            return;
        
        {
            std::lock_guard<std::mutex> lock{cuda_init_lock};

            if (cuda_initialized.load())
                return;
        
            int num_devices;
            CU_CHECK(cudaGetDeviceCount(&num_devices));

            gpu_device::list.resize(num_devices);
            for (int dev=0; dev<num_devices; dev++) {
                gpu_device& device = gpu_device::list[dev];
                device.id = dev;
                CU_CHECK(cudaGetDeviceProperties(&device.prop, dev));
            }

            CU_CHECK(cudaGetDevice(&gpu_device_number));
            {
                char* dev_string = std::getenv(INDEXER_GPU_DEVICE);
                if (dev_string != nullptr) {
                    std::istringstream iss{dev_string};
                    int dev = -1;
                    iss >> dev;
                    if (!iss || !iss.eof())
                        throw FF_EXCEPTION_OBJ << "wrong format for " << INDEXER_GPU_DEVICE << ": " << dev_string << " (should be an integer)\n";
                    if ((dev < 0) || (dev >= num_devices))
                        throw FF_EXCEPTION_OBJ << "illegal value for " << INDEXER_GPU_DEVICE << ": " << dev << " (should be in [0, " << num_devices << "[)\n";
                    CU_CHECK(cudaSetDevice(dev));
                    CU_CHECK(cudaGetDevice(&gpu_device_number));
                }
            }

            logger::info << stanza << "using GPU device " << gpu_device_number << '\n';
            cuda_initialized.store(true);
        } // release cuda_init_lock
    }

    // Check for an initialized CUDA runtime
    #define CU_CHECK_INIT {                                      \
        if (! cuda_initialized.load())                           \
            throw FF_EXCEPTION("cuda runtime not initialized");  \
    }

    // Cuda event wrapper
    template<unsigned flags=cudaEventDisableTiming>
    struct gpu_event final {
        cudaEvent_t event;

        inline gpu_event()
            : event{}
        {}

        gpu_event(const gpu_event&) = delete;
        gpu_event& operator=(const gpu_event&) = delete;

        gpu_event(gpu_event&& other)
            : event{}
        {
            std::swap(other.event, event);
        }
        
        gpu_event& operator=(gpu_event&& other)
        {
            event = cudaEvent_t{};
            std::swap(other.event, event);
            return *this;
        }

        inline ~gpu_event()
        {
            if (event != cudaEvent_t{})
                CU_CHECK(cudaEventDestroy(event));
        }

        inline void init()
        {
            if (event == cudaEvent_t{}) {
                CU_CHECK_INIT;
                CU_CHECK(cudaEventCreateWithFlags(&event, flags));
            }
        }

        inline void record(cudaStream_t stream=0)
        {
            CU_CHECK(cudaEventRecord(event, stream));
        }

        inline void sync()
        {
            CU_CHECK(cudaEventSynchronize(event));
        }
    };

    template<unsigned flags>
    static float gpu_timing(const gpu_event<flags>& start, const gpu_event<flags>& end)
    {
        static_assert(!(flags & cudaEventDisableTiming), "GPU timing measurement are only valid on events that don't have the cudaEventDisableTiming flag set");
        float time;
        CU_CHECK(cudaEventElapsedTime(&time, start.event, end.event));
        return time;
    }

    // On GPU compact vector representation
    template<typename float_type>
    struct compact_vec final {
        float_type x;
        float_type y;
        float_type z;
    };

    // On GPU data for indexer
    template<typename float_type>
    struct indexer_device_data final {
        fast_feedback::config_persistent<float_type> cpers;
        fast_feedback::config_runtime<float_type> crt;
        fast_feedback::input<float_type> input;
        fast_feedback::output<float_type> output;
        float_type* candidate_value;            // Candidate vector objective function values, [3 * max_input_cells * num_candidate_vectors]
        unsigned* candidate_sample;             // Sample number of candidate vectors, [3 * max_input_cells * num_candidate_vectors]
    };

    // Indexer GPU state representation on the Host side
    //
    // Candidate vector group = set of vectors with same length that are candidates for a unit cell vector
    //
    template<typename float_type>
    struct indexer_gpu_state final {
        using content_type = indexer_device_data<float_type>;
        using key_type = fast_feedback::state_id::type;
        using map_type = std::map<key_type, indexer_gpu_state>;
        using config_persistent = fast_feedback::config_persistent<float_type>;
        using memory_pin = fast_feedback::memory_pin;

        gpu_pointer<content_type> data;                     // indexer_device_data on GPU
        gpu_pointer<float_type> elements;                   // Input/output vector elements of data on GPU
        gpu_pointer<float_type> candidate_value;            // Candidate vector objective function values on GPU
        gpu_pointer<unsigned> candidate_sample;             // Sample number of candidate vectors

        // Temporary pinned space to transfer n_input_cells, n_output_cells, n_spots
        fast_feedback::pinned_ptr<config_persistent> tmp;   // Pointer to make move construction simple

        // Input coordinates on GPU pointing into elements
        float_type* ix;
        float_type* iy;
        float_type* iz;

        // Output coordinates on GPU pointing into elements
        float_type* ox;
        float_type* oy;
        float_type* oz;

        // Timings
        gpu_event<cudaEventDefault> start;
        gpu_event<cudaEventDefault> end;

        static std::mutex state_update; // Protect per indexer state map
        static map_type dev_ptr;        // Per indexer state map

        // Create hostside on GPU state representation with
        // d        on GPU state pointer
        // e        on GPU input and output elements pointed to by *d
        // cv       on GPU candidate vector objective function values
        // cs       on GPU candidate vector sample numbers
        // x/y/zi   on GPU input pointers d->input.x / y / z
        // x/y/zo   on GPU output pointers d->output.x / y / z
        indexer_gpu_state(gpu_pointer<content_type>&& d, gpu_pointer<float_type>&& e,
                          gpu_pointer<float_type>&& cv, gpu_pointer<unsigned>&& cs,
                          float_type* xi, float_type* yi, float_type* zi, float_type* xo, float_type* yo, float_type* zo)
            : data{std::move(d)}, elements{std::move(e)},
              candidate_value{std::move(cv)}, candidate_sample{std::move(cs)},
              ix{xi}, iy{yi}, iz{zi}, ox{xo}, oy{yo}, oz{zo}
        {
            tmp = fast_feedback::alloc_pinned<config_persistent>();
        }

        indexer_gpu_state() = default;                                      // Default with some uninitialized pointers
        indexer_gpu_state(const indexer_gpu_state&) = delete;
        indexer_gpu_state(indexer_gpu_state&&) = default;                   // Take over state representation
        indexer_gpu_state& operator=(const indexer_gpu_state&) = delete;
        indexer_gpu_state& operator=(indexer_gpu_state&&) = default;        // Take over state representation
        ~indexer_gpu_state() = default;                                     // Drop on GPU state data

        // Shortcut to get at reference
        static inline indexer_gpu_state& ref(const key_type& id)
        {
            return dev_ptr[id];
        }

        // Shortcut to get at state data pointer
        static inline gpu_pointer<content_type>& ptr(const key_type& id)
        {
            return ref(id).data;
        }

        // Copy runtime configuration to GPU
        static inline void copy_crt(const key_type& state_id, const fast_feedback::config_runtime<float_type>& crt, cudaStream_t stream=0)
        {
            const auto crt_dp = &ptr(state_id)->crt;
            logger::debug << stanza << "copy runtime config data: " << &crt << "-->" << crt_dp << '\n';
            CU_CHECK(cudaMemcpyAsync(crt_dp, &crt, sizeof(crt), cudaMemcpyHostToDevice, stream));
        }

        // Copy input data to GPU
        static inline void copy_in(const key_type& state_id, const config_persistent& cpers,
                                   const fast_feedback::input<float_type>& input, cudaStream_t stream=0)
        {
            const auto& gpu_state = ref(state_id);
            auto& pinned_tmp = const_cast<config_persistent&>(*gpu_state.tmp);
            const auto& device_data = gpu_state.data;
            const auto n_cells = std::min(input.n_cells, cpers.max_input_cells);
            const auto n_spots = std::min(input.n_spots, cpers.max_spots);

            // NOTE: the following code assumes consecutive storage of two data members in tmp and in device_data->input
            pinned_tmp.max_input_cells = n_cells;
            pinned_tmp.max_spots = n_spots;
            CU_CHECK(cudaMemcpyAsync(&device_data->input.n_cells, &pinned_tmp.max_input_cells, sizeof(n_cells) + sizeof(n_spots), cudaMemcpyHostToDevice, stream));

            if (n_cells + n_spots > 0) {
                if (n_cells == input.n_cells) {
                    const auto combi_sz = (3 * n_cells + n_spots) * sizeof(float_type);
                    CU_CHECK(cudaMemcpyAsync(gpu_state.ix, input.x, combi_sz, cudaMemcpyHostToDevice, stream));
                    CU_CHECK(cudaMemcpyAsync(gpu_state.iy, input.y, combi_sz, cudaMemcpyHostToDevice, stream));
                    CU_CHECK(cudaMemcpyAsync(gpu_state.iz, input.z, combi_sz, cudaMemcpyHostToDevice, stream));
                } else {
                    const auto dst_offset = 3 * n_cells;
                    if (n_cells > 0) {
                        const auto cell_sz = dst_offset * sizeof(float_type);
                        CU_CHECK(cudaMemcpyAsync(gpu_state.ix, input.x, cell_sz, cudaMemcpyHostToDevice, stream));
                        CU_CHECK(cudaMemcpyAsync(gpu_state.iy, input.y, cell_sz, cudaMemcpyHostToDevice, stream));
                        CU_CHECK(cudaMemcpyAsync(gpu_state.iz, input.z, cell_sz, cudaMemcpyHostToDevice, stream));
                    }
                    if (n_spots > 0) {
                        const auto src_offset = input.n_cells;
                        const auto spot_sz = n_spots * sizeof(float_type);
                        CU_CHECK(cudaMemcpyAsync(&gpu_state.ix[dst_offset], &input.x[src_offset], spot_sz, cudaMemcpyHostToDevice, stream));
                        CU_CHECK(cudaMemcpyAsync(&gpu_state.iy[dst_offset], &input.y[src_offset], spot_sz, cudaMemcpyHostToDevice, stream));
                        CU_CHECK(cudaMemcpyAsync(&gpu_state.iz[dst_offset], &input.z[src_offset], spot_sz, cudaMemcpyHostToDevice, stream));
                    }
                }
                // CU_CHECK(cudaStreamSynchronize(stream));
            }

            logger::debug << stanza << "copy in: " << n_cells << " cells, " << n_spots << " spots, elements=" << gpu_state.elements.get() << ": "
                          << input.x << "-->" << gpu_state.ix << ", "
                          << input.y << "-->" << gpu_state.iy << ", "
                          << input.z << "-->" << gpu_state.iz << '\n';
        }

        // Copy output data from GPU
        static inline void copy_out(const key_type& state_id, fast_feedback::output<float_type>& output, cudaStream_t stream=0)
        {
            const auto& gpu_state = ref(state_id);
            auto& pinned_tmp = const_cast<config_persistent&>(*gpu_state.tmp);
            const auto& device_data = gpu_state.data;

            CU_CHECK(cudaMemcpyAsync(&pinned_tmp.max_output_cells, &device_data->output.n_cells, sizeof(output.n_cells), cudaMemcpyDeviceToHost, stream));
            CU_CHECK(cudaStreamSynchronize(stream));
            output.n_cells = pinned_tmp.max_output_cells;

            if (output.n_cells > 0) {
                const auto cell_sz = 3 * output.n_cells * sizeof(float_type);
                CU_CHECK(cudaMemcpyAsync(output.x, gpu_state.ox, cell_sz, cudaMemcpyDeviceToHost, stream));
                CU_CHECK(cudaMemcpyAsync(output.y, gpu_state.oy, cell_sz, cudaMemcpyDeviceToHost, stream));
                CU_CHECK(cudaMemcpyAsync(output.z, gpu_state.oz, cell_sz, cudaMemcpyDeviceToHost, stream));
                CU_CHECK(cudaStreamSynchronize(stream));
            }

            logger::debug << stanza << "copy out: " << output.n_cells << " cells, elements=" << gpu_state.elements.get() << ": "
                          << gpu_state.ox << "-->" << output.x << ", "
                          << gpu_state.oy << "-->" << output.y << ", "
                          << gpu_state.oz << "-->" << output.z << '\n';
        }
    };

    template<> std::mutex indexer_gpu_state<float>::state_update{};
    template<> indexer_gpu_state<float>::map_type indexer_gpu_state<float>::dev_ptr{};

    template<typename float_type>
    struct indexer_args final {
        float_type candidate_length;    // candidate vector length for current kernel
        unsigned candidate_group;       // candidate group for current kernel
        unsigned num_candidate_groups;  // number of candidate vector groups
        unsigned num_angular_steps;     // M_PI / angular_step
    };

    template<typename float_type>
    struct vec_cand_t final {
        float_type value;   // objective function value
        unsigned sample;    // | alpha_index | beta_index |
    };

    template<typename float_type>
    using BlockRadixSort = cub::BlockRadixSort<float_type, n_threads, 1, unsigned>;

    // -----------------------------------
    //            GPU Kernels
    // -----------------------------------

    template<typename float_type>
    struct util final {};

    template<>
    struct util<float> final {
        __device__ __forceinline__ static void sincos(float angle, float* sine, float* cosine)
        {
            return sincosf(angle, sine, cosine);
        }

        __device__ __forceinline__ static float cos(float angle)
        {
            return cosf(angle);
        }
    };

    // Dummy kernel for dummy indexer test: copy first input cell to output cell
    template<typename float_type>
    __global__ void gpu_dummy_test(indexer_device_data<float_type>* data, [[maybe_unused]] indexer_args<float_type> args)
    {
        // NOTE: assume max_output_cells is >= 1u
        for (unsigned i=0; i<3u; ++i) {
            data->output.x[i] = data->input.x[i];
            data->output.y[i] = data->input.y[i];
            data->output.z[i] = data->input.z[i];
        }
        data->output.n_cells = 1u;
    }

    template<typename float_type>
    __global__ void gpu_find_candidates(indexer_device_data<float_type>* data, indexer_args<float_type> args)
    {
        extern __shared__ double* shared_ptr[];

        const unsigned sample = blockDim.x * blockIdx.x + threadIdx.x;
        const float_type r = args.candidate_length;
        float_type v{}; // objective function value for sample vector

        {   // calculate v
            const unsigned n_steps = args.num_angular_steps;
            const unsigned i_alpha = sample / n_steps;
            float_type x, y, z; // sample vector
            {
                const unsigned i_beta = sample % n_steps;
                const float_type angle = data->crt.angular_step;
                const float_type alpha = i_alpha * angle;
                const float_type beta = i_beta * angle;

                float_type r_;
                util<float_type>::sincos(alpha, &z, &r_); r_ *= r; z *= r;
                util<float_type>::sincos(beta, &x, &y); x *= r_, y*= r_;
            }

            if (i_alpha < n_steps) {
                const fast_feedback::input<float_type>& in = data->input;
                const float* sx = in.x;
                const float* sy = in.y;
                const float* sz = in.z;
                const unsigned n_cells = in.n_cells;
                const unsigned n_spots = in.n_spots;
                
                unsigned i = 3 * n_cells;
                const float_type r2 = r * r;
                do {
                    const auto p = x * sx[i] + y * sy[i] + z * sz[i];
                    v -= util<float_type>::cos(constant<float_type>::pi2 * p / r2);
                } while (++i < n_spots);
            }
        }

        const unsigned num_candidate_vectors = data->cpers.num_candidate_vectors;
        {   // sort within block {objective function value, sample} ascending by objective function value
            float_type key[1] = {v};
            unsigned val[1] = {sample};
            {
                auto sort_ptr = reinterpret_cast<typename BlockRadixSort<float_type>::TempStorage*>(shared_ptr);
                BlockRadixSort<float_type>(*sort_ptr).Sort(key, val);
            }
            __syncthreads();    // protect sort_ptr[] (it's in the same memory as cand_ptr)
            {   // store top candidate vectors into cand_ptr
                auto cand_ptr = reinterpret_cast<vec_cand_t<float_type>*>(shared_ptr); // [num_candidate_vectors]
                if (threadIdx.x < num_candidate_vectors) {
                    cand_ptr[threadIdx.x] = { *key, *val };
                }
            }
        }
        __syncthreads();    // wait for cand_ptr[]
        {   // merge block result into global 
            auto value_ptr = &data->candidate_value[args.candidate_group * num_candidate_vectors];
            auto sample_ptr = &data->candidate_sample[args.candidate_group * num_candidate_vectors];

            // TODO: aquire merge permission, then merge
        }
    }
}

namespace gpu {

    template <typename float_type>
    void init (const indexer<float_type>& instance)
    {
        using gpu_state = indexer_gpu_state<float_type>;
        using device_data = indexer_device_data<float_type>;

        cuda_init();
        drop<float_type>(instance); // drop state if any

        const auto state_id = instance.state;
        const auto& cpers = instance.cpers;

        device_data* data = nullptr;
        float_type* candidate_value = nullptr;
        unsigned* candidate_sample = nullptr;
        float* elements = nullptr;
        float* ix;
        float* iy;
        float* iz;
        float* ox;
        float* oy;
        float* oz;
        {
            CU_CHECK(cudaMalloc(&data, sizeof(device_data)));
            gpu_pointer<device_data> data_ptr{data};
        
            {
                std::size_t vector_dimension = 3 * (cpers.max_output_cells + cpers.max_input_cells) + cpers.max_spots;
                CU_CHECK(cudaMalloc(&elements, 3 * vector_dimension * sizeof(float_type)));
            }
            gpu_pointer<float_type> element_ptr{elements};

            {
                std::size_t vector_dimension = 3 * cpers.num_candidate_vectors * cpers.max_input_cells;
                CU_CHECK(cudaMalloc(&candidate_value, vector_dimension * sizeof(float_type)));
            }
            gpu_pointer<float_type> candidate_value_ptr{candidate_value};

            {
                std::size_t vector_dimension = 3 * cpers.num_candidate_vectors * cpers.max_input_cells;
                CU_CHECK(cudaMalloc(&candidate_sample, vector_dimension * sizeof(unsigned)));
            }
            gpu_pointer<unsigned> candidate_sample_ptr{candidate_sample};

            std::size_t dim = 3 * cpers.max_output_cells;
            ox = elements;
            oy = ox + dim;
            oz = oy + dim;

            ix = oz + dim;
            dim = 3 * cpers.max_input_cells + cpers.max_spots;
            iy = ix + dim;
            iz = iy + dim;

            {
                std::lock_guard<std::mutex> state_lock{gpu_state::state_update};
                gpu_state::ref(state_id) = gpu_state{std::move(data_ptr), std::move(element_ptr),
                                                     std::move(candidate_value_ptr), std::move(candidate_sample_ptr),
                                                     ix, iy, iz, ox, oy, oz};
            }

            logger::debug << stanza << "init id=" << state_id << ", elements=" << elements << '\n'
                          << stanza << "     ox=" << ox << ", oy=" << oy << ", oz=" << oz << '\n'
                          << stanza << "     ix=" << ix << ", iy=" << iy << ", iz=" << iz << '\n'
                          << stanza << "     candval=" << candidate_value << ", candsmpl=" << candidate_sample << '\n';
        }

        {
            device_data _data{cpers, fast_feedback::config_runtime<float_type>{},
                              fast_feedback::input<float_type>{ix, iy, iz, 0, 0},
                              fast_feedback::output<float_type>{ox, oy, oz, 0},
                              candidate_value, candidate_sample};
            CU_CHECK(cudaMemcpy(data, &_data, sizeof(_data), cudaMemcpyHostToDevice));
        }
    }

    // Drop state if any
    template <typename float_type>
    void drop (const indexer<float_type>& instance)
    {
        using gpu_state = indexer_gpu_state<float_type>;

        CU_CHECK_INIT;
        gpu_state::dev_ptr.erase(instance.state);
    }

    // Run indexer
    template <typename float_type>
    void index (const indexer<float_type>& instance, const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt)
    {
        using gpu_state = indexer_gpu_state<float_type>;
        using clock = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<double, std::milli>;
        using time_point = std::chrono::time_point<clock>;

        CU_CHECK_INIT;
        auto state_id = instance.state;
        auto& state = gpu_state::ref(state_id);

        bool timing = logger::level_active<logger::l_info>();
        time_point start, end;
        if (timing) {
            start = clock::now();
            state.start.init();
            state.end.init();
        }

        // Check input
        const auto n_cells = std::min(in.n_cells, instance.cpers.max_input_cells);
        logger::debug << stanza << "n_cells = " << n_cells << ", n_spots = " << in.n_spots << '\n';
        if (n_cells <= 0u)
            throw FF_EXCEPTION("no given input cells");
        if (in.n_spots <= 0u)
            throw FF_EXCEPTION("no spots");
        if (conf_rt.angular_step <= float_type{})
            throw FF_EXCEPTION("nonpositive angular step");
        if (conf_rt.angular_step < constant<float_type>::pi / std::numeric_limits<uint16_t>::max())
            throw FF_EXCEPTION("angular step too small");
        
        // Extra indexer arguments
        indexer_args<float_type> extra_args;

        // Calculate input vector groups
        std::vector<float_type> candidate_length(3 * n_cells, float_type{});
        for (unsigned i=0u; i<candidate_length.size(); ++i) {
            const auto x = in.x[i];
            const auto y = in.y[i];
            const auto z = in.z[i];
            candidate_length[i] = std::sqrt(x*x + y*y + z*z);
        }
        {   // Only keep elements that differ by more than length_threshold
            std::sort(std::begin(candidate_length), std::end(candidate_length), std::greater<float_type>{});

            if (logger::level_active<logger::l_debug>()) {
                logger::debug << stanza << "candidate_length =";
                for (const auto& e : candidate_length)
                    logger::debug << ' ' << e;
                logger::debug << ", threshold = " << conf_rt.length_threshold << '\n';                
            }
            
            unsigned i=0, j=1;
            do {
                if ((candidate_length[i] - candidate_length[j]) < conf_rt.length_threshold) {
                    logger::debug << stanza << "  ignore " << candidate_length[j] << '\n';
                } else if (++i != j) {
                    candidate_length[i] = candidate_length[j];
                }
            } while(++j != candidate_length.size());
            extra_args.num_candidate_groups = i + 1;
            logger::debug << stanza << "n_candidate_groups = " << extra_args.num_candidate_groups << '\n';
        }
        {   // call kernel
            extra_args.num_angular_steps = constant<float_type>::pi / conf_rt.angular_step;
            const unsigned n_samples = extra_args.num_angular_steps * extra_args.num_angular_steps;
            const unsigned n_blocks = (n_samples + n_threads - 1) / n_threads;
            const unsigned shared_sz = std::max(instance.cpers.num_candidate_vectors * sizeof(vec_cand_t<float_type>),
                                                sizeof(typename BlockRadixSort<float_type>::TempStorage));

            if (n_samples < instance.cpers.num_candidate_vectors)
                throw FF_EXCEPTION("fewer samples than required candidate vectors");

            gpu_state::copy_crt(state_id, conf_rt);
            gpu_state::copy_in(state_id, instance.cpers, in);
            state.start.record();
            for (unsigned i=0; i<extra_args.num_candidate_groups; ++i) {
                extra_args.candidate_length = candidate_length[i];
                extra_args.candidate_group = i;
                gpu_find_candidates<float_type><<<n_blocks, n_threads, shared_sz, 0>>>(gpu_state::ptr(state_id).get(), extra_args);
            }
            gpu_dummy_test<float_type><<<1, 1, 0, 0>>>(gpu_state::ptr(state_id).get(), extra_args);
            state.end.record();
            gpu_state::copy_out(state_id, out);
        }
        if (timing) {
            end = clock::now();
            duration elapsed = end - start;
            logger::info << stanza << "indexing_time: " << elapsed.count() << "ms\n";
            state.end.sync(); // probably unnecessary right now, since copy_out syncs on stream
            logger::info << stanza << "kernel_time: " << gpu_timing(state.start, state.end) << "ms\n";
        }
    }

    // Raw memory pin
    void pin_memory(void* ptr, std::size_t size)
    {
        CU_CHECK(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
    }

    // Raw memory unpin
    void unpin_memory(void* ptr)
    {
        CU_CHECK(cudaHostUnregister(ptr));
    }

    void* alloc_pinned(std::size_t num_bytes)
    {
        void* ptr;
        CU_CHECK(cudaHostAlloc(&ptr, num_bytes, cudaHostAllocDefault));
        return ptr;
    }

    void dealloc_pinned(void* ptr)
    {
        CU_CHECK(cudaFreeHost(ptr));
    }

    template void init<float> (const indexer<float>&);
    template void drop<float> (const indexer<float>&);
    template void index<float> (const indexer<float>&, const input<float>&, output<float>&, const config_runtime<float>&);

} // namespace gpu
