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
#include <algorithm>
#include <cmath>    // for M_PI amongst others    
#include "exception.h"
#include "log.h"
#include "indexer_gpu.h"
#include "cuda_runtime.h"
#include <cub/block/block_radix_sort.cuh>

using logger::stanza;

namespace {

    constexpr char INDEXER_GPU_DEVICE[] = "INDEXER_GPU_DEVICE";
    constexpr unsigned n_threads = 512; // num cuda threads per block for find_candidates kernel
    constexpr unsigned warp_size = 32;  // number of threads in a warp

    template<typename float_type>
    struct constant final {
        // static constexpr float_type pi = M_PI;
        // static constexpr float_type pi2 = M_PI_2;
        static constexpr float_type dl = 0.7639320225002102;    // for spiral sample points on a half sphere
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
        float_type* candidate_length;           // Candidate vector groups length, [3 * max_input_cells]
        float_type* candidate_value;            // Candidate vector objective function values, [3 * max_input_cells * num_candidate_vectors]
        unsigned* candidate_sample;             // Sample number of candidate vectors, [3 * max_input_cells * num_candidate_vectors]
        float_type* cell_score;                 // Per output cell score, [max_output_cells]
        unsigned* seq_block;                    // Per candidate vector group sequentializer for thread blocks (explicit init to 0, set to 0 in kernel after use)
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

        gpu_pointer<content_type> data;                     // Indexer_device_data on GPU
        gpu_pointer<float_type> elements;                   // Input/output vector elements of data on GPU
        gpu_pointer<float_type> candidate_length;           // Candidate vector groups length
        gpu_pointer<float_type> candidate_value;            // Candidate vectors objective function values on GPU
        gpu_pointer<unsigned> candidate_sample;             // Sample number of candidate vectors
        gpu_pointer<float_type> cell_score;                 // Cells objective function value
        gpu_pointer<unsigned> seq_block;                    // Thread block sequentializers

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
        // score    on GPU per output cell scoring function values
        // sb       on GPU per candidate vector group/output cell block sequentializers
        // x/y/zi   on GPU input pointers d->input.x / y / z
        // x/y/zo   on GPU output pointers d->output.x / y / z
        indexer_gpu_state(gpu_pointer<content_type>&& d, gpu_pointer<float_type>&& e,
                          gpu_pointer<float_type>&& cl, gpu_pointer<float_type>&& cv, gpu_pointer<unsigned>&& cs,
                          gpu_pointer<float_type>&& score, gpu_pointer<unsigned>&& sb,
                          float_type* xi, float_type* yi, float_type* zi, float_type* xo, float_type* yo, float_type* zo)
            : data{std::move(d)}, elements{std::move(e)},
              candidate_length{std::move(cl)}, candidate_value{std::move(cv)}, candidate_sample{std::move(cs)},
              cell_score{std::move(score)}, seq_block{std::move(sb)},
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
            }

            logger::debug << stanza << "copy in: " << n_cells << " cells, " << n_spots << " spots, elements=" << gpu_state.elements.get() << ": "
                          << input.x << "-->" << gpu_state.ix << ", "
                          << input.y << "-->" << gpu_state.iy << ", "
                          << input.z << "-->" << gpu_state.iz << '\n';
        }

        static inline void init_cand(const key_type& state_id, unsigned n_cand_groups, const config_persistent& cpers,
                                     std::vector<float_type>& cand_len, cudaStream_t stream=0)
        {
            const auto n_cand_vecs = n_cand_groups * cpers.num_candidate_vectors;
            const auto& gpu_state = ref(state_id);

            CU_CHECK(cudaMemsetAsync(gpu_state.candidate_value.get(), 0, n_cand_vecs * sizeof(float_type), stream));
            CU_CHECK(cudaMemcpyAsync(gpu_state.candidate_length.get(), cand_len.data(), n_cand_groups * sizeof(float_type), cudaMemcpyHostToDevice, stream));
        }

        static inline void init_score(const key_type& state_id, const config_persistent& cpers, cudaStream_t stream=0)
        {
            const auto n_cells = cpers.max_output_cells;
            const auto& gpu_state = ref(state_id);

            CU_CHECK(cudaMemsetAsync(gpu_state.cell_score.get(), 0, n_cells * sizeof(float_type), stream));
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
                CU_CHECK(cudaMemcpyAsync(output.score, gpu_state.cell_score.get(), output.n_cells * sizeof(float_type), cudaMemcpyDeviceToHost, stream));
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
        unsigned candidate_group;       // candidate group for current kernel
        unsigned num_candidate_groups;  // number of candidate vector groups
    };

    template<typename float_type>
    struct vec_cand_t final {
        float_type value;   // objective function value
        unsigned sample;    // | alpha_index | beta_index |
    };

    template<typename float_type>
    using BlockRadixSort = cub::BlockRadixSort<float_type, n_threads, 1, unsigned>;

    // -----------------------------------
    //            GPU Auxiliary
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

        __device__ __forceinline__ static float sqrt(float x)
        {
            return sqrtf(x);
        }
        
        __device__ __forceinline__ static float rem(float x, float y)
        {
            return remainderf(x, y);
        }

        __device__ __forceinline__ static float norm(float x, float y, float z)
        {   // sqrt(x*x + y*y + z*z)
            return norm3df(x, y, z);
        }

        __device__ __forceinline__ static float fma(float x, float y, float z)
        {   // x * y + z
            return fmaf(x, y, z);
        }

        __device__ __forceinline__ static void sincospi(float a, float* sinp, float* cosp)
        {
            sincospif(a, sinp, cosp);
        }

        __device__ __forceinline__ static float cospi(float a)
        {
            return cospif(a);
        }
    };

    // acquire block sequentializer in busy wait loop
    __device__ __forceinline__ void seq_acquire(unsigned& seq)
    {
        while (atomicCAS(&seq, 0u, 1u) != 0u);
    }

    // release block sequentializer
    __device__ __forceinline__ void seq_release(unsigned& seq)
    {
        __threadfence();
        __stwt(&seq, 0u);
    }

    // single threaded merge of block local sorted candidate vector array into global sorted candidate vector array
    template<typename float_type>
    __device__ __forceinline__ void merge_top(float_type* top_val, unsigned* top_sample, vec_cand_t<float_type>* cand, unsigned n_cand)
    {
        // ---- python equivalent ----
        // def a_top(A, B):
        //     if A[-1] <= B[0]:
        //         return
        //     if B[-1] <= A[0]:
        //         A[:] = B[:]
        //         return
        //     a, b0, b1, b2 = 0, 0, len(B)-1, len(B)-1
        //     while a < len(A):
        //         val = A[a]
        //         if b1 < b2:
        //             if val < B[b1]:
        //                 B[b1] = val; b1 -= 1
        //             if B[b0] < B[b2]:
        //                 A[a] = B[b0]; b0 += 1
        //             else:
        //                 A[a] = B[b2]; b2 -= 1
        //         else:
        //             if val > B[b0]:
        //                 if val < B[b1]:
        //                     B[b1] = val; b1 -= 1
        //                 else:
        //                     A[a:] = B[b0:b0+len(A)-a]
        //                     return
        //                 A[a] = B[b0]; b0 += 1
        //         a += 1
        // ----------------------------
        if (top_val[n_cand-1] <= cand[0].value)
            return;

        if (cand[n_cand-1].value <= top_val[0]) {
            for (unsigned i=0; i<n_cand; ++i) {
                top_val[i] = cand[i].value;
                top_sample[i] = cand[i].sample;
            }
            return;
        }

        unsigned top = 0;
        unsigned cand0 = 0;
        unsigned cand1 = n_cand - 1;
        unsigned cand2 = n_cand - 1;

        while (top < n_cand) {
            auto val = top_val[top];
            if (cand1 < cand2) {
                if (val < cand[cand1].value) {
                    cand[cand1].value = val;
                    cand[cand1].sample = top_sample[top];
                    --cand1;
                }
                if (cand[cand0].value < cand[cand2].value) {
                    top_val[top] = cand[cand0].value;
                    top_sample[top] = cand[cand0].sample;
                    ++cand0;
                } else {
                    top_val[top] = cand[cand2].value;
                    top_sample[top] = cand[cand2].sample;
                    --cand2;
                }
            } else if (val > cand[cand0].value) {
                if (val < cand[cand1].value) {
                    cand[cand1].value = val;
                    cand[cand1].sample = top_sample[top];
                    --cand1;
                } else {
                    for (; top<n_cand; ++top, ++cand0) {
                        top_val[top] = cand[cand0].value;
                        top_sample[top] = cand[cand0].sample;
                    }
                    return;
                }
                top_val[top] = cand[cand0].value;
                top_sample[top] = cand[cand0].sample;
                ++cand0;
            }
            ++top;
        }
    }

    // Calculate sample point on half unit sphere
    // sample_idx   index of sample point 0..n_samples
    // n_samples    number of sampling points
    // x, y, z      sample point coordinates on half unit sphere
    template<typename float_type>
    __device__ __forceinline__ void sample_point(const unsigned sample_idx, const unsigned n_samples,
                                                 float_type& x, float_type& y, float_type& z)
    {
        // Python equivalent:
        // N = n_samples
        // dl = (3. - np.sqrt(5.)) # ~2.39996323 / pi
        // dz = 1. / N
        // z = (1. - .5 * dz) - np.arange(0, N, dtype=float) * dz
        // r = np.sqrt(1. - z * z)
        // l = np.arange(0, N, dtype=float) * dl
        // x = np.cos(np.pi * l) * r
        // y = np.sin(np.pi * l) * r
        const float_type dz = 1.f / static_cast<float_type>(n_samples);
        z = util<float_type>::fma(-dz, sample_idx, float_type{1.} - float_type{.5} * dz);
        const float_type r_xy = util<float_type>::sqrt(util<float_type>::fma(-z, z, 1.));
        const float_type l = constant<float_type>::dl * static_cast<float_type>(sample_idx);
        util<float_type>::sincospi(l, &y, &x);
        x *= r_xy;
        y *= r_xy;
    }

    // -----------------------------------
    //            GPU Kernels
    // -----------------------------------

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

        // Print out candidate vectors
        const unsigned ncg = args.num_candidate_groups;
        const unsigned ncv = data->cpers.num_candidate_vectors;
        const float_type* const cvp = data->candidate_value;
        const unsigned ns = data->crt.num_sample_points;
        const unsigned* const csp = data->candidate_sample;

        for (unsigned i=0; i<ncg; ++i) {
            printf("cg%u:", i);
            auto cv = &cvp[i * ncv];
            for (unsigned j=0; j<ncv; ++j)
                printf(" %0.2f", cv[j]);
            printf("\n");
        }

        for (unsigned i=0; i<ncg; ++i) {
            auto cv = &cvp[i * ncv];
            auto cs = &csp[i * ncv];
            const float_type v = *cv;
            const unsigned s = *cs;
            float_type x, y, z;     // unit length sample vector
            sample_point(s, ns, x, y, z);
            printf("%u: v=[%f, %f, %f] # %f\n", i, x, y, z, v);
        }
    }

    template<typename float_type>
    __global__ void gpu_find_candidates(indexer_device_data<float_type>* data, indexer_args<float_type> args)
    {
        extern __shared__ double* shared_ptr[];

        unsigned sample = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned n_samples = data->crt.num_sample_points;
        const unsigned c_group = args.candidate_group;
        const float_type r = data->candidate_length[c_group];
        float_type v = 0.;  // objective function value for sample vector

        if (sample < n_samples) {   // calculate v
            float_type x, y, z;     // unit length sample vector
            sample_point(sample, n_samples, x, y, z);

            const fast_feedback::input<float_type>& in = data->input;
            const unsigned n_cells = in.n_cells;
            const unsigned n_spots = in.n_spots;
            const float_type* sx = &in.x[3u * n_cells];
            const float_type* sy = &in.y[3u * n_cells];
            const float_type* sz = &in.z[3u * n_cells];
            
            for (unsigned i=0u; i<n_spots; ++i) {
                const float_type dp = x * sx[i] + y * sy[i] + z * sz[i];
                const float_type dv = util<float_type>::cospi(float_type{2.} * dp / r);
                v -= dv;
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
            v = *key;
            sample = *val;
            __syncthreads();    // protect sort_ptr[] (it's in the same memory as cand_ptr)
        }

        auto cand_ptr = reinterpret_cast<vec_cand_t<float_type>*>(shared_ptr); // [num_candidate_vectors]
        if (threadIdx.x < num_candidate_vectors)    // store top candidate vectors into cand_ptr
            cand_ptr[threadIdx.x] = { v, sample };

        if (num_candidate_vectors < warp_size) {    // wait for cand_ptr
            if (threadIdx.x < warp_size)
                __syncwarp();
        } else
            __syncthreads();

        if (threadIdx.x == 0) { // merge block local into global result
            auto value_ptr = &data->candidate_value[args.candidate_group * num_candidate_vectors];
            auto sample_ptr = &data->candidate_sample[args.candidate_group * num_candidate_vectors];

            seq_acquire(data->seq_block[c_group]);
            merge_top(value_ptr, sample_ptr, cand_ptr, num_candidate_vectors);
            seq_release(data->seq_block[c_group]);
        }
    }

    template<typename float_type>
    __global__ void gpu_find_cells(indexer_device_data<float_type>* data)
    {

    }

} // anonymous namespace

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
        float_type* candidate_length = nullptr;
        float_type* candidate_value = nullptr;
        unsigned* candidate_sample = nullptr;
        float_type* scores = nullptr;
        unsigned* sequentializers = nullptr;
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
                std::size_t vector_dimension = std::max(3 * cpers.max_input_cells, cpers.max_output_cells);
                CU_CHECK(cudaMalloc(&sequentializers, vector_dimension * sizeof(unsigned)));
            }
            gpu_pointer<unsigned> sequentializers_ptr{sequentializers};
            {   // initialize sequentializers
                std::size_t vector_dimension = 3 * cpers.max_input_cells;
                CU_CHECK(cudaMemsetAsync(sequentializers, 0, vector_dimension * sizeof(unsigned), 0));
            }

            {
                std::size_t vector_dimension = 3 * (cpers.max_output_cells + cpers.max_input_cells) + cpers.max_spots;
                CU_CHECK(cudaMalloc(&elements, 3 * vector_dimension * sizeof(float_type)));
            }
            gpu_pointer<float_type> element_ptr{elements};

            {
                std::size_t vector_dimension = 3 * cpers.max_input_cells;
                CU_CHECK(cudaMalloc(&candidate_length, vector_dimension * sizeof(float_type)));
            }
            gpu_pointer<float_type> candidate_length_ptr{candidate_length};

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

            {
                std::size_t vector_dimension = cpers.max_output_cells;
                CU_CHECK(cudaMalloc(&scores, vector_dimension * sizeof(float_type)));
            }
            gpu_pointer<float_type> score_ptr{scores};

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
                                                     std::move(candidate_length_ptr), std::move(candidate_value_ptr), std::move(candidate_sample_ptr),
                                                     std::move(score_ptr), std::move(sequentializers_ptr),
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
                              candidate_length, candidate_value, candidate_sample,
                              scores, sequentializers};
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

        // Check input/output
        const auto n_cells_in = std::min(in.n_cells, instance.cpers.max_input_cells);
        const auto n_cells_out = instance.cpers.max_output_cells;
        logger::debug << stanza << "n_cells = " << n_cells_in << "(in)/" << n_cells_out << "(out), n_spots = " << in.n_spots << '\n';
        if (n_cells_in <= 0u)
            throw FF_EXCEPTION("no given input cells");
        if (n_cells_out <= 0)
            throw FF_EXCEPTION("no output cells");
        if (in.n_spots <= 0u)
            throw FF_EXCEPTION("no spots");
        if (instance.cpers.num_candidate_vectors <= 1u)
            throw FF_EXCEPTION("nonpositive number of candidate vectors");
        if (conf_rt.num_sample_points < instance.cpers.num_candidate_vectors)
            throw FF_EXCEPTION("fewer sample points than required candidate vectors");
        
        // Extra indexer arguments
        indexer_args<float_type> extra_args;

        // Calculate input vector groups
        std::vector<unsigned> candidate_idx(3 * n_cells_in);
        std::vector<float_type> candidate_length(3 * n_cells_in, float_type{});
        for (unsigned i=0u; i<candidate_length.size(); ++i) {
            const auto x = in.x[i];
            const auto y = in.y[i];
            const auto z = in.z[i];
            candidate_length[i] = std::sqrt(x*x + y*y + z*z);
        }
        {   // Only keep elements that differ by more than length_threshold
            std::sort(std::begin(candidate_length), std::end(candidate_length), std::greater<float_type>{});

            const float_type l_threshold = conf_rt.length_threshold;
            if (logger::level_active<logger::l_debug>()) {
                logger::debug << stanza << "candidate_length =";
                for (const auto& e : candidate_length)
                    logger::debug << ' ' << e;
                logger::debug << ", threshold = " << l_threshold << '\n';                
            }
            
            unsigned i=0, j=1;
            do {
                if ((candidate_length[i] - candidate_length[j]) < l_threshold) {
                    logger::debug << stanza << "  ignore " << candidate_length[j] << '\n';
                } else if (++i != j) {
                    candidate_length[i] = candidate_length[j];
                }
            } while(++j != candidate_length.size());
            extra_args.num_candidate_groups = i + 1;
            for (unsigned i=0u; i<candidate_idx.size(); ++i) {
                const auto x = in.x[i];
                const auto y = in.y[i];
                const auto z = in.z[i];
                float_type length = std::sqrt(x*x + y*y + z*z);
                auto it = std::lower_bound(std::cbegin(candidate_length), std::cbegin(candidate_length)+extra_args.num_candidate_groups, length,
                                        [l_threshold](const float_type& a, const float_type& l) -> bool {
                                                return (a - l) >= l_threshold;
                                        });
                candidate_idx[i] = it - std::cbegin(candidate_length);
            }
            if (logger::level_active<logger::l_debug>()) {
                logger::debug << stanza << "candidate_idx =";
                for (const auto& e : candidate_idx)
                    logger::debug << ' ' << e;
                logger::debug << ", n_cand_groups = " << extra_args.num_candidate_groups << '\n';
            }
        }
        {   // find vector candidates
            const unsigned n_samples = conf_rt.num_sample_points;
            const unsigned n_blocks = (n_samples + n_threads - 1) / n_threads;
            const unsigned shared_sz = std::max(instance.cpers.num_candidate_vectors * sizeof(vec_cand_t<float_type>),
                                                sizeof(typename BlockRadixSort<float_type>::TempStorage));
            gpu_state::copy_crt(state_id, conf_rt);
            gpu_state::copy_in(state_id, instance.cpers, in);
            gpu_state::init_cand(state_id, extra_args.num_candidate_groups, instance.cpers, candidate_length);
            state.start.record();
            for (unsigned i=0; i<extra_args.num_candidate_groups; ++i) {
                extra_args.candidate_group = i;
                gpu_find_candidates<float_type><<<n_blocks, n_threads, shared_sz, 0>>>(gpu_state::ptr(state_id).get(), extra_args);
            }
        }
        {   // find cells
            const unsigned n_samples = (1.5 * std::sqrt(conf_rt.num_sample_points) + n_threads - 1.) / n_threads;
            const dim3 n_blocks(n_samples, n_cells_in);
            const unsigned shared_sz = n_cells_out * 4 * sizeof(float_type);    // score, x, y, z

            gpu_state::init_score(state_id, instance.cpers);
            gpu_find_cells<float_type><<<n_blocks, n_threads, shared_sz, 0>>>(gpu_state::ptr(state_id).get());
            state.end.record();
            gpu_dummy_test<float_type><<<1, 1, 0, 0>>>(gpu_state::ptr(state_id).get(), extra_args);
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
