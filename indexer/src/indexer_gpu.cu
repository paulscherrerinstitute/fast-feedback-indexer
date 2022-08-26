#include <iostream>
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <map>
#include <memory>
#include "exception.h"
#include "log.h"
#include "indexer_gpu.h"
#include "cuda_runtime.h"

using logger::stanza;

namespace {

    constexpr char INDEXER_GPU_DEVICE[] = "INDEXER_GPU_DEVICE";

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

    // On GPU data for indexer
    template<typename float_type>
    struct indexer_device_data final {
        fast_feedback::config_persistent<float_type> cpers;
        fast_feedback::config_runtime<float_type> crt;
        fast_feedback::input<float_type> input;
        fast_feedback::output<float_type> output;
    };

    // Indexer GPU state representation on the Host side
    template<typename float_type>
    struct indexer_gpu_state final {
        using content_type = indexer_device_data<float_type>;
        using key_type = fast_feedback::state_id::type;
        using map_type = std::map<key_type, indexer_gpu_state>;
        using config_persistent = fast_feedback::config_persistent<float_type>;
        using memory_pin = fast_feedback::memory_pin;

        gpu_pointer<content_type> data;     // indexer_device_data on GPU
        gpu_pointer<float_type> elements;   // Input and output vector elements of data on GPU

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

        static std::mutex state_update; // Protect per indexer state map
        static map_type dev_ptr;        // Per indexer state map

        // Create hostside on GPU state representation with
        // d        on GPU state pointer
        // e        on GPU input and output elements pointed to by *d
        // x/y/zi   on GPU input pointers d->input.x / y / z
        // x/y/zo   on GPU output pointers d->output.x / y / z
        indexer_gpu_state(gpu_pointer<content_type>&& d, gpu_pointer<float_type>&& e, float* xi, float* yi, float* zi, float* xo, float* yo, float* zo)
            : data{std::move(d)}, elements{std::move(e)}, ix{xi}, iy{yi}, iz{zi}, ox{xo}, oy{yo}, oz{zo}
        {
            tmp = fast_feedback::alloc_pinned<config_persistent>();
        }

        indexer_gpu_state() = default;                                      // Default with some uninitialized pointers
        indexer_gpu_state(const indexer_gpu_state&) = delete;
        indexer_gpu_state(indexer_gpu_state&&) = default;                   // Take over state representation
        indexer_gpu_state& operator=(const indexer_gpu_state&) = delete;
        indexer_gpu_state& operator=(indexer_gpu_state&&) = default;        // Take over state representation
        ~indexer_gpu_state() = default;                                     // Drop on GPU state data

        // Shortcut to get at state data pointer
        static inline gpu_pointer<content_type>& ptr(const key_type& id)
        {
            return dev_ptr[id].data;
        }

        // Copy runtime configuration to GPU
        static inline void copy_crt(const key_type& state_id, const fast_feedback::config_runtime<float_type>& crt, cudaStream_t stream=0)
        {
            CU_CHECK(cudaMemcpyAsync(&ptr(state_id)->crt, &crt, sizeof(crt), cudaMemcpyHostToDevice, stream));
        }

        // Copy input data to GPU
        static inline void copy_in(const key_type& state_id, const config_persistent& cpers,
                                   const fast_feedback::input<float_type>& input, cudaStream_t stream=0)
        {
            const auto& gpu_state = dev_ptr[state_id];
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
                CU_CHECK(cudaStreamSynchronize(stream));
            }

            logger::debug << stanza << "copy in: " << n_cells << " cells, " << n_spots << " spots, elements=" << gpu_state.elements.get() << ": "
                          << input.x << "-->" << gpu_state.ix << ", "
                          << input.y << "-->" << gpu_state.iy << ", "
                          << input.z << "-->" << gpu_state.iz << '\n';
        }

        // Copy output data from GPU
        static inline void copy_out(const key_type& state_id, fast_feedback::output<float_type>& output, cudaStream_t stream=0)
        {
            const auto& gpu_state = dev_ptr[state_id];
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

    // -----------------------------------
    //            GPU Kernels
    // -----------------------------------

    // Dummy kernel for dummy indexer test: copy first input cell to output cell
    template<typename float_type>
    __global__ void gpu_index(indexer_device_data<float_type>* data)
    {
        // NOTE: assume max_output_cells is >= 1u
        for (unsigned i=0; i<3u; ++i) {
            data->output.x[i] = data->input.x[i];
            data->output.y[i] = data->input.y[i];
            data->output.z[i] = data->input.z[i];
        }
        data->output.n_cells = 1u;
    }
}

namespace gpu {

    // Check for an initialized CUDA runtime
    #define CU_CHECK_INIT {                                      \
        if (! cuda_initialized.load())                           \
            throw FF_EXCEPTION("cuda runtime not initialized");  \
    }

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
                gpu_state::dev_ptr[state_id] = gpu_state{std::move(data_ptr), std::move(element_ptr), ix, iy, iz, ox, oy, oz};
            }

            logger::debug << stanza << "init id=" << state_id << ", elements=" << elements << '\n'
                          << stanza << "     ox=" << ox << ", oy=" << oy << ", oz=" << oz << '\n'
                          << stanza << "     ix=" << ix << ", iy=" << iy << ", iz=" << iz << '\n';
        }

        CU_CHECK(cudaMemcpy(&data->cpers, &cpers, sizeof(cpers), cudaMemcpyHostToDevice));

        {
            fast_feedback::output<float_type> output{ox, oy, oz, 0};
            CU_CHECK(cudaMemcpy(&data->output, &output, sizeof(output), cudaMemcpyHostToDevice));

            fast_feedback::input<float_type> input{ix, iy, iz, 0, 0};
            CU_CHECK(cudaMemcpy(&data->input, &input, sizeof(input), cudaMemcpyHostToDevice));
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

        CU_CHECK_INIT;
        auto state_id = instance.state;
        gpu_state::copy_crt(state_id, conf_rt);
        gpu_state::copy_in(state_id, instance.cpers, in);
        gpu_index<float_type><<<1, 1, 0, 0>>>(gpu_state::ptr(state_id).get());
        gpu_state::copy_out(state_id, out);
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
