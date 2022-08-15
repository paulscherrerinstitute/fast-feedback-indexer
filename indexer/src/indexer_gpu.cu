#include <iostream>
#include <atomic>
#include <mutex>
#include <vector>
#include "exception.h"
#include "indexer_gpu.h"
#include "cuda_runtime.h"

namespace {

    // Cuda device infos
    struct gpu_device final {
        int id;
        cudaDeviceProp prop;

        // Device list
        static std::vector<gpu_device> list;
    };

    std::vector<gpu_device> gpu_device::list;

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

    // Is the cuda runtime initialized?
    std::atomic_bool cuda_initialized = false;

    // Protect cuda runtime initalisation
    std::mutex cuda_init_lock;

    void cuda_init()
    {
        if (cuda_initialized.load())
            return;
        
        {
            std::lock_guard<std::mutex> lock(cuda_init_lock);

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

            cuda_initialized.store(true);
        } // release cuda_init_lock
    }
}

namespace gpu {

    #define CU_CHECK_INIT {                                      \
        if (! cuda_initialized.load())                           \
            throw FF_EXCEPTION("cuda runtime not initialized");  \
    }

    template <typename float_type>
    void init (const indexer<float_type>& instance)
    {
        cuda_init();
    }

    template <typename float_type>
    void drop (const indexer<float_type>& instance)
    {
        CU_CHECK_INIT;
    }

    template <typename float_type>
    void index (const indexer<float_type>& instance, const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt)
    {
        CU_CHECK_INIT;
        std::cout << "hello: " << sizeof(float_type) << '\n';
    }

    template void init<float> (const indexer<float>&);
    template void drop<float> (const indexer<float>&);
    template void index<float> (const indexer<float>&, const input<float>&, output<float>&, const config_runtime<float>&);

} // namespace gpu
