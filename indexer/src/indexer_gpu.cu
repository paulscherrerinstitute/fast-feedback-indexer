#include <iostream>
#include <atomic>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <sstream>
#include "exception.h"
#include "indexer_gpu.h"
#include "cuda_runtime.h"

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

    int gpu_device_number;

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

            CU_CHECK(cudaGetDevice(&gpu_device_number));
            {
                char* dev_string = std::getenv(INDEXER_GPU_DEVICE);
                if (dev_string != nullptr) {
                    std::istringstream iss(dev_string);
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
