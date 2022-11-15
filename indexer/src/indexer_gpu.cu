/*
Copyright 2022 Paul Scherrer Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

------------------------

Author: hans-christian.stadler@psi.ch
*/

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
#include <cfloat>   // for FLT_EPSILON and others
#include "exception.h"
#include "log.h"
#include "indexer_gpu.h"
#include "cuda_runtime.h"
#include <cub/block/block_radix_sort.cuh>

using logger::stanza;

namespace {

    constexpr char INDEXER_GPU_DEVICE[] = "INDEXER_GPU_DEVICE";
    constexpr char INDEXER_GPU_DEBUG[] = "INDEXER_GPU_DEBUG";
    constexpr unsigned n_threads = 1024;    // num cuda threads per block for find_candidates kernel
    constexpr unsigned warp_size = 32;      // number of threads in a warp

    template<typename float_type>
    struct constant final {
        // static constexpr float_type pi = M_PI;
        #pragma nv_diag_suppress 177 // suppress unused warning
        [[maybe_unused]] static constexpr float_type pi2 = M_PI_2;
        #pragma nv_diag_suppress 177 // suppress unused warning
        [[maybe_unused]] static constexpr float_type dl = 0.76393202250021030359082633126873; // 3 - sqrt(5), for spiral sample points on a half sphere
        // static constexpr float_type eps = FLT_EPSILON;
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
    bool gpu_debug_output;  // Print gpu debug output if true

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

                char* debug_string = std::getenv(INDEXER_GPU_DEBUG);
                if (debug_string != nullptr) {
                    const std::vector<std::string> accepted = {"1", "true", "yes", "on", "0", "false", "no", "off"}; // [4] == "0"
                    unsigned i;
                    for (i=0u; i<accepted.size(); i++) {
                        if (accepted[i] == debug_string) {
                            gpu_debug_output = (i < 4);                                                              // [4] == "0"
                            break;
                        }
                    }
                    if (i >= accepted.size()) {
                        std::ostringstream oss;
                        oss << "illegal value for " << INDEXER_GPU_DEBUG << ": \"" << debug_string << "\" (use one of";
                        for (const auto& s : accepted)
                            oss << " \"" << s << '\"';
                        oss << ')';
                        throw FF_EXCEPTION(oss.str());
                    }
                } else
                    gpu_debug_output = false;
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
        unsigned* cellvec_to_cand;              // Input cell vector to candidate group mapping, [3 * max_input_cells]
        unsigned* seq_block;                    // Per candidate vector group sequentializer for thread blocks (explicit init to 0, set to 0 in kernel after use)
                                                //     First sequentializer is also used for candidate cell search
    };

    // Cuda stream wrapper
    struct gpu_stream final {
        bool ready;                             // Is stream ready = initialized
        cudaStream_t stream;                    // Cuda stream
        static std::mutex unused_update;        // Protect unused streams container
        static std::map<unsigned int, std::vector<gpu_stream>> unused; // Stock of unused streams per flags

        // Return uninitialized stream
        gpu_stream() noexcept
            : ready{false}, stream{cudaStream_t{}}
        {}

        // Initialize stream with existing cuda stream
        // The cuda stream given as argument will be managed by this object after the call,
        // which basically means it should not be used afterwards.
        gpu_stream(cudaStream_t s)
            : ready{true}, stream{s}
        {
            unsigned int flags;
            CU_CHECK(cudaStreamGetFlags(s, &flags));    // check that the s is initialized
        }

        // Move s into this
        gpu_stream(gpu_stream&& s)
            : ready{false}, stream{cudaStream_t{}}
        {
            std::swap(ready, s.ready);
            std::swap(stream, s.stream);
        }

        // Move s into this
        gpu_stream& operator=(gpu_stream&& s)
        {
            if (ready) {
                CU_CHECK(cudaStreamDestroy(stream));
                ready = false;
            }
            std::swap(ready, s.ready);
            std::swap(stream, s.stream);
            return *this;
        }

        gpu_stream(const gpu_stream&) = delete;             // No copying
        gpu_stream& operator=(const gpu_stream&) = delete;  // No copying

        // Destroy this
        ~gpu_stream()
        {
            release();
        }

        // Release and unititialize the cuda stream
        // This is left in uninitialized state
        void release()
        {
            if (ready) {
                CU_CHECK(cudaStreamDestroy(stream));
                ready = false;
            }
        }

        // (Re)initialize the cuda stream
        void init(unsigned int flags=cudaStreamDefault)
        {
            release();
            CU_CHECK(cudaStreamCreateWithFlags(&stream, flags));
            ready = true;
        }

        // Sync with cuda stream
        void sync()
        {
            if (ready)
                CU_CHECK(cudaStreamSynchronize(stream));
        }

        // Get an unused gpu_stream from cache
        static gpu_stream from_cache(unsigned int flags=cudaStreamDefault)
        {
            std::lock_guard lock_unused{unused_update};
            if (unused[flags].empty()) {
                unused[flags].emplace(std::end(unused[flags]));
                unused[flags].back().init(flags);
            }
            gpu_stream result = std::move(unused[flags].back());
            unused[flags].pop_back();
            return result;
        }

        // Return an initialized gpu_stream to the cache of unused streams
        static void to_cache(gpu_stream&& s)
        {
            unsigned int flags;
            CU_CHECK(cudaStreamGetFlags(s, &flags));
            std::lock_guard lock_unused{unused_update};
            unused[flags].insert(std::end(unused[flags]), std::move(s));
        }

        operator cudaStream_t&() noexcept { return stream; }    // Cast to cuda stream
    };

    std::mutex gpu_stream::unused_update;
    std::map<unsigned int, std::vector<gpu_stream>> gpu_stream::unused;

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
        gpu_pointer<unsigned> cellvec_to_cand;              // Input cell vector to candidate group mapping
        gpu_pointer<unsigned> seq_block;                    // Thread block sequentializers
        gpu_stream init_stream;                             // CUDA stream for initialization

        // Temporary pinned space to transfer n_input_cells, n_output_cells, n_spots
        fast_feedback::pinned_ptr<config_persistent> tmp;   // Pointer to make move construction simple

        // Input cell coordinates on GPU pointing into elements
        float_type* ix;
        float_type* iy;
        float_type* iz;

        // Output cell coordinates and scores on GPU pointing into elements
        float_type* ox;
        float_type* oy;
        float_type* oz;
        float_type* cell_score;

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
                          gpu_pointer<float_type>&& cl, gpu_pointer<float_type>&& cv,
                          gpu_pointer<unsigned>&& cs, gpu_pointer<unsigned>&& v2c,
                          gpu_pointer<unsigned>&& sb, gpu_stream&& stream,
                          float_type* xi, float_type* yi, float_type* zi,
                          float_type* xo, float_type* yo, float_type* zo,
                          float_type* scores)
            : data{std::move(d)}, elements{std::move(e)},
              candidate_length{std::move(cl)}, candidate_value{std::move(cv)},
              candidate_sample{std::move(cs)}, cellvec_to_cand{std::move(v2c)},
              seq_block{std::move(sb)}, init_stream{std::move(stream)},
              ix{xi}, iy{yi}, iz{zi},
              ox{xo}, oy{yo}, oz{zo},
              cell_score{scores}
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

        // Shortcut to cuda stream
        static inline gpu_stream& stream(const key_type& id)
        {
            return ref(id).init_stream;
        }

        // State exists
        static inline bool exists(const key_type& id)
        {
            std::lock_guard<std::mutex> state_lock{state_update};
            return dev_ptr.find(id) != std::end(dev_ptr);
        }

        // Drop state
        static inline void drop(const key_type& id)
        {
            std::lock_guard<std::mutex> state_lock{state_update};
            dev_ptr.erase(id);
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
                                   const fast_feedback::input<float_type>& input,
                                   fast_feedback::output<float_type>& output,
                                   cudaStream_t stream=0)
        {
            const auto& gpu_state = ref(state_id);
            auto& pinned_tmp = const_cast<config_persistent&>(*gpu_state.tmp);
            const auto& device_data = gpu_state.data;
            const auto n_input_cells = std::min(input.n_cells, cpers.max_input_cells);
            const auto n_spots = std::min(input.n_spots, cpers.max_spots);
            output.n_cells = std::min(output.n_cells, cpers.max_output_cells);

            pinned_tmp.max_output_cells = output.n_cells;
            CU_CHECK(cudaMemcpyAsync(&device_data->output.n_cells, &pinned_tmp.max_output_cells, sizeof(output.n_cells), cudaMemcpyHostToDevice, stream));
            // NOTE: the following code assumes consecutive storage of two data members in tmp and in device_data->input
            pinned_tmp.max_input_cells = n_input_cells;
            pinned_tmp.max_spots = n_spots;
            CU_CHECK(cudaMemcpyAsync(&device_data->input.n_cells, &pinned_tmp.max_input_cells, sizeof(n_input_cells) + sizeof(n_spots), cudaMemcpyHostToDevice, stream));

            if (n_input_cells + n_spots > 0u) {
                if (n_input_cells == input.n_cells) {
                    const auto combi_sz = (3u * n_input_cells + n_spots) * sizeof(float_type);
                    CU_CHECK(cudaMemcpyAsync(gpu_state.ix, input.x, combi_sz, cudaMemcpyHostToDevice, stream));
                    CU_CHECK(cudaMemcpyAsync(gpu_state.iy, input.y, combi_sz, cudaMemcpyHostToDevice, stream));
                    CU_CHECK(cudaMemcpyAsync(gpu_state.iz, input.z, combi_sz, cudaMemcpyHostToDevice, stream));
                } else {
                    const auto dst_offset = 3u * n_input_cells;
                    if (n_input_cells > 0u) {
                        const auto cell_sz = dst_offset * sizeof(float_type);
                        CU_CHECK(cudaMemcpyAsync(gpu_state.ix, input.x, cell_sz, cudaMemcpyHostToDevice, stream));
                        CU_CHECK(cudaMemcpyAsync(gpu_state.iy, input.y, cell_sz, cudaMemcpyHostToDevice, stream));
                        CU_CHECK(cudaMemcpyAsync(gpu_state.iz, input.z, cell_sz, cudaMemcpyHostToDevice, stream));
                    }
                    if (n_spots > 0) {
                        const auto src_offset = 3u * input.n_cells;
                        const auto spot_sz = n_spots * sizeof(float_type);
                        CU_CHECK(cudaMemcpyAsync(&gpu_state.ix[dst_offset], &input.x[src_offset], spot_sz, cudaMemcpyHostToDevice, stream));
                        CU_CHECK(cudaMemcpyAsync(&gpu_state.iy[dst_offset], &input.y[src_offset], spot_sz, cudaMemcpyHostToDevice, stream));
                        CU_CHECK(cudaMemcpyAsync(&gpu_state.iz[dst_offset], &input.z[src_offset], spot_sz, cudaMemcpyHostToDevice, stream));
                    }
                }
            }

            logger::debug << stanza << "copy in: " << n_input_cells << " cells(in), " << output.n_cells << " cells(out), "
                          << n_spots << " spots, elements=" << gpu_state.elements.get() << ": "
                          << input.x << "-->" << gpu_state.ix << ", "
                          << input.y << "-->" << gpu_state.iy << ", "
                          << input.z << "-->" << gpu_state.iz << '\n';
        }

        static inline void init_cand(const key_type& state_id, unsigned n_cand_groups, const config_persistent& cpers,
                                     std::vector<float_type>& cand_len, std::vector<unsigned>& cand_idx, cudaStream_t stream=0)
        {
            const auto n_cand_vecs = n_cand_groups * cpers.num_candidate_vectors;
            const auto& gpu_state = ref(state_id);

            CU_CHECK(cudaMemsetAsync(gpu_state.candidate_value.get(), 0, n_cand_vecs * sizeof(float_type), stream));
            CU_CHECK(cudaMemcpyAsync(gpu_state.candidate_length.get(), cand_len.data(), n_cand_groups * sizeof(float_type), cudaMemcpyHostToDevice, stream));
            CU_CHECK(cudaMemcpyAsync(gpu_state.cellvec_to_cand.get(), cand_idx.data(), cand_idx.size() * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
        }

        static inline void init_score(const key_type& state_id, const config_persistent& cpers, cudaStream_t stream=0)
        {
            const auto n_cells = cpers.max_output_cells;
            const auto& gpu_state = ref(state_id);

            CU_CHECK(cudaMemsetAsync(gpu_state.cell_score, 0, n_cells * sizeof(float_type), stream));
        }

        // Copy output data from GPU
        // This is a blocking call that synchronizes on stream
        static inline void copy_out(const key_type& state_id, fast_feedback::output<float_type>& output, cudaStream_t stream=0)
        {
            const auto& gpu_state = ref(state_id);
            auto& pinned_tmp = const_cast<config_persistent&>(*gpu_state.tmp);
            const auto& device_data = gpu_state.data;

            CU_CHECK(cudaMemcpyAsync(&pinned_tmp.max_output_cells, &device_data->output.n_cells, sizeof(output.n_cells), cudaMemcpyDeviceToHost, stream));
            CU_CHECK(cudaStreamSynchronize(stream));
            output.n_cells = pinned_tmp.max_output_cells;

            if (output.n_cells > 0u) {
                const auto cell_sz = 3u * output.n_cells * sizeof(float_type);
                CU_CHECK(cudaMemcpyAsync(output.x, gpu_state.ox, cell_sz, cudaMemcpyDeviceToHost, stream));
                CU_CHECK(cudaMemcpyAsync(output.y, gpu_state.oy, cell_sz, cudaMemcpyDeviceToHost, stream));
                CU_CHECK(cudaMemcpyAsync(output.z, gpu_state.oz, cell_sz, cudaMemcpyDeviceToHost, stream));
                CU_CHECK(cudaMemcpyAsync(output.score, gpu_state.cell_score, output.n_cells * sizeof(float_type), cudaMemcpyDeviceToHost, stream));
                CU_CHECK(cudaStreamSynchronize(stream));
            }

            logger::debug << stanza << "copy out: " << output.n_cells << " cells, elements=" << gpu_state.elements.get() << ": "
                          << gpu_state.ox << "-->" << output.x << ", "
                          << gpu_state.oy << "-->" << output.y << ", "
                          << gpu_state.oz << "-->" << output.z << ", "
                          << gpu_state.cell_score << "-->" << output.score << '\n';
        }
    };

    template<> std::mutex indexer_gpu_state<float>::state_update{};
    template<> indexer_gpu_state<float>::map_type indexer_gpu_state<float>::dev_ptr{};

    template<typename float_type>
    struct vec_cand_t final {
        float_type value;   // objective function value
        unsigned sample;    // sample point index
    };

    template<typename float_type>
    struct cell_cand_t final {
        float_type value;   // objective function value
        unsigned vsample;   // sample point index
        unsigned rsample;   // sample rotation angle index
        unsigned cell_vec;  // cell vector index
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
        __device__ __forceinline__ static float abs(float val)
        {
            return fabsf(val);
        }

        __device__ __forceinline__ static void sincos(float angle, float* sine, float* cosine)
        {
            return sincosf(angle, sine, cosine);
        }

        __device__ __forceinline__ static float cos(float angle)
        {
            return cosf(angle);
        }

        __device__ __forceinline__ static float acos(float angle)
        {
            return acosf(angle);
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

        __device__ __forceinline__ static float rnorm(float x, float y, float z)
        {   // 1 / sqrt(x*x + y*y + z*z)
            return rnorm3df(x, y, z);
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

    // a 🞄 a
    template<typename float_type>
    __device__ __forceinline__ float_type norm2(const float_type a[3])
    {
        return util<float_type>::fma(a[0], a[0], util<float_type>::fma(a[1], a[1], a[2] * a[2]));
    }

    // a 🞄 b
    template<typename float_type>
    __device__ __forceinline__ float_type dot(const float_type a[3], const float_type b[3])
    {
        return util<float_type>::fma(a[0], b[0], util<float_type>::fma(a[1], b[1], a[2] * b[2]));
    }

    // a = b X c
    template<typename float_type>
    __device__ __forceinline__ void cross(float_type a[3], const float_type b[3], const float_type c[3])
    {
        a[0] = util<float_type>::fma(b[1], c[2], -b[2] * c[1]);
        a[1] = util<float_type>::fma(b[2], c[0], -b[0] * c[2]);
        a[2] = util<float_type>::fma(b[0], c[1], -b[1] * c[0]);
    }

    // a = (a + l * b); a /= |a|
    template<typename float_type>
    __device__ __forceinline__ void add_unify(float_type a[3], const float_type b[3], const float_type l)
    {
        for (unsigned i=0u; i<3u; i++)
            a[i] = util<float_type>::fma(l, b[i], a[i]);
        const float_type f = util<float_type>::rnorm(a[0], a[1], a[2]);
        for (unsigned i=0u; i<3u; i++)
            a[i] *= f;
    }

    // a = 2 * (a 🞄 b) * b - a
    // pre: |b| == 1
    template<typename float_type>
    __device__ __forceinline__ void mirror(float_type a[3], float_type b[3])
    {
        // const float_type p2 = float_type{2.f} * dot(a, b);
        const float_type p2 = float_type{2.f} * dot(a, b);
        for (unsigned i=0u; i<3u; i++)
            a[i] = util<float_type>::fma(p2, b[i], -a[i]);
    }

    // laz = a 🞄 z
    // x = a - laz * z
    // x /= |x|
    // laxy = a 🞄 x
    // pre: |z| == 1
    template<typename float_type>
    __device__ __forceinline__ void project_unify(float_type x[3], const float_type a[3], const float_type z[3],
                                                  float_type &laz, float_type &laxy)
    {
        laz = dot(a, z);
        for (unsigned i=0u; i<3u; i++)
            x[i] = util<float_type>::fma(-laz, z[i], a[i]);
        const float_type f = util<float_type>::rnorm(x[0], x[1], x[2]);
        for (unsigned i=0u; i<3u; i++)
            x[i] *= f;
        laxy = dot(a, x);
    }

    // a = laz * z + (cos(alpha) * x + sin(alpha) * y) * laxy
    template<typename float_type>
    __device__ __forceinline__ void rotate(float_type a[3],
                                           const float_type x[3], const float_type y[3], const float_type z[3],
                                           const float_type laz, const float_type laxy,
                                           const float_type alpha)
    {
        float_type s, c;
        util<float_type>::sincos(alpha, &s, &c);
        for (unsigned i=0u; i<3u; i++)
            a[i] = util<float_type>::fma(laz, z[i], (util<float_type>::fma(c, x[i], s * y[i]) * laxy));
    }

    // single threaded merge of block local sorted candidate vector array into global sorted candidate vector array
    template<typename float_type>
    __device__ __forceinline__ void merge_top_vecs(float_type* top_val, unsigned* top_sample, vec_cand_t<float_type>* cand, unsigned n_cand)
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

    // Single threaded merge of block local sorted candidate cell array into global sorted output cell array.
    // Misuse output cell vector as storage for { score[i]=value, x[i]=vsample, y[i]=rsample, z[i]=cell_vector_index }
    // Run a separate kernel to expand vsample, rsample, cell_vector_index into cell vectors
    template<typename float_type>
    __device__ __forceinline__ void merge_top_cells(fast_feedback::output<float_type>& out, cell_cand_t<float_type>* cand, unsigned n_cand)
    {
        if (out.score[n_cand-1] <= cand[0].value)
            return;

        if (cand[n_cand-1].value <= out.score[0]) {
            for (unsigned i=0u; i<n_cand; i++) {
                out.score[i] = cand[i].value;
                *reinterpret_cast<unsigned*>(&out.x[3u * i]) = cand[i].vsample;
                *reinterpret_cast<unsigned*>(&out.y[3u * i]) = cand[i].rsample;
                *reinterpret_cast<unsigned*>(&out.z[3u * i]) = cand[i].cell_vec;
            }
            return;
        }

        unsigned top = 0;
        unsigned cand0 = 0;
        unsigned cand1 = n_cand - 1;
        unsigned cand2 = n_cand - 1;

        while (top < n_cand) {
            auto val = out.score[top];
            if (cand1 < cand2) {
                if (val < cand[cand1].value) {
                    cand[cand1].value = val;
                    cand[cand1].vsample = *reinterpret_cast<unsigned*>(&out.x[3u * top]);
                    cand[cand1].rsample = *reinterpret_cast<unsigned*>(&out.y[3u * top]);
                    cand[cand1].cell_vec = *reinterpret_cast<unsigned*>(&out.z[3u * top]);
                    --cand1;
                }
                if (cand[cand0].value < cand[cand2].value) {
                    out.score[top] = cand[cand0].value;
                    *reinterpret_cast<unsigned*>(&out.x[3u * top]) = cand[cand0].vsample;
                    *reinterpret_cast<unsigned*>(&out.y[3u * top]) = cand[cand0].rsample;
                    *reinterpret_cast<unsigned*>(&out.z[3u * top]) = cand[cand0].cell_vec;
                    ++cand0;
                } else {
                    out.score[top] = cand[cand2].value;
                    *reinterpret_cast<unsigned*>(&out.x[3u * top]) = cand[cand2].vsample;
                    *reinterpret_cast<unsigned*>(&out.y[3u * top]) = cand[cand2].rsample;
                    *reinterpret_cast<unsigned*>(&out.z[3u * top]) = cand[cand2].cell_vec;
                    --cand2;
                }
            } else if (val > cand[cand0].value) {
                if (val < cand[cand1].value) {
                    cand[cand1].value = val;
                    cand[cand1].vsample = *reinterpret_cast<unsigned*>(&out.x[3u * top]);
                    cand[cand1].rsample = *reinterpret_cast<unsigned*>(&out.y[3u * top]);
                    cand[cand1].cell_vec = *reinterpret_cast<unsigned*>(&out.z[3u * top]);
                    --cand1;
                } else {
                    for (; top<n_cand; ++top, ++cand0) {
                        out.score[top] = cand[cand0].value;
                        *reinterpret_cast<unsigned*>(&out.x[3u * top]) = cand[cand0].vsample;
                        *reinterpret_cast<unsigned*>(&out.y[3u * top]) = cand[cand0].rsample;
                        *reinterpret_cast<unsigned*>(&out.z[3u * top]) = cand[cand0].cell_vec;
                    }
                    return;
                }
                out.score[top] = cand[cand0].value;
                *reinterpret_cast<unsigned*>(&out.x[3u * top]) = cand[cand0].vsample;
                *reinterpret_cast<unsigned*>(&out.y[3u * top]) = cand[cand0].rsample;
                *reinterpret_cast<unsigned*>(&out.z[3u * top]) = cand[cand0].cell_vec;
                ++cand0;
            }
            ++top;
        }
    }

    // Calculate sample point on half unit sphere
    // sample_idx   index of sample point 0..n_samples
    // n_samples    number of sampling points
    // v            sample point coordinates on half unit sphere multiplied by factor
    template<typename float_type>
    __device__ __forceinline__ void sample_point(const unsigned sample_idx, const unsigned n_samples, float_type v[3])
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

        // Numerically problematic:
        // float_type& x = v[0];
        // float_type& y = v[1];
        // float_type& z = v[2];
        // const float_type dz = float_type{1.} / static_cast<float_type>(n_samples);
        // z = util<float_type>::fma(-dz, sample_idx, util<float_type>::fma(float_type{-.5}, dz, float_type{1.}));
        // const float_type r_xy = util<float_type>::sqrt(util<float_type>::fma(-z, z, 1.));
        // const float_type l = constant<float_type>::dl * static_cast<float_type>(sample_idx);
        // util<float_type>::sincospi(l, &y, &x);
        // x *= r_xy;
        // y *= r_xy;

        double x, y, z;
        double sample_d = static_cast<double>(sample_idx);
        const double dz = double{1.} / static_cast<double>(n_samples);
        z = fma(-dz, sample_d, fma(double{-.5}, dz, double{1.})); // (double{1.} - double{.5} * dz) - sample_d * dz;
        const double r_xy = sqrt(fma(-z, z, double{1.}));
        const double l = constant<double>::dl * sample_d;
        sincospi(l, &y, &x);
        x *= r_xy;
        y *= r_xy;
        v[0] = static_cast<float_type>(x);
        v[1] = static_cast<float_type>(y);
        v[2] = static_cast<float_type>(z);
    }

    // Get sample cell vectors a, b, and unified c
    // z            sample cell vector c scaled to unit length
    // a            sample cell vector a
    // b            sample cell vector b
    // cx           input cell vectors x coordinates
    // cy           input cell vectors y coordinates
    // cz           input cell vectors z coordinates
    // vlength      length of sample cell vector c
    // vsample      sample point index of sample cell vector c [0 .. n_vsamples[
    // n_vsamples   number of sample points on the half sphere
    // rsample      sample rotation angle index of sample cell [0 .. n_rsamples[
    // n_rsamples   number of sample angles around sample cell vector c
    // cell_vec     input cell vector index [0 .. 3*n_input_cells[
    template<typename float_type>
    __device__ __forceinline__ void sample_cell(float_type z[3], float_type a[3], float_type b[3],
                                                const float_type* cx, const float_type* cy, const float_type* cz,                                                
                                                const float_type vlength,
                                                const unsigned vsample, const unsigned n_vsamples,
                                                const unsigned rsample, const unsigned n_rsamples,
                                                const unsigned cell_vec)
    {
        const unsigned cell_base = cell_vec / 3u;
        float_type t[3] = { cx[cell_vec], cy[cell_vec], cz[cell_vec] };
        sample_point(vsample, n_vsamples, z);
        // Align cell to sample vector z by mirroring on z + t
        add_unify(t, z, vlength);
        unsigned idx = cell_base + (cell_vec + 1u) % 3u;
        a[0] = cx[idx]; a[1] = cy[idx]; a[2] = cz[idx];
        idx = cell_base + (cell_vec + 2u) % 3u;
        b[0] = cx[idx]; b[1] = cy[idx]; b[2] = cz[idx];
        mirror(a, t);
        mirror(b, t);
        // Basis perpendicular to z and projections of a/b to z, xy
        float_type x[3], laz, laxy;         // unit vector x, length of a projected to z and xy plane ⊥ v
        float_type y[3], lbz, lbxy;         // unit vector y, length of b projected to z and xy plane ⊥ v
        project_unify(x, a, z, laz, laxy);
        project_unify(t, b, z, lbz, lbxy);
        cross(y, z, x);
        // Rotation around z for a, b
        float_type delta = util<float_type>::acos(dot(t, x));                   // angle delta between axy and bxy vectors
        if (dot(t, y) < .0f)
            delta = -delta;
        float_type alpha = rsample * constant<float_type>::pi2 / n_rsamples;    // sample angle
        rotate(a, x, y, z, laz, laxy, alpha);
        rotate(b, x, y, z, lbz, lbxy, alpha + delta);
    }

    // sum(s ∈ spots | s 🞄 v >= eps) -cos(2π * s 🞄 v / vlength)
    // v        unit vector in sample vector direction, |v| == 1
    // vlength  sample vector length
    // s{x,y,z} spot coordinate pointers [n_spots]
    // n_spots  number of spots
    template<typename float_type>
    __device__ __forceinline__ float_type sample1(const float_type v[3], const float_type vlength,
                                                  const float_type *sx, const float_type *sy, const float_type *sz,
                                                  const unsigned n_spots)
    {
        float_type sval = float_type{0.f};
        const float_type t_vl = float_type{2.f} / vlength;
        for (unsigned i=0u; i<n_spots; i++) {
            const float_type s[3] = { sx[i], sy[i], sz[i] };
            const float_type dp = dot(v, s);
            const float_type dv = util<float_type>::fma(.5f, util<float_type>::cospi(t_vl * dp), .5f);
            sval -= dv;
        }
        return sval;
    }

    // sum(s ∈ spots) -cos(2π * s 🞄 vi / |vi|²) for i in [1,2]
    // v1, v2   sample vectors
    // s{x,y,z} spot coordinate pointers [n_spots]
    // n_spots  number of spots
    template<typename float_type>
    __device__ __forceinline__ float_type sample2(const float_type v1[3], const float_type v2[3],
                                                  const float_type *sx, const float_type *sy, const float_type *sz,
                                                  const unsigned n_spots)
    {
        float_type sval = float_type{0.f};
        const float_type t_v1l2 = float_type{2.f} / norm2(v1);
        const float_type t_v2l2 = float_type{2.f} / norm2(v2);
        for (unsigned i=0u; i<n_spots; i++) {
            float_type t;
            const float_type s[3] = { sx[i], sy[i], sz[i] };
            {   // handle v1
                const float_type dp = dot(v1, s);
                const float_type dv = util<float_type>::fma(.5f, util<float_type>::cospi(t_v1l2 * dp), .5f);
                t = -dv;
            }
            {   // handle v2
                const float_type dp = dot(v2, s);
                const float_type dv = util<float_type>::fma(.5f, util<float_type>::cospi(t_v2l2 * dp), .5f);
                t -= dv;
            }
            sval += t;
        }
        return sval;
    }

    // -----------------------------------
    //            GPU Kernels
    // -----------------------------------

    // single thread kernel
    // kind 0-candidate vectors + output cells (float)
    //      1-output cells (unsigned)
    //      2-output cells (float)
    template<typename float_type>
    __global__ void gpu_debug_out(indexer_device_data<float_type>* data, const unsigned kind)
    {
        printf("### INDEXER_GPU_DEBUG\n");
        if (kind == 0u) {
            // Print out candidate vectors
            const unsigned ncv = data->cpers.num_candidate_vectors;
            const float_type* const cvp = data->candidate_value;
            const unsigned* const csp = data->candidate_sample;

            unsigned ncg = 0u;
            const unsigned n_cells_in  = data->input.n_cells;
            for (unsigned i=0; i<n_cells_in; ++i) {
                printf("input cell%u cgi:", i);
                for (unsigned j=3*i; j<3*i+3; ++j) {
                    unsigned cg = data->cellvec_to_cand[j];
                    printf(" %u", cg);
                    if (cg > ncg)
                        ncg = cg;
                }
                printf("\n");
            }

            for (unsigned i=0; i<=ncg; ++i) {
                printf("cg%u:", i);
                auto cv = &cvp[i * ncv];
                auto cs = &csp[i * ncv];
                for (unsigned j=0; j<ncv; ++j)
                    printf(" %0.2f/%u", (float)cv[j], cs[j]);
                printf("\n");
            }
        }

        const fast_feedback::output<float_type>& out = data->output;
        const unsigned n_cells_out  = data->cpers.max_output_cells;
        printf("output n_cells=%u\n", out.n_cells);
        for (unsigned i=0; i<n_cells_out; ++i) {
            printf("output cell%u s=%f:\n", i, out.score[i]);
            for (unsigned j=3*i; j<3*i+3; ++j) {
                if (kind == 0u) {
                    printf(" %u  %u  %u\n",
                        *reinterpret_cast<unsigned*>(&out.x[j]),
                        *reinterpret_cast<unsigned*>(&out.y[j]),
                        *reinterpret_cast<unsigned*>(&out.z[j]));
                } else {
                    printf(" %f  %f  %f\n", out.x[j], out.y[j], out.z[j]);
                }
            }
        }
        printf("###\n");
    }

    // sample = blockDim.x * blockIdx.x + threadIdx.x
    // candidate group = blockIdx.y
    template<typename float_type>
    __global__ void gpu_find_candidates(indexer_device_data<float_type>* data)
    {
        extern __shared__ double* shared_ptr[];

        unsigned sample = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned n_samples = data->crt.num_sample_points;
        const unsigned c_group = blockIdx.y;
        float_type v = 0.;  // objective function value for sample vector

        if (sample < n_samples) {                                   // calculate v
            const float_type sl = data->candidate_length[c_group];  // sample vector length
            float_type sv[3];                                       // unit vector in sample direction
            sample_point(sample, n_samples, sv);

            const unsigned spot_offset = 3 * data->cpers.max_input_cells;
            const fast_feedback::input<float_type>& in = data->input;
            const unsigned n_spots = in.n_spots;
            const float_type* sx = &in.x[spot_offset];
            const float_type* sy = &in.y[spot_offset];
            const float_type* sz = &in.z[spot_offset];
            
            v = sample1(sv, sl, sx, sy, sz, n_spots);
        }

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

        const unsigned num_candidate_vectors = data->cpers.num_candidate_vectors;
        auto cand_ptr = reinterpret_cast<vec_cand_t<float_type>*>(shared_ptr); // [num_candidate_vectors]
        if (threadIdx.x < num_candidate_vectors)    // store top candidate vectors into cand_ptr
            cand_ptr[threadIdx.x] = { v, sample };

        if (num_candidate_vectors < warp_size) {    // wait for cand_ptr
            if (threadIdx.x < warp_size)
                __syncwarp();
        } else
            __syncthreads();

        if (threadIdx.x == 0) { // merge block local into global result
            auto value_ptr = &data->candidate_value[c_group * num_candidate_vectors];
            auto sample_ptr = &data->candidate_sample[c_group * num_candidate_vectors];

            seq_acquire(data->seq_block[c_group]);
            merge_top_vecs(value_ptr, sample_ptr, cand_ptr, num_candidate_vectors);
            seq_release(data->seq_block[c_group]);
        }

        // clear output cell scores
        if (sample < data->cpers.max_output_cells) {
            data->output.score[sample] = float_type{.0f};
        }
    }

    // sample rotation angle index = blockDim.x * blockIdx.x + threadIdx.x
    // input cell vector index = blockIdx.y
    // sample vector index = blockIdx.z
    template<typename float_type>
    __global__ void gpu_find_cells(indexer_device_data<float_type>* data)
    {
        extern __shared__ double* shared_ptr[];

        const unsigned n_vsamples = data->crt.num_sample_points;
        unsigned rsample = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned cell_vec = blockIdx.y;
        const unsigned cand = blockIdx.z;
        const unsigned cand_grp = data->cellvec_to_cand[cell_vec];
        const unsigned n_cand = data->cpers.num_candidate_vectors;
        const unsigned vsample = data->candidate_sample[cand_grp * n_cand + cand];
        float_type vabc=.0f;    // accumulated sample values for a, b, c cell vectors

        { // calculate vabc for vsample/rsample/
            const fast_feedback::input<float_type>& in = data->input;
            const unsigned n_rsamples = gridDim.x * blockDim.x;
            const float_type vlength = data->candidate_length[cand_grp];
            float_type z[3], a[3], b[3];

            sample_cell(z, a, b, in.x, in.y, in.z, vlength, vsample, n_vsamples, rsample, n_rsamples, cell_vec);

            const unsigned n_spots = in.n_spots;
            const unsigned spot_offset = 3u * data->cpers.max_input_cells;
            const float_type* sx = &in.x[spot_offset];
            const float_type* sy = &in.y[spot_offset];
            const float_type* sz = &in.z[spot_offset];
            vabc = sample2(a, b, sx, sy, sz, n_spots);
            
            const float_type vvalue = data->candidate_value[cand_grp * n_cand + cand];
            vabc += vvalue;
        }

        // Get best output cells for block
        {   // sort within block {objective function value, sample} ascending by objective function value
            float_type key[1] = {vabc};
            unsigned val[1] = {rsample};
            {
                auto sort_ptr = reinterpret_cast<typename BlockRadixSort<float_type>::TempStorage*>(shared_ptr);
                BlockRadixSort<float_type>(*sort_ptr).Sort(key, val);
            }
            vabc = *key;
            rsample = *val;
            __syncthreads();                // protect sort_ptr[] (it's in the same memory as cand_ptr)
        }

        const unsigned n_output_cells = data->output.n_cells;
        auto cand_ptr = reinterpret_cast<cell_cand_t<float_type>*>(shared_ptr); // [output.n_cells]
        if (threadIdx.x < n_output_cells)   // store top candidate cells into cand_ptr
            cand_ptr[threadIdx.x] = { vabc, vsample, rsample, cell_vec };

        if (n_output_cells < warp_size) {   // wait for cand_ptr
            if (threadIdx.x < warp_size)
                __syncwarp();
        } else
            __syncthreads();

        if (threadIdx.x == 0) {             // merge block local into global result
            seq_acquire(data->seq_block[0]);
            merge_top_cells(data->output, cand_ptr, n_output_cells);
            seq_release(data->seq_block[0]);
        }
    }

    // expand {cand, rsample, cell_vec} to coordinates of unit cell vectors
    // threadIdx.x = output cell index
    // n_rsamples   number of rotation angle samples
    template<typename float_type>
    __global__ void gpu_expand_cells(indexer_device_data<float_type>* data, const unsigned n_rsamples)
    {
        const unsigned n_vsamples = data->crt.num_sample_points;
        const fast_feedback::input<float_type>& in = data->input;
        fast_feedback::output<float_type>& out = data->output;
        float_type* ox = out.x;
        float_type* oy = out.y;
        float_type* oz = out.z;
        const unsigned cell_base = 3u * threadIdx.x;
        const unsigned vsample = *reinterpret_cast<unsigned*>(&ox[cell_base]);
        const unsigned rsample = *reinterpret_cast<unsigned*>(&oy[cell_base]);
        const unsigned cell_vec = *reinterpret_cast<unsigned*>(&oz[cell_base]);
        const unsigned cand_grp = data->cellvec_to_cand[cell_vec];
        const float_type vlength = data->candidate_length[cand_grp];
        float_type z[3], a[3], b[3];

        sample_cell(z, a, b, in.x, in.y, in.z, vlength, vsample, n_vsamples, rsample, n_rsamples, cell_vec);

        ox[cell_base] = z[0] * vlength;
        ox[cell_base + 1u] = a[0];
        ox[cell_base + 2u] = b[0];
        oy[cell_base] = z[1] * vlength;
        oy[cell_base + 1u] = a[1];
        oy[cell_base + 2u] = b[1];
        oz[cell_base] = z[2] * vlength;
        oz[cell_base + 1u] = a[2];
        oz[cell_base + 2u] = b[2];
    }

} // anonymous namespace

namespace gpu {

    template <typename float_type>
    void init (const indexer<float_type>& instance)
    {
        using gpu_state = indexer_gpu_state<float_type>;
        using device_data = indexer_device_data<float_type>;

        cuda_init();

        const auto state_id = instance.state;
        if (gpu_state::exists(state_id))
            throw FF_EXCEPTION("instance illegal reinitialisation");

        const auto& cpers = instance.cpers;

        device_data* data = nullptr;
        float_type* candidate_length = nullptr;
        float_type* candidate_value = nullptr;
        unsigned* candidate_sample = nullptr;
        unsigned* cellvec_to_cand = nullptr;
        unsigned* sequentializers = nullptr;
        float_type* elements = nullptr;
        float_type* ix;
        float_type* iy;
        float_type* iz;
        float_type* ox;
        float_type* oy;
        float_type* oz;
        float_type* scores;
        {
            gpu_stream stream;
            stream.init();      // new cuda stream

            CU_CHECK(cudaMalloc(&data, sizeof(device_data)));
            gpu_pointer<device_data> data_ptr{data};
        
            {
                std::size_t vector_dimension = std::max(3 * cpers.max_input_cells, cpers.max_output_cells);
                CU_CHECK(cudaMalloc(&sequentializers, vector_dimension * sizeof(unsigned)));
            }
            gpu_pointer<unsigned> sequentializers_ptr{sequentializers};
            {   // initialize sequentializers
                std::size_t vector_dimension = 3 * cpers.max_input_cells;
                CU_CHECK(cudaMemsetAsync(sequentializers, 0, vector_dimension * sizeof(unsigned), stream));
            }

            {
                std::size_t vector_dimension = 10 * cpers.max_output_cells + 9 * cpers.max_input_cells + 3 * cpers.max_spots;
                CU_CHECK(cudaMalloc(&elements, vector_dimension * sizeof(float_type)));
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
                std::size_t vector_dimension = 3 * cpers.max_input_cells;
                CU_CHECK(cudaMalloc(&cellvec_to_cand, vector_dimension * sizeof(unsigned)));
            }
            gpu_pointer<unsigned> v2c_ptr{cellvec_to_cand};

            std::size_t dim = cpers.max_output_cells;
            scores = elements;

            ox = elements + dim;
            dim = 3 * cpers.max_output_cells;
            oy = ox + dim;
            oz = oy + dim;

            ix = oz + dim;
            dim = 3 * cpers.max_input_cells + cpers.max_spots;
            iy = ix + dim;
            iz = iy + dim;

            {
                std::lock_guard<std::mutex> state_lock{gpu_state::state_update};
                gpu_state::ref(state_id) = gpu_state{std::move(data_ptr), std::move(element_ptr),
                                                     std::move(candidate_length_ptr), std::move(candidate_value_ptr),
                                                     std::move(candidate_sample_ptr), std::move(v2c_ptr),
                                                     std::move(sequentializers_ptr), std::move(stream),
                                                     ix, iy, iz,
                                                     ox, oy, oz,
                                                     scores};
            }

            logger::debug << stanza << "init id=" << state_id << ", elements=" << elements << '\n'
                          << stanza << "     ox=" << ox << ", oy=" << oy << ", oz=" << oz << '\n'
                          << stanza << "     ix=" << ix << ", iy=" << iy << ", iz=" << iz << '\n'
                          << stanza << "     candval=" << candidate_value << ", candsmpl=" << candidate_sample << '\n';
        }

        {
            device_data _data{cpers, fast_feedback::config_runtime<float_type>{},
                              fast_feedback::input<float_type>{ix, iy, iz, 0, 0},
                              fast_feedback::output<float_type>{ox, oy, oz, scores, 0},
                              candidate_length, candidate_value,
                              candidate_sample, cellvec_to_cand,
                              sequentializers};
            CU_CHECK(cudaMemcpyAsync(data, &_data, sizeof(_data), cudaMemcpyHostToDevice, gpu_state::stream(state_id)));
        }
    }

    // Drop state if any
    template <typename float_type>
    void drop (const indexer<float_type>& instance)
    {
        using gpu_state = indexer_gpu_state<float_type>;

        cuda_init();
        gpu_state::drop(instance.state);
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
        const auto n_cells_out = std::min(out.n_cells, instance.cpers.max_output_cells);
        logger::debug << stanza << "n_cells = " << n_cells_in << "(in)/" << n_cells_out << "(out), n_spots = " << in.n_spots << '\n';
        if (n_cells_in <= 0u)
            throw FF_EXCEPTION("no given input cells");
        if (n_cells_out <= 0)
            throw FF_EXCEPTION("no output cells");
        if (in.n_spots <= 0u)
            throw FF_EXCEPTION("no spots");
        if (instance.cpers.num_candidate_vectors < 1u)
            throw FF_EXCEPTION("nonpositive number of candidate vectors");
        if (conf_rt.num_sample_points < instance.cpers.num_candidate_vectors)
            throw FF_EXCEPTION("fewer sample points than required candidate vectors");
        if (n_cells_out > n_threads)
            throw FF_EXCEPTION("fewer threads in a block than output cells");
        if (instance.cpers.num_candidate_vectors > n_threads)
            throw FF_EXCEPTION("fewer threads in a block than candidate vectors");
        
        // Calculate input vector candidate groups
        unsigned n_cand_groups = 0u;
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
            n_cand_groups = i + 1;
            for (unsigned i=0u; i<candidate_idx.size(); ++i) {
                const auto x = in.x[i];
                const auto y = in.y[i];
                const auto z = in.z[i];
                float_type length = std::sqrt(x*x + y*y + z*z);
                auto it = std::lower_bound(std::cbegin(candidate_length), std::cbegin(candidate_length) + n_cand_groups, length,
                                        [l_threshold](const float_type& a, const float_type& l) -> bool {
                                                return (a - l) >= l_threshold;
                                        });
                candidate_idx[i] = it - std::cbegin(candidate_length);
            }
            if (logger::level_active<logger::l_debug>()) {
                logger::debug << stanza << "candidate_idx =";
                for (const auto& e : candidate_idx)
                    logger::debug << ' ' << e;
                logger::debug << ", n_cand_groups = " << n_cand_groups << '\n';
            }
        }
        gpu_stream stream{gpu_stream::from_cache()};    // cuda stream from unused pool
        {   // find vector candidates
            const unsigned n_samples = conf_rt.num_sample_points;
            const dim3 n_blocks((n_samples + n_threads - 1) / n_threads, n_cand_groups);
            const unsigned shared_sz = std::max(instance.cpers.num_candidate_vectors * sizeof(vec_cand_t<float_type>),
                                                sizeof(typename BlockRadixSort<float_type>::TempStorage));
            gpu_state::stream(state_id).sync(); // sync on initialization
            gpu_state::copy_crt(state_id, conf_rt, stream);
            gpu_state::copy_in(state_id, instance.cpers, in, out, stream);
            gpu_state::init_cand(state_id, n_cand_groups, instance.cpers, candidate_length, candidate_idx, stream);
            state.start.record(stream);
            gpu_find_candidates<float_type><<<n_blocks, n_threads, shared_sz, stream>>>(gpu_state::ptr(state_id).get());
        }
        {   // find cells
            const unsigned n_xblocks = (1.5 * std::sqrt(conf_rt.num_sample_points) + n_threads - 1.) / n_threads;
            const dim3 n_blocks(n_xblocks, 3 * n_cells_in, instance.cpers.num_candidate_vectors);
            const unsigned shared_sz = std::max(n_cells_out * sizeof(cell_cand_t<float_type>),
                                                sizeof(typename BlockRadixSort<float_type>::TempStorage));     

            // gpu_state::init_score(state_id, instance.cpers);
            if (gpu_debug_output)
                gpu_debug_out<float_type><<<1, 1, 0, stream>>>(gpu_state::ptr(state_id).get(), 0u);
            gpu_find_cells<float_type><<<n_blocks, n_threads, shared_sz, 0>>>(gpu_state::ptr(state_id).get());
            if (gpu_debug_output)
                gpu_debug_out<float_type><<<1, 1, 0, stream>>>(gpu_state::ptr(state_id).get(), 1u);
            gpu_expand_cells<float_type><<<1, n_cells_out, 0, stream>>>(gpu_state::ptr(state_id).get(), n_xblocks * n_threads);
            if (gpu_debug_output)
                gpu_debug_out<float_type><<<1, 1, 0, stream>>>(gpu_state::ptr(state_id).get(), 2u);
            state.end.record(stream);
            gpu_state::copy_out(state_id, out, stream); // synchronizes on stream
        }
        if (timing) {
            end = clock::now();
            duration elapsed = end - start;
            logger::info << stanza << "indexing_time: " << elapsed.count() << "ms\n";
            logger::info << stanza << "kernel_time: " << gpu_timing(state.start, state.end) << "ms\n";
        }
        gpu_stream::to_cache(std::move(stream));    // return stream to unused pool
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
