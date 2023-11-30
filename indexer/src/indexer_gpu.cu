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
#include <limits>
#include "ffbidx/exception.h"
#include "ffbidx/log.h"
#include "ffbidx/indexer_gpu.h"
#include "cuda_runtime.h"
#include <cub/block/block_radix_sort.cuh>

#define VCR_NONE 0  // No vector candidate refinement
#define VCR_ROPT 1  // Vector candidate refinement with robust optimization

#ifndef VECTOR_CANDIDATE_REFINEMENT // Macro for switching on vector candidate refinement
    #define VCANDREF VCR_NONE   // Use no candidate vector refinement
#elif VECTOR_CANDIDATE_REFINEMENT == VCR_NONE
    #define VCANDREF VCR_NONE   // Use no candidate vector refinement
#elif VECTOR_CANDIDATE_REFINEMENT == VCR_ROPT
    #define VCANDREF VCR_ROPT   // Use robust optimization for candidate vector refinement
#else
    #error "VECTOR_CANDIDATE_REFINEMENT has an unsupported value! Use either 0 (none), or 1 (robust optimization)."
#endif

namespace logger = fast_feedback::logger;
using logger::stanza;

namespace {

    constexpr char INDEXER_GPU_DEVICE[] = "INDEXER_GPU_DEVICE";
    constexpr char INDEXER_GPU_DEBUG[] = "INDEXER_GPU_DEBUG";
    constexpr unsigned n_threads = 1024;    // num cuda threads per block for find_candidates kernel
    constexpr unsigned warp_size = 32;      // number of threads in a warp
    constexpr unsigned mem_txn_unit = 32;   // minimum memory transaction unit (cache line sector)

    template<typename float_type>
    struct constant final {
        // static constexpr float_type pi = 3.1415926535897932384628;
        #pragma nv_diag_suppress 177 // suppress unused warning
        [[maybe_unused]] static constexpr float_type pi2 = 6.2831853071795864769257;
        #pragma nv_diag_suppress 177 // suppress unused warning
        [[maybe_unused]] static constexpr float_type dl = 0.76393202250021030359082633; // 3 - sqrt(5), for spiral sample points on a half sphere
        #pragma nv_diag_suppress 177 // suppress unused warning
        [[maybe_unused]] static constexpr unsigned mem_txn_unit = ::mem_txn_unit / sizeof(float_type); // minimum memory transaction unit
    };

    // NOTE: not allowed as member of constant
    // (dl * 2^N) mod 2pi
    template<typename float_type>
    __constant__ static constexpr float_type dl2pNmod2pi[32] = {
        0.76393202250021030359082633,   // N = 0
        1.5278640450004206071816527,
        3.0557280900008412143633053,
        6.1114561800016824287266107,
        5.9397270528237783805279346,
        5.5962687984679702841305826,
        4.9093522897563540913358776,
        3.535519272333121705746469,
        0.78785323748665693456765102,   // 8
        1.575706474973313869135302,
        3.1514129499466277382706041,
        0.019640592713668999615921566,
        0.039281185427337999231843132,
        0.078562370854675998463686265,
        0.15712474170935199692737253,
        0.31424948341870399385474506,
        0.62849896683740798770949012,   // 16
        1.2569979336748159754189802,
        2.5139958673496319508379605,
        5.0279917346992639016759209,
        3.7727981622189413264265548,
        1.2624110172582961759278224,
        2.5248220345165923518556449,
        5.0496440690331847037112897,
        3.8161028308867829304972927,    // 24
        1.3490203545939793840692983,
        2.6980407091879587681385967,
        5.3960814183759175362771934,
        4.5089775295722485956291,
        2.7347697519649107143329137,
        5.4695395039298214286658275,
        4.6558937006800563804063691     // 31
    };

    std::atomic_bool gpu_debug_output{false};   // Print gpu debug output if true, set if INDEXER_GPU_DEBUG env var is set to true at indexer creation time

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

    // Cuda device infos
    struct gpu_device final {
        int id;
        cudaDeviceProp prop;

        static inline std::vector<gpu_device> list;             // Cuda device list
        static inline std::atomic_bool cuda_initialized{false}; // Is the cuda runtime initialized?
        static inline std::mutex init_lock;                     // Protect cuda runtime initalisation

        // Check if cuda has been initialized
        static void check_init()
        {
            if (! gpu_device::cuda_initialized.load())
                throw FF_EXCEPTION("cuda runtime not initialized");
        }

        // Get the list of cuda devices
        static void init()
        {
            if (cuda_initialized.load())
                return;
            
            {
                std::lock_guard<std::mutex> lock{init_lock};

                if (cuda_initialized.load())
                    return;
            
                int num_devices;
                CU_CHECK(cudaGetDeviceCount(&num_devices));

                list.resize(num_devices);
                for (int dev=0; dev<num_devices; dev++) {
                    gpu_device& device = list[dev];
                    device.id = dev;
                    CU_CHECK(cudaGetDeviceProperties(&device.prop, dev));
                }

                gpu_device::cuda_initialized.store(true);
            }
        }

        // Get a CUDA device
        // Take the device given with INDEXER_GPU_DEVICE if set,
        // otherwise take the current device.
        static int get()
        {
            int dev{-1};
            CU_CHECK(cudaGetDevice(&dev));

            char* dev_string = std::getenv(INDEXER_GPU_DEVICE);
            if (dev_string != nullptr) {
                std::istringstream iss{dev_string};
                iss >> dev;
                if (!iss || !iss.eof())
                    throw FF_EXCEPTION_OBJ << "wrong format for " << INDEXER_GPU_DEVICE << ": " << dev_string << " (should be an integer)";
                if ((dev < 0) || (dev >= (int)list.size()))
                    throw FF_EXCEPTION_OBJ << "illegal value for " << INDEXER_GPU_DEVICE << ": " << dev << " (should be in [0.." << list.size() << "[)";
            }

            LOG_START(logger::l_info) {
                logger::info << stanza << "using GPU device " << dev << '\n';
            } LOG_END;

            return dev;
        }

        // Set current CUDA device
        static void set(int dev)
        {
            if ((dev < 0) || (dev >= (int)list.size()))
                throw FF_EXCEPTION_OBJ << "illegal value for GPU device";

            CU_CHECK(cudaSetDevice(dev));
        }
    };

    // Set GPU debug output if INDEXER_GPU_DEBUG is set to true
    void gpu_debug_init()
    {
        char* debug_string = std::getenv(INDEXER_GPU_DEBUG);
        if (debug_string != nullptr) {
            const std::vector<std::string> accepted = {"1", "true", "yes", "on", "0", "false", "no", "off"}; // [4] == "0"
            unsigned i;
            for (i=0u; i<accepted.size(); i++) {
                if (accepted[i] == debug_string) {
                    gpu_debug_output.store(i < 4);                                                           // [4] == "0"
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
            gpu_debug_output.store(false);
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
            if (event == cudaEvent_t{})
                CU_CHECK(cudaEventCreateWithFlags(&event, flags));
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

    // Cuda stream wrapper
    struct gpu_stream final {
        bool ready;                             // Is stream ready = initialized
        cudaStream_t stream;                    // Cuda stream

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
            if (this != &s) {
                if (ready) {
                    CU_CHECK(cudaStreamDestroy(stream));
                    ready = false;
                }
                std::swap(ready, s.ready);
                std::swap(stream, s.stream);
            }
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

        // Check for completion status
        bool completed()
        {
            if (ready) {
                auto error = cudaStreamQuery(stream);
                switch (error) {
                    case cudaSuccess:
                        return true;
                    case cudaErrorNotReady:
                        return false;
                    default:
                        throw CU_EXCEPTION(error);
                }
            }
            return true;
        }

        operator cudaStream_t&() noexcept { return stream; }    // Cast to cuda stream
    };

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
        #if VCANDREF == VCR_ROPT
            float_type* refined_candidates;     // Refined candidate coordinates in consecutive aligned tripples (x,y,z,gap; alignment = mem_txn_unit)
        #endif
        unsigned* candidate_sample;             // Sample number of candidate vectors, [3 * max_input_cells * num_candidate_vectors]
        unsigned* cellvec_to_cand;              // Input cell vector to candidate group mapping, [3 * max_input_cells]
        unsigned* cell_to_cellvec;              // Input cell to representing cell vector mapping, [max_input_cells]
        unsigned* vec_cgrps;                    // Candidate vector groups of cell representing vectors, [max_input_cells]
        unsigned* seq_block;                    // Per candidate vector group sequentializer for thread blocks (explicit init to 0, set to 0 in kernel after use)
                                                //     First sequentializer is also used for candidate cell search
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
        using clock = std::chrono::high_resolution_clock;
        using time_point = std::chrono::time_point<clock>;

        gpu_pointer<content_type> data;                     // Indexer_device_data on GPU
        gpu_pointer<float_type> elements;                   // Input/output vector elements of data on GPU
        gpu_pointer<float_type> candidate_length;           // Candidate vector groups length
        gpu_pointer<float_type> candidate_value;            // Candidate vectors objective function values on GPU
        #if VCANDREF == VCR_ROPT
            gpu_pointer<float_type> refined_candidates;     // Refined candidate coordinates in consecutive aligned tripples
        #endif
        gpu_pointer<unsigned> candidate_sample;             // Sample number of candidate vectors
        gpu_pointer<unsigned> cellvec_to_cand;              // Input cell vector to candidate group mapping
        gpu_pointer<unsigned> cell_to_cellvec;              // Input cell to cell representing vector mapping
        gpu_pointer<unsigned> vec_cgrps;                    // Candidate vector groups of cell representing vectors
        gpu_pointer<unsigned> seq_block;                    // Thread block sequentializers
        gpu_stream cuda_stream;                             // CUDA stream

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
        time_point start_time{};

        // GPU device
        int device;

        // Callback mode
        bool callback_mode = false;

        static std::mutex state_update; // Protect per indexer state map
        static map_type dev_ptr;        // Per indexer state map

        // Create hostside on GPU state representation with
        // d        pointer to on GPU state
        // e        on GPU input and output elements pointed to by d->elements
        // cl       on GPU candidate lengths pointed to by d->candidate_length
        // cv       on GPU candidate vector objective function values pointed to by d->candidate_value
        // cs       on GPU candidate vector sample numbers pointed to by d->candidate_sample
        // v2c      on GPU input cell vector to candidate group mapping pointed to by d->cellvec_to_cand
        // c2v      on GPU input cell to cell representing vector mapping pointed to by d->cell_to_cellvec
        // vcgr     on GPU candidate groups of cell representing vectors pointed to by d->vec_cgrps
        // sb       on GPU per candidate vector group/output cell block sequentializers pointed to by d->seq_block
        // stream   GPU stream used for actions on this state
        // x/y/zi   on GPU input pointers d->input.x / y / z
        // x/y/z0   on GPU output pointers d->output.x / y / z
        // score    on GPU per output cell scoring function values
        // dev      cuda device number of the GPU holding this state
        // cbm      host callback function is called when results are ready
        indexer_gpu_state(gpu_pointer<content_type>&& d, gpu_pointer<float_type>&& e,
                          gpu_pointer<float_type>&& cl, gpu_pointer<float_type>&& cv,
                          #if VCANDREF == VCR_ROPT
                            gpu_pointer<float_type>&& rc,
                          #endif
                          gpu_pointer<unsigned>&& cs, gpu_pointer<unsigned>&& v2c,
                          gpu_pointer<unsigned>&& c2v, gpu_pointer<unsigned>&& vcgr,
                          gpu_pointer<unsigned>&& sb, gpu_stream&& stream,
                          float_type* xi, float_type* yi, float_type* zi,
                          float_type* xo, float_type* yo, float_type* zo,
                          float_type* scores, int dev)
            : data{std::move(d)}, elements{std::move(e)},
              candidate_length{std::move(cl)}, candidate_value{std::move(cv)},
              #if VCANDREF == VCR_ROPT
                refined_candidates{std::move(rc)},
              #endif
              candidate_sample{std::move(cs)}, cellvec_to_cand{std::move(v2c)},
              cell_to_cellvec{std::move(c2v)}, vec_cgrps{std::move(vcgr)},
              seq_block{std::move(sb)}, cuda_stream{std::move(stream)},
              ix{xi}, iy{yi}, iz{zi},
              ox{xo}, oy{yo}, oz{zo},
              cell_score{scores}, device{dev}
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
            return ref(id).cuda_stream;
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
            LOG_START(logger::l_debug) {
                logger::debug << stanza << "copy runtime config data: " << &crt << "-->" << crt_dp << '\n';
            } LOG_END;
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
            pinned_tmp.max_input_cells = n_input_cells;
            pinned_tmp.max_spots = n_spots;
            CU_CHECK(cudaMemcpyAsync(&device_data->output.n_cells, &pinned_tmp.max_output_cells, sizeof(output.n_cells), cudaMemcpyHostToDevice, stream));
            // NOTE: the following code assumes consecutive storage of two data members in pinned_tmp and in device_data->input
            CU_CHECK(cudaMemcpyAsync(&device_data->input.n_cells, &pinned_tmp.max_input_cells, sizeof(n_input_cells) + sizeof(n_spots), cudaMemcpyHostToDevice, stream));

            if (n_input_cells + n_spots > 0u) {
                if (input.new_cells && n_input_cells > 0u) {
                    const auto cell_sz = 3u * n_input_cells * sizeof(float_type);
                    CU_CHECK(cudaMemcpyAsync(gpu_state.ix, input.cell.x, cell_sz, cudaMemcpyHostToDevice, stream));
                    CU_CHECK(cudaMemcpyAsync(gpu_state.iy, input.cell.y, cell_sz, cudaMemcpyHostToDevice, stream));
                    CU_CHECK(cudaMemcpyAsync(gpu_state.iz, input.cell.z, cell_sz, cudaMemcpyHostToDevice, stream));
                }
                if (input.new_spots && n_spots > 0) {
                    const auto dst_offset = 3u * cpers.max_input_cells;
                    const auto spot_sz = n_spots * sizeof(float_type);
                    CU_CHECK(cudaMemcpyAsync(&gpu_state.ix[dst_offset], input.spot.x, spot_sz, cudaMemcpyHostToDevice, stream));
                    CU_CHECK(cudaMemcpyAsync(&gpu_state.iy[dst_offset], input.spot.y, spot_sz, cudaMemcpyHostToDevice, stream));
                    CU_CHECK(cudaMemcpyAsync(&gpu_state.iz[dst_offset], input.spot.z, spot_sz, cudaMemcpyHostToDevice, stream));
                }
            }

            LOG_START(logger::l_debug) {
                logger::debug << stanza << "copy in: " << n_input_cells << " cells(in), " << output.n_cells << " cells(out), "
                            << n_spots << " spots, elements=" << gpu_state.elements.get() << ": (cells) "
                            << input.cell.x << "-->" << gpu_state.ix << ", "
                            << input.cell.y << "-->" << gpu_state.iy << ", "
                            << input.cell.z << "-->" << gpu_state.iz << ", (spots) "
                            << input.cell.x << "-->" << &gpu_state.ix[3u * cpers.max_input_cells] << ", "
                            << input.cell.y << "-->" << &gpu_state.iy[3u * cpers.max_input_cells] << ", "
                            << input.cell.z << "-->" << &gpu_state.iz[3u * cpers.max_input_cells] << '\n';
            } LOG_END;
        }

        static inline void init_cand(const key_type& state_id, const unsigned n_cand_groups, const unsigned n_vec_cgrps,
                                     const config_persistent& cpers,
                                     const std::vector<float_type>& cand_len, const std::vector<unsigned>& cand_idx,
                                     const std::vector<unsigned>& cell_cand, const std::vector<unsigned>& vec_cgrps,
                                     cudaStream_t stream=0)
        {
            const auto n_cand_vecs = n_cand_groups * cpers.num_candidate_vectors;
            const auto& gpu_state = ref(state_id);

            CU_CHECK(cudaMemcpyAsync(gpu_state.candidate_length.get(), cand_len.data(), n_cand_groups * sizeof(float_type), cudaMemcpyHostToDevice, stream));
            CU_CHECK(cudaMemsetAsync(gpu_state.candidate_value.get(), 0, n_cand_vecs * sizeof(float_type), stream));
            CU_CHECK(cudaMemcpyAsync(gpu_state.cellvec_to_cand.get(), cand_idx.data(), cand_idx.size() * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
            CU_CHECK(cudaMemcpyAsync(gpu_state.cell_to_cellvec.get(), cell_cand.data(), cell_cand.size() * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
            CU_CHECK(cudaMemcpyAsync(gpu_state.vec_cgrps.get(), vec_cgrps.data(), n_vec_cgrps * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
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

            LOG_START(logger::l_debug) {
                logger::debug << stanza << "copy out: " << output.n_cells << " cells, elements=" << gpu_state.elements.get() << ": "
                            << gpu_state.ox << "-->" << output.x << ", "
                            << gpu_state.oy << "-->" << output.y << ", "
                            << gpu_state.oz << "-->" << output.z << ", "
                            << gpu_state.cell_score << "-->" << output.score << '\n';
            } LOG_END;
        }
    };

    // Calculate vector candidate groups
    //   All input cell vectors are mapped to a candidate vector group that uniquely represents the length of the vector.
    //   crt.length_threshold determines if two vectors are considered to have the same length.
    // Input Args:
    //   in        : indexing input
    //   crt       : runtime configuration
    //   n_cells_in: number of considered input cells
    // Output Args:
    //   cand_idx: input cell vector to candidate vector group mapping (preallocated size: 3 * n_cells_in)
    //   cand_len: sorted candidate vector group length                (preallocated size: 3 * n_cells_in, initialized to: 0)
    // Return:
    //   number of candidate vector groups N ∈ [1 ... 3 * n_cells_in]
    template <typename float_type>
    unsigned calc_cand_groups(std::vector<unsigned>& cand_idx, std::vector<float_type>& cand_len,
                              const fast_feedback::input<float_type>& in,
                              const fast_feedback::config_runtime<float_type>& crt, const unsigned n_cells_in)
    {
        const unsigned n_vecs = 3u * n_cells_in;

        if (cand_idx.size() < n_vecs)
            throw FF_EXCEPTION("candidate index vector too small");
        if (cand_len.size() < n_vecs)
            throw FF_EXCEPTION("candidate length vector too small");
        
        // All vector lengths
        for (unsigned i=0u; i<n_vecs; i++) {
            const auto x = in.cell.x[i];
            const auto y = in.cell.y[i];
            const auto z = in.cell.z[i];
            cand_len[i] = std::sqrt(x*x + y*y + z*z);
        }

        unsigned n_cand_groups{};
        {   // Only keep elements that differ by more than length_threshold
            std::sort(std::begin(cand_len), std::end(cand_len), std::greater<float_type>{});

            const float_type l_threshold = crt.length_threshold;
            LOG_START(logger::l_debug) {
                logger::debug << stanza << "candidate_length =";
                for (const auto& e : cand_len)
                    logger::debug << ' ' << e;
                logger::debug << ", threshold = " << l_threshold << '\n';                
            } LOG_END;
            
            unsigned i=0, j=1;
            do {
                if ((cand_len[i] - cand_len[j]) < l_threshold) {
                    LOG_START(logger::l_debug) {
                        logger::debug << stanza << "  ignore " << cand_len[j] << '\n';
                    } LOG_END;
                } else if (++i != j) {
                    cand_len[i] = cand_len[j];
                }
            } while(++j != n_vecs);
            n_cand_groups = i + 1;
            for (unsigned i=0u; i<n_vecs; i++) {
                const auto x = in.cell.x[i];
                const auto y = in.cell.y[i];
                const auto z = in.cell.z[i];
                float_type length = std::sqrt(x*x + y*y + z*z);
                auto it = std::lower_bound(std::cbegin(cand_len), std::cbegin(cand_len) + n_cand_groups, length,
                                        [l_threshold](const float_type& a, const float_type& l) -> bool {
                                                return (a - l) >= l_threshold;
                                        });
                cand_idx[i] = it - std::cbegin(cand_len);
            }
        }
        return n_cand_groups;
    }

    // Calculate a candidate vector for each cell
    //   Chose a vector for every input cell. This vector will be used for the initial half-sphere brute force sampling step.
    //   The algorithm relies on the decreasing order of the candidate group lengths and corresponding indices in cand_idx.
    //   So a lower candidate vector group index in cand_idx means a longer vector.
    // Input Args:
    //   cand_idx: input cell vector to candidate vector group mapping (see ordering requirement above, size 3 * n_cells_in)
    // Output Args:
    //   cell_vec: cell to chosen input vector mapping (preallocated size: n_cells_in)
    //   vec_cand: candidate groups of chosen input vectors (preallocated size: n_cells_in)
    // Return:
    //   Number of candidate groups for the chosen vectors
    unsigned calc_cell_cand(std::vector<unsigned>& cell_vec, std::vector<unsigned>& vec_cand,
                            const std::vector<unsigned>& cand_idx, const unsigned n_cells_in)
    {
        const unsigned n_vecs = 3u * n_cells_in;

        if (cand_idx.size() < n_vecs)
            throw FF_EXCEPTION("candidate index vector too small");
        if (cell_vec.size() < n_cells_in)
            throw FF_EXCEPTION("cell candidate vector too small");
        if (vec_cand.size() < n_cells_in)
            throw FF_EXCEPTION("candidate groups vector too small");

        unsigned num_cand_grps = 0u;
        unsigned vec = 0u;
        for (unsigned cell=0u; cell<n_cells_in; cell++) {
            unsigned cand = vec++;
            for (unsigned v=1u; v<3u; vec++, v++) {
                if (cand_idx[cand] > cand_idx[vec])
                    cand = vec;
            }
            cell_vec[cell] = cand;
            unsigned cand_grp = cand_idx[cand];
            auto it = std::find(&vec_cand[0u], &vec_cand[num_cand_grps], cand_grp);
            if (it == &vec_cand[num_cand_grps])
                vec_cand[num_cand_grps++] = cand_grp;
        }

        return num_cand_grps;
    }

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
        __device__ __forceinline__ static float abs(const float val) noexcept
        {
            return fabsf(val);
        }

        __device__ __forceinline__ static float rint(const float val) noexcept
        {
            return rintf(val);
        }

        __device__ __forceinline__ static void sincos(const float angle, float* sine, float* cosine) noexcept
        {
            return sincosf(angle, sine, cosine);
        }

        __device__ __forceinline__ static float cos(const float angle) noexcept
        {
            return cosf(angle);
        }

        __device__ __forceinline__ static float acos(const float x) noexcept
        {
            return acosf(x);
        }

        __device__ __forceinline__ static float log2(const float x) noexcept
        {
            return log2f(x);
        }

        __device__ __forceinline__ static float exp2(const float x) noexcept
        {
            return exp2f(x);
        }

        __device__ __forceinline__ static float sqrt(const float x) noexcept
        {
            return sqrtf(x);
        }
        
        __device__ __forceinline__ static float rem(const float x, const float y) noexcept
        {
            return remainderf(x, y);
        }

        __device__ __forceinline__ static float norm(const float x, const float y, const float z) noexcept
        {   // sqrt(x*x + y*y + z*z)
            return norm3df(x, y, z);
        }

        __device__ __forceinline__ static float rnorm(const float x, const float y, const float z) noexcept
        {   // 1 / sqrt(x*x + y*y + z*z)
            return rnorm3df(x, y, z);
        }

        __device__ __forceinline__ static float fma(const float x, const float y, const float z) noexcept
        {   // x * y + z
            return fmaf(x, y, z);
        }

        __device__ __forceinline__ static void sincospi(const float a, float* sinp, float* cosp) noexcept
        {
            sincospif(a, sinp, cosp);
        }

        __device__ __forceinline__ static float cospi(const float a) noexcept
        {
            return cospif(a);
        }
        
        __device__ __forceinline__ static void from_unsigned(float& f1, float& f2, const unsigned n) noexcept
        {
            static constexpr unsigned nbits = 8u * sizeof(unsigned);
            static constexpr unsigned mant_bits = std::numeric_limits<float>::digits;
            static constexpr unsigned mask_upper = ((1u << mant_bits) - 1u) << (nbits - mant_bits);;
            static constexpr unsigned mask_lower = ~mask_upper;

            f1 = n & mask_upper;
            f2 = n & mask_lower;
        }
    };

    // acquire block sequentializer in busy wait loop
    __device__ __forceinline__ void seq_acquire(unsigned& seq) noexcept
    {
        while (atomicCAS(&seq, 0u, 1u) != 0u);
    }

    // release block sequentializer
    __device__ __forceinline__ void seq_release(unsigned& seq) noexcept
    {
        __threadfence();
        __stwt(&seq, 0u);
    }
    
    // kahan sum
    // (a, rest) = a + b + rest
    template<typename float_type>
    __device__ __forceinline__ void ksum(float_type& a, float_type& rest, const float_type b) noexcept
    {
        const float_type s = rest + b;
        const float_type t = a;
        a = t + s;
        rest = s - (a - t);
    }

    // max of a, b, c
    template<typename float_type>
    __device__ __forceinline__ float_type max3(const float_type a, const float_type b, const float_type c) noexcept
    {
        const float_type t = (a < b) ? b : a;
        return (t < c) ? c : t;
    }

    // return value trimmed to the range [triml..trimh]
    // assume triml <= trimh
    template<typename float_type>
    __device__ __forceinline__ float_type trim(const fast_feedback::config_runtime<float_type>& crt, const float_type val) noexcept
    {
        return min(max(crt.triml, val), crt.trimh);
    }

    // return distance to nearest integer for value
    // The function assumes that 0 <= triml <= trimh <= 0.5
    template<typename float_type>
    __device__ __forceinline__ float_type dist2int(const float_type val) noexcept
    {
        return util<float_type>::abs(val - util<float_type>::rint(val));
    }

    // a 🞄 a
    template<typename float_type>
    __device__ __forceinline__ float_type norm2(const float_type a[3]) noexcept
    {
        return util<float_type>::fma(a[0], a[0], util<float_type>::fma(a[1], a[1], a[2] * a[2]));
    }

    // a 🞄 b
    template<typename float_type>
    __device__ __forceinline__ float_type dot(const float_type a[3], const float_type b[3]) noexcept
    {
        return util<float_type>::fma(a[0], b[0], util<float_type>::fma(a[1], b[1], a[2] * b[2]));
    }

    // a = b X c
    template<typename float_type>
    __device__ __forceinline__ void cross(float_type a[3], const float_type b[3], const float_type c[3]) noexcept
    {
        a[0] = util<float_type>::fma(b[1], c[2], -b[2] * c[1]);
        a[1] = util<float_type>::fma(b[2], c[0], -b[0] * c[2]);
        a[2] = util<float_type>::fma(b[0], c[1], -b[1] * c[0]);
    }

    // a = (a + l * b); a /= |a|
    template<typename float_type>
    __device__ __forceinline__ void add_unify(float_type a[3], const float_type b[3], const float_type l) noexcept
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
    __device__ __forceinline__ void mirror(float_type a[3], const float_type b[3]) noexcept
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
                                                  float_type &laz, float_type &laxy) noexcept
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
                                           const float_type alpha) noexcept
    {
        float_type s, c;
        util<float_type>::sincos(alpha, &s, &c);
        for (unsigned i=0u; i<3u; i++)
            a[i] = util<float_type>::fma(laz, z[i], (util<float_type>::fma(c, x[i], s * y[i]) * laxy));
    }

    // single threaded merge of block local sorted candidate vector array into global sorted candidate vector array
    template<typename float_type>
    __device__ __forceinline__ void merge_top_vecs(float_type* top_val, unsigned* top_sample, vec_cand_t<float_type>* cand, const unsigned n_cand) noexcept
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
    __device__ __forceinline__ void merge_top_cells(fast_feedback::output<float_type>& out, cell_cand_t<float_type>* cand, const unsigned n_cand) noexcept
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
    __device__ __forceinline__ void sample_point(const unsigned sample_idx, const unsigned n_samples, float_type v[3]) noexcept
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

        float_type x, y, z;
        float_type si1, si2, ns1, ns2;
        util<float_type>::from_unsigned(si1, si2, sample_idx);
        util<float_type>::from_unsigned(ns1, ns2, n_samples);
        const float_type dz = float_type{1.} / ns1 - ns2 / (ns1 * ns1 + ns1 * ns2);

        float_type rest = float_type{.0};
        z = float_type{1.};
        ksum(z, rest, -si1 * dz);
        ksum(z, rest, -si2 * dz);
        ksum(z, rest, -float_type{.5} * dz);
        const float_type r_xy = util<float_type>::sqrt(float_type{1.} - z * z);

        static_assert(sizeof(unsigned) == 4, "assumption about sizeof(unsigned) violated");
        float_type l = float_type{.0};
        for (unsigned i=0; i<32; i++) {
            if (sample_idx & (1u << i))
                ksum(l, rest, dl2pNmod2pi<float_type>[i]);
        }
        util<float_type>::sincos(l, &y, &x);

        x *= r_xy;
        y *= r_xy;
        v[0] = x;
        v[1] = y;
        v[2] = z;
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
                                                const unsigned cell_vec) noexcept
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
        const float_type alpha = rsample * constant<float_type>::pi2 / static_cast<float_type>(n_rsamples); // sample angle
        rotate(a, x, y, z, laz, laxy, alpha);
        rotate(b, x, y, z, lbz, lbxy, alpha + delta);
    }

    // sum(s ∈ spots) log2(trim[triml..trimh](dist2int(s 🞄 v / vlength)) + delta)
    // v            unit vector in sample vector direction, |v| == 1
    // vlength      sample vector length
    // s{x,y,z}     spot coordinate pointers [n_spots]
    // trim{l,h}    lower trim value
    // n_spots      number of spots
    template<typename float_type>
    __device__ __forceinline__ float_type sample1(const fast_feedback::config_runtime<float_type>& crt,
                                                  const float_type v[3], const float_type vlength,
                                                  const float_type *sx, const float_type *sy, const float_type *sz,
                                                  const unsigned n_spots) noexcept
    {
        float_type sval = float_type{0.f};
        float_type rest = float_type{0.f};
        // unsigned n_good = 0u;

        const float_type delta = crt.delta;
        for (unsigned i=0u; i<n_spots; i++) {
            const float_type s[3] = { sx[i], sy[i], sz[i] };
            const float_type d = dist2int(vlength * dot(v, s));
            // n_good += (d < crt.trimh) ? 1u : 0u;
            const float_type dv = util<float_type>::log2(trim(crt, d) + delta);
            ksum(sval, rest, dv);
        }
        // return util<float_type>::exp2(sval / (float_type)n_spots) - delta - (float_type)n_good;
        return sval;
    }

    // sum(s ∈ spots) sum(log2(trim[triml..trimh](sqrt(sum[i=a,b,c](dist2int(s 🞄 vi / |vi|²)²))) + delta))
    // crt          runtime configuration with triml/h and delta
    // z, a, b      sample vectors, z is c normalized
    // s{x,y,z}     spot coordinate pointers [n_spots]
    // lz           length of vector c, c = z * lz
    // n_spots      number of spots
    template<typename float_type>
    __device__ __forceinline__ float_type sample3(const fast_feedback::config_runtime<float_type>& crt,
                                                  const float_type z[3], const float_type a[3], const float_type b[3],
                                                  const float_type *sx, const float_type *sy, const float_type *sz,
                                                  const float_type lz, const unsigned n_spots) noexcept
    {
        float_type sval = float_type{0.f};
        float_type rest = float_type{0.f};
        unsigned n_good = 0u;

        const float_type delta = crt.delta;
        for (unsigned i=0u; i<n_spots; i++) {
            const float_type s[3] = { sx[i], sy[i], sz[i] };
            const float_type cc = dist2int(lz * dot(z, s));
            const float_type ca = dist2int(dot(a, s));
            const float_type cb = dist2int(dot(b, s));
            const float_type dn = util<float_type>::norm(cc, ca, cb);
            n_good += (dn < crt.trimh) ? 1u : 0u;
            const float_type dv = util<float_type>::log2(trim(crt, dn) + delta);
            ksum(sval, rest, dv);
        }
        return util<float_type>::exp2(sval / (float_type)n_spots) - delta - (float_type)n_good;
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
            for (unsigned i=0u; i<n_cells_in; ++i) {
                printf("input cell%u rep %u, cgi:", i, data->cell_to_cellvec[i]);
                for (unsigned j=3*i; j<3*i+3; ++j) {
                    unsigned cg = data->cellvec_to_cand[j];
                    printf(" %u", cg);
                    if (cg > ncg)
                        ncg = cg;
                }
                printf("\n");
            }

            printf("redundant_comp=%u, rcgrps:", unsigned(data->cpers.redundant_computations));
            for (unsigned i=0u; i<n_cells_in; ++i)
                printf(" %u", data->vec_cgrps[i]);
            printf("\n");

            for (unsigned i=0u; i<=ncg; ++i) {
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
        for (unsigned i=0u; i<n_cells_out; ++i) {
            printf("output cell%u s=%f:\n", i, out.score[i]);
            for (unsigned j=3u*i; j<3u*i+3u; ++j) {
                if (kind == 1u) {
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
    // for non-redundant (cpers.redundant_computations=false) calculations:
    //      representing vector candidate group = blockIdx.y
    // for redundant computations:
    //      candidate group index = blockIdx.y
    template<typename float_type>
    __global__ void gpu_find_candidates(indexer_device_data<float_type>* data)
    {
        extern __shared__ double* shared_ptr[];

        unsigned sample = blockDim.x * blockIdx.x + threadIdx.x;
        const unsigned n_samples = data->crt.num_halfsphere_points;
        const unsigned c_group = data->cpers.redundant_computations ? blockIdx.y : data->vec_cgrps[blockIdx.y];
        float_type v = 0.;  // objective function value for sample vector

        if (sample < n_samples) {                                   // calculate v
            const float_type sl = data->candidate_length[c_group];  // sample vector length
            float_type sv[3];                                       // unit vector in sample direction
            sample_point(sample, n_samples, sv);

            const fast_feedback::input<float_type>& in = data->input;
            const unsigned n_spots = in.n_spots;
            v = sample1(data->crt, sv, sl, in.spot.x, in.spot.y, in.spot.z, n_spots);
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

    #if VCANDREF == VCR_ROPT
        // vector candidate index = blockIdx.x
        // for non-redundant (cpers.redundant_computations=false) calculations:
        //      input cell index = blockIdx.y
        // for redundant computations:
        //      cell vector index = blockIdx.y
        template<typename float_type>
        __global__ void gpu_refine_cand(indexer_device_data<float_type>* data)
        {
            // if ((blockIdx.x == 0) && (blockIdx.y == 0) && (threadIdx.x == 0)) {
            //     printf("gpu_refine_cand() threads=%u, cands=%u, ncg=%u\n", (unsigned)blockDim.x, (unsigned)gridDim.x, (unsigned)gridDim.y);
            // }
        }
    #endif

    // sample rotation angle index = blockDim.x * blockIdx.x + threadIdx.x
    // for non-redundant (cpers.redundant_computations=false) calculations:
    //      input cell index = blockIdx.y
    // for redundant computations:
    //      cell vector index = blockIdx.y
    // vector candidate index = blockIdx.z
    template<typename float_type>
    __global__ void gpu_find_cells(indexer_device_data<float_type>* data)
    {
        extern __shared__ double* shared_ptr[];

        const unsigned n_vsamples = data->crt.num_halfsphere_points;
        unsigned rsample = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned cell_vec = data->cpers.redundant_computations ? blockIdx.y : data->cell_to_cellvec[blockIdx.y];
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

            sample_cell(z, a, b, in.cell.x, in.cell.y, in.cell.z, vlength, vsample, n_vsamples, rsample, n_rsamples, cell_vec);

            const unsigned n_spots = in.n_spots;
            vabc = sample3(data->crt, z, a, b, in.spot.x, in.spot.y, in.spot.z, vlength, n_spots);
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
    // n_rsamples   number of rotation angle samples rounded up to n_threads (must match n_rsamples in gpu_find_cells)
    template<typename float_type>
    __global__ void gpu_expand_cells(indexer_device_data<float_type>* data, const unsigned n_rsamples)
    {
        const unsigned n_vsamples = data->crt.num_halfsphere_points;
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

        sample_cell(z, a, b, in.cell.x, in.cell.y, in.cell.z, vlength, vsample, n_vsamples, rsample, n_rsamples, cell_vec);

        const unsigned iz = cell_vec % 3u;
        const unsigned ia = (iz + 1u) % 3u;
        const unsigned ib = (iz + 2u) % 3u;
        ox[cell_base + iz] = z[0] * vlength;
        ox[cell_base + ia] = a[0];
        ox[cell_base + ib] = b[0];
        oy[cell_base + iz] = z[1] * vlength;
        oy[cell_base + ia] = a[1];
        oy[cell_base + ib] = b[1];
        oz[cell_base + iz] = z[2] * vlength;
        oz[cell_base + ia] = a[2];
        oz[cell_base + ib] = b[2];
    }

} // anonymous namespace

namespace gpu {

    template <typename float_type>
    void init (const indexer<float_type>& instance)
    {
        using gpu_state = indexer_gpu_state<float_type>;
        using device_data = indexer_device_data<float_type>;

        gpu_device::init();
        gpu_debug_init();

        const auto state_id = instance.state;
        if (gpu_state::exists(state_id))
            throw FF_EXCEPTION("instance illegal reinitialisation");

        const auto& cpers = instance.cpers;

        device_data* data = nullptr;
        float_type* candidate_length = nullptr;
        float_type* candidate_value = nullptr;
        float_type* refined_candidates = nullptr;
        unsigned* candidate_sample = nullptr;
        unsigned* cellvec_to_cand = nullptr;
        unsigned* cell_to_cellvec = nullptr;
        unsigned* vec_cgrps = nullptr;
        unsigned* sequentializers = nullptr;
        float_type* elements = nullptr;
        float_type* ix;
        float_type* iy;
        float_type* iz;
        float_type* ox;
        float_type* oy;
        float_type* oz;
        float_type* scores;

        int dev = gpu_device::get();
        gpu_device::set(dev);

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
                static_assert(mem_txn_unit >= 3 * sizeof(float_type));
                std::size_t vector_dimension = constant<float_type>::mem_txn_unit * cpers.num_candidate_vectors * cpers.max_input_cells;
                CU_CHECK(cudaMalloc(&refined_candidates, vector_dimension * sizeof(float_type)));
            }
            gpu_pointer<float_type> refined_candidates_ptr{refined_candidates};

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

            {
                std::size_t vector_dimension = cpers.max_input_cells;
                CU_CHECK(cudaMalloc(&cell_to_cellvec, vector_dimension * sizeof(unsigned)));
            }
            gpu_pointer<unsigned> c2v_ptr{cell_to_cellvec};

            {
                std::size_t vector_dimension = cpers.max_input_cells;
                CU_CHECK(cudaMalloc(&vec_cgrps, vector_dimension * sizeof(unsigned)));
            }
            gpu_pointer<unsigned> vcgrp_ptr{vec_cgrps};

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
                                                     #if VCANDREF == VCR_ROPT
                                                        std::move(refined_candidates_ptr),
                                                     #endif
                                                     std::move(candidate_sample_ptr), std::move(v2c_ptr),
                                                     std::move(c2v_ptr), std::move(vcgrp_ptr),
                                                     std::move(sequentializers_ptr), std::move(stream),
                                                     ix, iy, iz,
                                                     ox, oy, oz,
                                                     scores, dev};
            }

            LOG_START(logger::l_debug) {
                logger::debug << stanza << "init id=" << state_id << " on " << dev << ", data=" << data_ptr.get() << '\n'
                              << stanza << "  cand len=" << candidate_length << ", val=" << candidate_value << ", smpl=" << candidate_sample << '\n'
                              << stanza << "  vec2cand=" << cellvec_to_cand << ", cell2vec=" << cell_to_cellvec << ", vec_cgrp=" << vec_cgrps << '\n'
                              << stanza << "  seq=" << sequentializers << ", elements=" << elements << ", score=" << scores << '\n'
                              << stanza << "    ox=" << ox << ", oy=" << oy << ", oz=" << oz << '\n'
                              << stanza << "    ix=" << ix << ", iy=" << iy << ", iz=" << iz << '\n';
            } LOG_END;
        }

        {
            const unsigned so = 3 * cpers.max_input_cells; // spot offset
            device_data _data{cpers, fast_feedback::config_runtime<float_type>{},
                              fast_feedback::input<float_type>{{ix, iy, iz}, {&ix[so], &iy[so], &iz[so]}, 0, 0, true, true},
                              fast_feedback::output<float_type>{ox, oy, oz, scores, 0},
                              candidate_length, candidate_value,
                              #if VCANDREF == VCR_ROPT
                                refined_candidates,
                              #endif
                              candidate_sample, cellvec_to_cand,
                              cell_to_cellvec, vec_cgrps,
                              sequentializers};
            CU_CHECK(cudaMemcpyAsync(data, &_data, sizeof(_data), cudaMemcpyHostToDevice, gpu_state::stream(state_id)));
        }
    }

    // Drop state if any
    template <typename float_type>
    void drop (const indexer<float_type>& instance)
    {
        using gpu_state = indexer_gpu_state<float_type>;

        gpu_device::init();
        gpu_state::drop(instance.state);
    }

    // Run indexer asynchronously
    template <typename float_type>
    void index_start (const indexer<float_type>& instance, const input<float_type>& in, output<float_type>& out, const config_runtime<float_type>& conf_rt,
                      void(*host_callback)(void*), void* callback_data)
    {
        using gpu_state = indexer_gpu_state<float_type>;
        using clock = std::chrono::high_resolution_clock;

        gpu_device::check_init();
        auto state_id = instance.state;
        auto& state = gpu_state::ref(state_id);

        gpu_device::set(state.device);

        bool timing = logger::level_active<logger::l_info>();
        if (timing) {
            state.start_time = clock::now();
            state.start.init();
            state.end.init();
        }

        // Check input/output
        const auto n_cells_in = std::min(in.n_cells, instance.cpers.max_input_cells);
        const auto n_cells_out = std::min(out.n_cells, instance.cpers.max_output_cells);
        if (n_cells_in <= 0u)
            throw FF_EXCEPTION("no given input cells");
        if (n_cells_out <= 0)
            throw FF_EXCEPTION("no output cells");
        if (in.n_spots <= 0u)
            throw FF_EXCEPTION("no spots");
        if (instance.cpers.num_candidate_vectors < 1u)
            throw FF_EXCEPTION("nonpositive number of candidate vectors");
        if (conf_rt.num_halfsphere_points < instance.cpers.num_candidate_vectors)
            throw FF_EXCEPTION("fewer halfsphere sample points than required candidate vectors");
        if ((conf_rt.num_angle_points > 0u) && (conf_rt.num_angle_points < instance.cpers.max_output_cells))
            throw FF_EXCEPTION("fewer angle sample points than required candidate cells");
        if (conf_rt.delta <= .0f)
            throw FF_EXCEPTION("nonpositive delta value in runtime configuration");
        if (conf_rt.triml >= conf_rt.trimh)
            throw FF_EXCEPTION("lower trim value bigger than higher trim value");
        if (conf_rt.triml < .0f)
            throw FF_EXCEPTION("negative lower trim value");
        if (n_cells_out > n_threads)
            throw FF_EXCEPTION("fewer threads in a block than output cells");
        if (instance.cpers.num_candidate_vectors > n_threads)
            throw FF_EXCEPTION("fewer threads in a block than candidate vectors");
        
        // Calculate input vector candidate groups
        std::vector<unsigned> cell_candidate(n_cells_in);                           // cell -> chosen vector idx
        std::vector<unsigned> vec_cgrps(n_cells_in);                                // chosen vector candidate groups
        std::vector<unsigned> candidate_idx(3u * n_cells_in);                       // vector -> cand group idx
        std::vector<float_type> candidate_length(3u * n_cells_in, float_type{});    // cand group length (sorted !)
        unsigned n_cand_groups = calc_cand_groups(candidate_idx, candidate_length, in, conf_rt, n_cells_in);
        unsigned n_vec_cgrps = calc_cell_cand(cell_candidate, vec_cgrps, candidate_idx, n_cells_in);
        LOG_START(logger::l_debug) {
            logger::debug << stanza << "index on " << state.device << ", n_cells = " << n_cells_in << "(in:" << (in.new_cells?"new":"old") << ")/"
                                    << n_cells_out << "(out), n_spots = " << in.n_spots << (in.new_spots?"new":"old") << '\n'
                          << stanza << "  candidate_idx =";
            for (const auto& e : candidate_idx)
                logger::debug << ' ' << e;
            logger::debug << ", n_cand_groups = " << n_cand_groups << '\n';
            logger::debug << stanza << "cell_candidates =";
            for (const auto& e : cell_candidate)
                logger::debug << ' ' << e;
            logger::debug << ", vec_cgrps =";
            for (unsigned i=0u; i<n_vec_cgrps; i++)
                logger::debug << ' ' << vec_cgrps[i];
            logger::debug << ", n_vec_cgrps = " << n_vec_cgrps << '\n';
        } LOG_END;

        gpu_stream& stream = gpu_state::stream(state_id);
        {   // find vector candidates
            const unsigned n_samples = conf_rt.num_halfsphere_points;
            const dim3 n_blocks((n_samples + n_threads - 1) / n_threads,                                // samples
                                instance.cpers.redundant_computations ? n_cand_groups : n_vec_cgrps);   // number of (cell representing vector) candidate groups
            const unsigned shared_sz = std::max(instance.cpers.num_candidate_vectors * sizeof(vec_cand_t<float_type>),
                                                sizeof(typename BlockRadixSort<float_type>::TempStorage));
            gpu_state::copy_crt(state_id, conf_rt, stream);
            gpu_state::copy_in(state_id, instance.cpers, in, out, stream);
            gpu_state::init_cand(state_id, n_cand_groups, n_vec_cgrps, instance.cpers, candidate_length, candidate_idx, cell_candidate, vec_cgrps, stream);
            state.start.record(stream);
            gpu_find_candidates<float_type><<<n_blocks, n_threads, shared_sz, stream>>>(gpu_state::ptr(state_id).get());
        }
        #if VCANDREF == VCR_ROPT
        {
            const dim3 n_blocks{
                instance.cpers.num_candidate_vectors,                                   // num cand vecs
                instance.cpers.redundant_computations ? 3u * n_cells_in : n_cells_in,   // num cell vectors / num cells
            };
            gpu_refine_cand<float_type><<<n_blocks, warp_size, 0, stream>>>(gpu_state::ptr(state_id).get());
        }
        #endif
        {   // find cells
            const unsigned n_xblocks = (conf_rt.num_angle_points == 0u ?
                                            (2.5 * std::sqrt(conf_rt.num_halfsphere_points) + n_threads - 1.) / n_threads // 2*pi*r^2 (half sphere) --> 2*pi*r (circumference)
                                           :(conf_rt.num_angle_points + n_threads - 1u) / n_threads);
            const dim3 n_blocks(n_xblocks,                                                              // rotation samples
                                instance.cpers.redundant_computations ? 3u * n_cells_in : n_cells_in,   // num cell vectors / num cells
                                instance.cpers.num_candidate_vectors);                                  // num cand vecs
            const unsigned shared_sz = std::max(n_cells_out * sizeof(cell_cand_t<float_type>),
                                                sizeof(typename BlockRadixSort<float_type>::TempStorage));
            bool dbg_flag = gpu_debug_output.load();
            if (dbg_flag)
                gpu_debug_out<float_type><<<1, 1, 0, stream>>>(gpu_state::ptr(state_id).get(), 0u);
            gpu_find_cells<float_type><<<n_blocks, n_threads, shared_sz, stream>>>(gpu_state::ptr(state_id).get());
            if (dbg_flag)
                gpu_debug_out<float_type><<<1, 1, 0, stream>>>(gpu_state::ptr(state_id).get(), 1u);
            gpu_expand_cells<float_type><<<1, n_cells_out, 0, stream>>>(gpu_state::ptr(state_id).get(), n_xblocks * n_threads);
            if (dbg_flag)
                gpu_debug_out<float_type><<<1, 1, 0, stream>>>(gpu_state::ptr(state_id).get(), 2u);
            state.end.record(stream);
        }
        if (host_callback != nullptr) {
            state.callback_mode = true;
            CU_CHECK(cudaLaunchHostFunc(stream, host_callback, callback_data));
        } else
            state.callback_mode = false;
    }

    template <typename float_type>
    void index_end (const indexer<float_type>& instance, output<float_type>& out)
    {
        using gpu_state = indexer_gpu_state<float_type>;
        using clock = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<double, std::milli>;
        using time_point = std::chrono::time_point<clock>;

        auto state_id = instance.state;
        auto& state = gpu_state::ref(state_id);

        gpu_stream& stream = state.cuda_stream;

        if (! state.callback_mode)
            stream.sync();
        state.callback_mode = false;

        gpu_state::copy_out(state_id, out, stream); // synchronizes on stream
        if (logger::level_active<logger::l_info>()) {
            time_point end = clock::now();
            duration elapsed = end - state.start_time;
            LOG_START(logger::l_info) {
                logger::info << stanza << "indexing_time: " << elapsed.count() << "ms\n";
                logger::info << stanza << "kernel_time: " << gpu_timing(state.start, state.end) << "ms\n";
            } LOG_END;
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
    template void index_start<float> (const indexer<float>&, const input<float>&, output<float>&, const config_runtime<float>&, void(*)(void*), void*);
    template void index_end<float> (const indexer<float>&, output<float>&);

} // namespace gpu
