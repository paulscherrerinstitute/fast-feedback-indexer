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

#include <ffbidx/indexer.h>
#include <getopt.h>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <cerrno>
#include <string>
#include <sys/cdefs.h>
#include <sys/types.h>
#include <vector>
#include <thread>
#include <Eigen/Dense>
#include "cuda_runtime.h"
#include "ffbidx/simple_data.h"
#include "ffbidx/refine.h"

using namespace fast_feedback;

namespace {
    using clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<clock>;
    using duration = std::chrono::duration<double>;

    constexpr struct success_type final {} success;
    constexpr struct failure_type final {} failure;

    template <typename stream>
    [[noreturn]] stream& operator<< (stream& out, [[maybe_unused]] const success_type& data)
    {
        out.flush();
        std::exit((EXIT_SUCCESS));
    }

    template <typename stream>
    [[noreturn]] stream& operator<< (stream& out, [[maybe_unused]] const failure_type& data)
    {
        out.flush();
        std::exit((EXIT_FAILURE));
    }

    [[noreturn]] void error (std::string msg)
    {
        std::cout << "error: " << msg << '\n' << failure;
    }

    [[noreturn]] void usage (std::string msg = {})
    {
        std::cout << "usage: " << program_invocation_short_name << " --method=(raw|ifssr|ifss|ifse) [options] file1 file2 ...\n\n"
                     "  Index simple data files.\n\n"
                     "output options:\n"
                     "  --help         show this help\n"
                     "  --quiet        no indexing result output\n"
                     "  --notiming     no timing output\n"
                     "  --allcells     output all cells instead of only the best\n"
                     "  --cellgeom     output cell geometry\n"
                     "  --crystals     output crystal indices\n"
                     "indexer options:\n"
                     "  --method       output cell refinement method, one of raw, ifssr(default), ifss, ifse\n"
                     "  --reducalc     calculate candidate vectors for all cell vectors instead of one\n"
                     "  --maxspots     maximum number of spots\n"
                     "  --cells        number of output cells with scores\n"
                     "  --cands        number of candidate vectors\n"
                     "  --hsp          number of brute force halfsphere sample points\n"
                     "  --ap           number of brute force angle sample points (0: heuristic)\n"
                     "  --triml        trim lows\n"
                     "  --trimh        trim heights\n"
                     "  --delta        log2 curve position\n"
                     "  --dist1        vector score: max inlier distance\n"
                     "  --dist3        cell score: max inlier distance\n"
                     "  --vrminpts     vector candidate refinement: minimum number of points for fitting (0: no refinement)\n"
                     "  --contr        ifssr/ifss/ifse threshold contraction\n"
                     "  --maxdist      ifssr/ifss/ifse maximum distance to integer\n"
                     "  --minpts       ifssr/ifss/ifse minimum number of points for fitting\n"
                     "  --iter         ifssr/ifss/ifse maximum iterations\n"
                     "pipeline options:\n"
                     "  --gpus         comma separated list of gpu idranges. range = single number or dash separated numbers\n"
                     "  --ipg          indexer objects per gpu\n"
                     "  --ths          worker threads\n"
                     "  --rblks        refinement blocks\n"
                     "  --rep          repetitions, every file will be indexed that many times\n\n";
        if (! msg.empty())
            error(msg);
        std::cout << success;
    }

    // global program argument variables
    std::vector<std::string> files;     // list of simple data files
    std::vector<unsigned> gpus;         // list of cuda device ids
    unsigned maxspot;
    unsigned ncells;
    unsigned cands;
    unsigned halfsphere_points;
    unsigned angle_points;
    float triml;
    float trimh;
    float delta;
    float dist1;
    float dist3;
    float contr;
    float maxdist;
    unsigned vrminpts;
    unsigned minpts;
    unsigned iter;
    unsigned worker_threads = 1u;       // number of worker threads
    unsigned refinement_blocks = 1u;    // number of output cell blocks for parallel refinement
    unsigned indexers_per_gpu = 1u;     // number of indexer objects per cuda device
    unsigned repetitions = 1u;          // number of times each file is indexed
    bool notiming = false;              // don't print out timing output
    bool quiet = false;                 // don't produce indexing result output
    bool reducalc = false;              // calculate candidates for all 3 cell vectors instead of one
    bool best_only = true;              // output best cell only
    bool cell_geom = false;             // output cell geometry
    bool crystals = false;              // output crystal indices
    std::string method{};               // refinement method

    void check_method()
    {
        if (method.empty())
            method = "ifssr";
        else if ((method != "ifss") && (method != "ifse") && (method != "ifssr") && (method != "raw"))
            error(std::string("unsupported method: ") + method);
    }

    template<typename T>
    void parse_val(T& val, const std::string& opt, const std::string& for_method="")
    {
        if (! for_method.empty() && (method != for_method))
            error(std::string("option ")+opt+" requires method "+for_method);
        std::istringstream iss(opt);
        iss >> val;
        if (! iss.eof())
            usage(std::string("invalid value: '") + opt + "'");
    }

    void parse_gpus_range(std::vector<unsigned>& gpus, const std::string& opt)
    {
        using size_type = std::string::size_type;

        unsigned start, end;
        size_type pos = opt.find('-');
        parse_val(start, opt.substr(0, pos));
        if (pos < opt.size()) {
            parse_val(end, opt.substr(pos + 1));
            if (start >= end)
                usage(std::string("invalid gpus range: ") + opt);
            for (; start<=end; start++)
                gpus.push_back(start);
        } else {
            gpus.push_back(start);
        }
    }

    void parse_gpus_list(std::vector<unsigned>& gpus, const std::string& opt)
    {
        using size_type = std::string::size_type;

        size_type start=0, end=std::string::npos;
        do {
            end = opt.find(',', start);
            parse_gpus_range(gpus, opt.substr(start, end - start));
            start = end + 1;
        } while (end < opt.size());
    }

    // parse program arguments and store values into the global prog arg vars
    void argparse(int argc, char* argv[])
    {
        const option opt[] = {
            { "gpus",     1, nullptr, 0 },
            { "maxspots", 1, nullptr, 1 },
            { "cells",    1, nullptr, 2 },
            { "cands",    1, nullptr, 3 },
            { "hsp",      1, nullptr, 4 },
            { "ap",       1, nullptr, 5 },
            { "triml",    1, nullptr, 6 },
            { "trimh",    1, nullptr, 7 },
            { "delta",    1, nullptr, 8 },
            { "dist1",    1, nullptr, 9 },
            { "dist3",    1, nullptr, 10},
            { "contr",    1, nullptr, 11},
            { "maxdist",  1, nullptr, 12},
            { "minpts",   1, nullptr, 13},
            { "iter",     1, nullptr, 14},
            { "ths",      1, nullptr, 15},
            { "rblks",    1, nullptr, 16},
            { "ipg",      1, nullptr, 17},
            { "rep",      1, nullptr, 18},
            { "quiet",    0, nullptr, 19},
            { "method",   1, nullptr, 20},
            { "reducalc", 0, nullptr, 21},
            { "allcells", 0, nullptr, 22},
            { "cellgeom", 0, nullptr, 23},
            { "vrminpts", 1, nullptr, 24},
            { "notiming", 0, nullptr, 25},
            { "crystals", 0, nullptr, 26},
            { "help",     0, nullptr, 27},
            { nullptr,    0, nullptr, -1}
        };

        do {
            int oidx = -1;
            if (getopt_long_only(argc, argv, "", opt, &oidx) < 0)
                break;
            if (oidx == -1)
                usage("option parsing failed");
            switch (oidx) {
                case 0:
                    parse_gpus_list(gpus, optarg); break;
                case 1:
                    parse_val(maxspot, optarg); break;
                case 2:
                    parse_val(ncells, optarg); break;
                case 3:
                    parse_val(cands, optarg); break;
                case 4:
                    parse_val(halfsphere_points, optarg); break;
                case 5:
                    parse_val(angle_points, optarg); break;
                case 6:
                    parse_val(triml, optarg); break;
                case 7:
                    parse_val(trimh, optarg); break;
                case 8:
                    parse_val(delta, optarg); break;
                case 9:
                    parse_val(dist1, optarg); break;
                case 10:
                    parse_val(dist3, optarg); break;
                case 11:
                    parse_val(contr, optarg); break;
                case 12:
                    parse_val(maxdist, optarg); break;
                case 13:
                    parse_val(minpts, optarg); break;
                case 14:
                    parse_val(iter, optarg); break;
                case 15:
                    parse_val(worker_threads, optarg);
                    if (worker_threads < 1u)
                        error("no worker threads");
                    break;
                case 16:
                    parse_val(refinement_blocks, optarg);
                    if (refinement_blocks < 1u)
                        error("no refinement blocks");
                    break;
                case 17:
                    parse_val(indexers_per_gpu, optarg);
                    if (indexers_per_gpu < 1u)
                        error("no indexers");
                    break;
                case 18:
                    parse_val(repetitions, optarg);
                    if (repetitions < 1u)
                        error("no repetitions");
                    break;
                case 19:
                    quiet = true;
                    break;
                case 20:
                    if (! method.empty())
                        error("method already set");
                    parse_val(method, optarg);
                    break;
                case 21:
                    reducalc = true;
                    break;
                case 22:
                    best_only = false;
                    break;
                case 23:
                    cell_geom = true;
                    break;
                case 24:
                    parse_val(vrminpts, optarg); break;
                case 25:
                    notiming = true;
                    break;
                case 26:
                    crystals = true;
                    break;
                case 27:
                    usage();
                default:
                    error("internal: unknown option id");
            }
        } while (true);
        check_method();
        for (; optind<argc; optind++)
            files.emplace_back(argv[optind]);
        if (files.empty())
            usage("no data files");
    }

    // initialize global program argument variables to defaults
    void initargs(const config_persistent<float>& cpers,
                  const config_runtime<float>& crt) noexcept
    {
        maxspot = cpers.max_spots;
        ncells = cpers.max_output_cells;
        cands = cpers.num_candidate_vectors;
        halfsphere_points = crt.num_halfsphere_points;
        angle_points = crt.num_angle_points;
        triml = crt.triml;
        trimh = crt.trimh;
        delta = crt.delta;
        dist1 = crt.dist1;
        dist3 = crt.dist3;
        vrminpts = crt.min_spots;
        // initialize these later, set to invalid here
        contr = -1.;
        maxdist = -1.;
        minpts = 0u;
        iter = 0u;
    }

    // check global program argument variables
    // TODO
    void checkargs()
    {
        if (refinement_blocks > worker_threads)
            error("more refinement blocks than worker threads");
        ;   // TODO
    }

    // sync fast feedback indexer configuration structs with global prog arg vars
    void setconf(config_persistent<float>& cpers,
                 config_runtime<float>& crt,
                 refine::config_ifss<float>& cifss,
                 refine::config_ifse<float>& cifse,
                 refine::config_ifssr<float>& cifssr)
    {
        cpers.max_spots = maxspot;
        cpers.max_output_cells = ncells;
        cpers.num_candidate_vectors = cands;
        cpers.redundant_computations = reducalc;
        crt.num_halfsphere_points = halfsphere_points;
        crt.num_angle_points = angle_points;
        crt.triml = triml;
        crt.trimh = trimh;
        crt.delta = delta;
        crt.dist1 = dist1;
        crt.dist3 = dist3;
        crt.min_spots = vrminpts;
        if (contr >= .0f)
            cifss.threshold_contraction = cifse.threshold_contraction = cifssr.threshold_contraction = contr;
        if (maxdist >= .0f)
            cifss.max_distance = cifse.max_distance = cifssr.max_distance = maxdist;
        if (minpts > 0u)
            cifss.min_spots = cifse.min_spots = cifssr.min_spots = minpts;
        if (iter == 0u)
            cifss.max_iter = cifse.max_iter = cifssr.max_iter = iter;
    }

    // cyclic nonnegative integer queue
    // writers ensure by external means there are never more than size items in the queue
    // items are nonnegative integers
    class CNIQ final {
        constexpr static int invalid = 0;                       // invalid ticket
        constexpr static int valid = 1;                         // valid ticket
        std::vector<std::unique_ptr<std::atomic_int>> items;    // always at least one slot free
        std::vector<std::unique_ptr<std::atomic_int>> tickets;  // ticket fore every slot
        std::atomic_uint head;                                  // 1 before front
        std::atomic_uint tail;                                  // end
        unsigned cap;                                           // capacity

      public:
        constexpr static int free = -1;                         // free slot value

        explicit CNIQ(unsigned size=0u)
        {
            reset(size);
        }

        void reset (unsigned size)
        {
            cap = size+1u;
            items.resize(cap);
            tickets.resize(cap);
            for (auto& ptr: items)
                ptr.reset(new std::atomic_int{free});
            for (auto& ptr: tickets)
                ptr.reset(new std::atomic_int{invalid});
            head = 0u;
            tail = 0u;
        }

        void push_front (int item) noexcept
        {
            int e;
            unsigned h = head.load();
            while (! head.compare_exchange_weak(h, (h + cap - 1u) % cap));
            do {
                e = free;
            } while (! items[h]->compare_exchange_strong(e, item));
        }

        void push_back (int item) noexcept
        {
            int e;
            unsigned t = tail.load();
            while (! tail.compare_exchange_weak(t, (t + 1u) % cap));
            t = (t + 1u) % cap;
            do {
                e = free;
            } while (! items[t]->compare_exchange_strong(e, item));
        }

        int pop_back () noexcept
        {
            int e;
            int ticket;
            unsigned s;
            unsigned t;
            do {
                t = tail.load();
                if (head.load() == t)
                    return free;
                ticket = invalid;
                if (! tickets[t]->compare_exchange_weak(ticket, valid))
                    continue;
                s = t;
                e = items[t]->load();
                if ((e != free) && tail.compare_exchange_strong(t, (t + cap - 1u) % cap))
                    break;
                tickets[s]->store(invalid);
            } while (true);
            items[s]->store(free);
            tickets[s]->store(invalid);
            return e;
        }
   }; // CNIQ

    using indexer_t = indexer<float>;
    using input_t = input<float>;
    using output_t = output<float>;
    using cfgrt_t = config_runtime<float>;
    using cfgps_t = config_persistent<float>;
    using indexer_ifss = refine::indexer_ifss<float>;
    using indexer_ifse = refine::indexer_ifse<float>;
    using indexer_ifssr = refine::indexer_ifssr<float>;
    using cifss_t = refine::config_ifss<float>;
    using cifse_t = refine::config_ifse<float>;
    using cifssr_t = refine::config_ifssr<float>;
    using mempin_t = memory_pin;
    using SimpleData = simple_data::SimpleData<float, simple_data::raise>;
    using Mx3 = Eigen::Matrix<float, Eigen::Dynamic, 3u>;
    using Vx = Eigen::Vector<float, Eigen::Dynamic>;

    std::vector<std::unique_ptr<indexer_t>> indexer;    // indexer[gpu * indexers_per_gpu], array of indexer objects
    CNIQ indexer_idle;                                  // queue of indices to idle indexer objects

    enum state_t : int {    // work item next step states
        read_file,          // read the input data from simple data file
        index_start,        // launch indexer object asynchronously
        index_end,          // finish indexing step
        refine_block,       // refine one block of output cells
        finished
    };

    // state_t --> std::string
    std::string to_string (const state_t& st)
    {
        switch (st) {
            case read_file:
                return "read_file";
            case index_start:
                return "index_start";
            case index_end:
                return "index_end";
            case refine_block:
                return "refine_block";
            case finished:
                return "finished";
            default:
                return std::string("invalid-") + std::to_string(st);
        }
    }

    // item of work progressed by the worker threads
    struct work_item final {
        std::string filename;                   // simple data file name
        std::unique_ptr<SimpleData> data_ptr;   // pointer to data object obtained with simple_data lib
        std::vector<unsigned> crystals;         // crystal indices
        Mx3 coords;                             // one unit cell plus spot coordinates
        Mx3 cells;                              // output cells
        Vx scores;                              // output cell scores
        input_t in;                             // indexer input
        output_t out;                           // indexer output
        mempin_t pin_coords;                    // pin object for coords matrix
        mempin_t pin_cells;                     // pin object for cells matrix
        mempin_t pin_scores;                    // pin object for scores vector
        unsigned best_cell;                     // best cell output

        unsigned repetition = 0u;               // repetition counter

        unsigned rblock;                        // output cell block number to be refined next
        std::atomic_uint blkcnt;                // output cell block refined counter
        int id;                                 // index into work item list

        time_point tp;                          // start time of a work item step
        int indexer = CNIQ::free;               // associated indexer object
        state_t state = read_file;              // work item next step state

        work_item (const std::string& fname, int wid)
            : filename{fname}, coords{3u + maxspot, 3u},
              cells{3u * ncells, 3u}, scores{ncells},
              in{{&coords(0,0), &coords(0,1), &coords(0,2)}, {&coords(3,0), &coords(3,1), &coords(3,2)}, 1u, maxspot, true, true},
              out{&cells(0,0), &cells(0,1), &cells(0,2), scores.data(), ncells},
              pin_coords{coords}, pin_cells{cells}, pin_scores(scores), rblock{0u}, blkcnt{0}, id{wid}
        {}
    };

    std::vector<std::unique_ptr<work_item>> witem_list; // list of work items
    CNIQ work_queue;                                    // queue of indices to progressible work items

    std::vector<std::thread> thread_pool;               // pool of worker threads
    std::atomic_bool pool_start = false;                // worker threads start switch
    std::atomic_uint counter = 0u;                      // counter for finished work items

    std::atomic<double> read_time{.0};                  // accumulated simple data file read time
    std::atomic<double> indexer_time{.0};               // accumulated indexing time
    std::atomic<double> refine_time{.0};                // accumulated output cell refinement time

    // atomic add for double
    void atomic_add (std::atomic<double>& dest, double val)
    {
        double old_val = dest.load();
        double new_val = old_val + val;

        while (! dest.compare_exchange_strong(old_val, new_val))
            new_val = old_val + val;
    }

    // check cuda call return values for proper completion
    void cuda_call(cudaError_t result)
    {
        if (result != cudaSuccess)
            error(std::string(cudaGetErrorName(result)) + ": " + cudaGetErrorString(result));
    }

    // initialize indexer array and indexer_idle queue
    void init_indexers (const config_persistent<float>& cpers)
    {
        if (gpus.empty())
            gpus.push_back(0);

        for (unsigned i=0; i<indexers_per_gpu; i++) { // round robin across cuda devices
            for (auto dev : gpus) {
                cuda_call(cudaSetDevice(dev));
                indexer.emplace_back(new indexer_t{cpers});
            }
        }

        if (indexer.empty())
            error("no indexers");

        indexer_idle.reset(indexer.size());
        for (int i=0; i<(int)indexer.size(); i++) // all indexers idle
            indexer_idle.push_front(i);
    }

    // initialize work items and progressible work item queue
    void init_work ()
    {
        work_queue.reset(files.size());
        for (int i=0; i<(int)files.size(); i++) {
            work_queue.push_front(static_cast<int>(witem_list.size())); // all work items are progressible
            witem_list.emplace_back(new work_item{files[i], i});
        }
    }

    // asynchronous indexer action callback
    void result_ready (void* data)
    {
        const work_item* work = (work_item*)data;
        //std::cout << "-> data: " << work->id << '\n';
        work_queue.push_back(work->id); // work item is now progressible again
    }

    // read data and return false if the file should be dropped
    // because there are to few spots
    template<std::size_t N>
    bool read_data (work_item* work, std::array<char, N>& buffer)
    {
        using logger::debug;
        using logger::info;
        using logger::stanza;

        std::ifstream ifs(work->filename);
        if (! ifs)
            throw std::invalid_argument(std::string{"unable to open file "} + work->filename);

        unsigned n = 0u;        // coordinate tripple number
        float* x = work->in.cell.x;
        float* y = work->in.cell.y;
        float* z = work->in.cell.z;

        for (unsigned line=1u; ifs.getline(buffer.data(), N); line++) {
            std::istringstream iss(buffer.data());
            while ((bool)(iss >> x[n])) {
                if (!(iss >> y[n] >> z[n]))
                    throw std::invalid_argument(std::string{"wrong file format for file "} + work->filename + ", line " + std::to_string(line));
                n++;
                if (n >= maxspot + 3u)
                    goto stop_reading;
            }
        }
    stop_reading:
        if (n < 4u) {
            LOG_START(logger::l_info) {
                info << stanza << "file " << work->filename << " dropped\n";
            } LOG_END;
            work->in.n_spots = 0u;
            return false;
        }
        work->in.n_spots = n - 3u;

        LOG_START(logger::l_debug) {
            debug << stanza << "input cell:\n";
            for (unsigned i=0u; i<3; i++)
                debug << stanza << work->in.cell.x[i] << " " << work->in.cell.y[i] << " " << work->in.cell.z[i] << '\n';
        } LOG_END;

        return true;
    }

    // worker thread
    void worker (const cfgrt_t& crt, const cifss_t& cifss, const cifse_t& cifse, const cifssr_t& cifssr, unsigned id)
    {
        try {
            double read_time_priv = .0;     // thread private accumulator for simple data reading time
            double indexer_time_priv = .0;  // thread private accumulator for indexing time
            double refine_time_priv = .0;   // thread private accumulator for refinement time

            std::array<char, 1024> buffer;  // buffer for file reading

            while (! pool_start.load());    // wait for start switch

            do {
                int witem_id = work_queue.pop_back();

                if (witem_id == CNIQ::free) {
                    std::this_thread::yield();  // no progressible work item
                } else {
                    std::unique_ptr<work_item>& work = witem_list[witem_id];

                    switch (work->state) {

                        case read_file: {   // read simple file data and accumulate reading time
                                // std::cout << id << ": " << witem_id << "-read_file " << work->filename << '\n';

                                auto t = clock::now();

                                bool ok = read_data(work.get(), buffer);

                                read_time_priv += duration{clock::now() - t}.count();

                                if (! ok) { // drop file
                                    work->cells.setZero();
                                    work->scores.setZero();
                                    counter++;
                                    break;
                                }

                                work->state = index_start;
                            }
                            // fall through
                        case index_start: { // launch indexer asynchronously, set start time
                                int idx = indexer_idle.pop_back();

                                if (idx == CNIQ::free) { // no idle indexer object, push work item back to the queue
                                    // std::cout << id << ": " << witem_id << "-enqueue\n";
                                    work_queue.push_front(witem_id);
                                    std::this_thread::yield();
                                } else {
                                    // std::cout << id << ": " << witem_id << "-index_start(" << idx << ") " << work->filename << '\n';
                                    work->tp = clock::now();    // indexing start time
                                    work->indexer = idx;        // associated indexer (currently idle)
                                    work->state = index_end;

                                    indexer[idx]->index_start(work->in, work->out, crt, result_ready, work.get()); // launch indexer asynchronously

                                    work->in.new_cells = false;
                                }
                            } break;

                        case index_end: {   // finish asynchronous indexing step, accumulate indexing time
                                // std::cout << id << ": " << witem_id << "-index_end(" << work->indexer << ") " << work->filename << '\n';

                                auto& ind = *indexer[work->indexer];
                                ind.index_end(work->out);
                                indexer_idle.push_back(work->indexer); // associated indexer object is now idle

                                auto t = clock::now();
                                indexer_time_priv += duration{t - work->tp}.count();

                                if (method == "raw") { // skip refine state by doing work here
                                    work->best_cell = refine::best_cell(work->scores);
                                    if (crystals)
                                        work->crystals = refine::select_crystals(work->cells, work->coords.bottomRows(work->in.n_spots), work->scores, maxdist, minpts, method=="ifssr");
                                    counter++;
                                    work->repetition++;
                                    if (work->repetition < repetitions) {
                                        work->state = index_start;
                                        work_queue.push_back(witem_id);
                                    } else {
                                        work->state = finished;
                                    }
                                    break;
                                }

                                work->state = refine_block;
                            }
                            // fall through
                        case refine_block: {    // refine one output cell block, accumulate refinement time

                                unsigned block = work->rblock; // output cell block to refine
                                // std::cout << id << ": " << witem_id << "-refine(" << block << '/' << refinement_blocks << ") " << work->filename << '\n';

                                work->rblock++;
                                if (work->rblock < refinement_blocks)
                                    work_queue.push_back(witem_id); // refine next block
                                else
                                    counter++; // no more blocks to refine, work for this item is finished

                                auto t = clock::now();

                                if (method == "ifss")
                                    indexer_ifss::refine(work->coords.bottomRows(work->in.n_spots), work->cells, work->scores, cifss, block, refinement_blocks);
                                else if (method == "ifse")
                                    indexer_ifse::refine(work->coords.bottomRows(work->in.n_spots), work->cells, work->scores, cifse, block, refinement_blocks);
                                else if (method == "ifssr")
                                    indexer_ifssr::refine(work->coords.bottomRows(work->in.n_spots), work->cells, work->scores, cifssr, block, refinement_blocks);
                                else
                                    throw std::runtime_error(method+": unexpected method in refine state");

                                refine_time_priv += duration{clock::now() - t}.count();

                                block = work->blkcnt.fetch_add(1u); // finished block
                                if (block + 1u >= refinement_blocks) {
                                    work->best_cell = refine::best_cell(work->scores);
                                    if (crystals)
                                        work->crystals = refine::select_crystals(work->cells, work->coords.bottomRows(work->in.n_spots), work->scores, maxdist, minpts, method=="ifssr");
                                    work->repetition++;
                                    if (work->repetition < repetitions) {
                                        // init work item
                                        work->rblock = 0u;
                                        work->blkcnt = 0u;
                                        work->state = index_start;
                                        work_queue.push_back(witem_id);
                                    } else {
                                        work->state = finished;
                                    }
                                }
                            } break;

                        default: {
                                std::cerr << 'T' << id << ": invalid state (" << to_string(work->state) << ") for work item " << witem_id << '\n';
                                std::exit(1);
                            }
                    }
                }

            } while (counter.load() < repetitions * files.size()); // while not all files indexed and refined

            atomic_add(read_time, read_time_priv);
            atomic_add(indexer_time, indexer_time_priv);
            atomic_add(refine_time, refine_time_priv);

        } catch (std::exception& ex) {
            std::cerr << "Error: " << ex.what() << '\n';
            std::exit(1);
        }
    }

    // initialize thread pool
    void init_pool (const cfgrt_t& crt, const cifss_t& cifss, const cifse_t& cifse, const cifssr_t& cifssr)
    {
        for (unsigned i=1u; i<worker_threads; i++) // don't put the main thread into the list
            thread_pool.push_back(std::thread(worker, crt, cifss, cifse, cifssr, i));
    }

    // join all threads in the thread pool (except main thread)
    void join_workers ()
    {
        for (auto& thread : thread_pool)
            thread.join();
    }

    // print cell geometry
    void print_cell_geometry(const Eigen::Matrix3<float>&  cell)
    {
        const auto a = cell.row(0);
        const auto b = cell.row(1);
        const auto c = cell.row(2);
        const auto la = a.norm();
        const auto lb = b.norm();
        const auto lc = c.norm();
        const auto r2d = 180. / M_PI;
        const auto ab = std::acos(a.dot(b) / (la * lb)) * r2d;
        const auto bc = std::acos(b.dot(c) / (lb * lc)) * r2d;
        const auto ca = std::   acos(c.dot(a) / (lc * la)) * r2d;
        std::cout << "geom: " << la << ' ' << lb << ' ' << lc << " / "
                              << ab << ' ' << ca << ' ' << bc << '\n';
    }

} // namespace

int main (int argc, char *argv[])
{
    using logger::debug;
    using logger::stanza;

    try {
        logger::init_log_level();

        cfgps_t cpers{};
        cfgrt_t crt{};
        cifss_t cifss{};
        cifse_t cifse{};
        cifssr_t cifssr{};
        mempin_t pin_crt{mempin_t::on(crt)};

        initargs(cpers, crt);
        argparse(argc, argv);
        checkargs();
        setconf(cpers, crt, cifss, cifse, cifssr);

        debug << stanza << "cpers: cells=" << cpers.max_output_cells << ", maxspots=" << cpers.max_spots
                        << ", cands=" << cpers.num_candidate_vectors << ", reducalc=" << cpers.redundant_computations << '\n'
              << stanza << "crt: hs-points=" << crt.num_halfsphere_points << ", a-points=" << crt.num_angle_points
                        << ", triml=" << crt.triml << ", trimh=" << crt.trimh << ", delta=" << crt.delta
                        << ", dist1=" << crt.dist1 << ", dist3=" << crt.dist3 << ", min_spots=" << crt.min_spots << '\n';
        if (method == "ifss") {
            debug << stanza << "cifss: contr=" << cifss.threshold_contraction << ", minpts=" << cifss.min_spots << ", iter=" << cifss.max_iter << '\n';
        } else if (method == "ifse") {
            debug << stanza << "cifse: contr=" << cifse.threshold_contraction << ", minpts=" << cifss.min_spots << ", iter=" << cifse.max_iter << '\n';
        } else if (method == "ifssr") {
            debug << stanza << "cifssr: contr=" << cifssr.threshold_contraction << ", minpts=" << cifssr.min_spots << ", iter=" << cifssr.max_iter << '\n';
        }

        init_indexers(cpers);
        init_work();
        init_pool(crt, cifss, cifse, cifssr);

        auto t0 = clock::now();

        pool_start.store(true);                 // activate start switch
        worker(crt, cifss, cifse, cifssr, 0);   // become part of the thread pool
        join_workers();

        auto t1 = clock::now();
        double elapsed_sec = duration{t1 - t0}.count();

        if (! quiet) {
            std::cout.precision(12);
            for (const auto& res : witem_list) {
                if (crystals) {
                    std::cout << res->filename << ": crystal ids";
                    if (res->crystals.empty()) {
                        std::cout << " - none\n";
                    } else {
                        for (unsigned i : res->crystals)
                            std::cout << ' ' << i;
                        std::cout << '\n';
                    }
                }
                if (best_only) {
                    std::cout << res->filename <<": cell score " << res->scores[res->best_cell] << '\n';
                    std::cout << res->cells.block(3u * res->best_cell, 0u, 3u, 3u) << '\n';
                    if (cell_geom)
                        print_cell_geometry(res->cells.block(3u * res->best_cell, 0u, 3u, 3u));
                    std::cout << '\n';
                } else {
                    std::cout << res->filename <<":\n";
                    for (unsigned j=0u; j<cpers.max_output_cells; j++) {
                        if (res->best_cell == j)
                            std::cout << "(best) ";                    
                        std::cout << "cell " << j << ": score " << res->scores[j] << '\n';
                        std::cout << res->cells.block(3u * j, 0u, 3u, 3u) << '\n';
                        if (cell_geom)
                            print_cell_geometry(res->cells.block(3u * res->best_cell, 0u, 3u, 3u));
                        std::cout << '\n';
                    }
                }
            }
        }

        if (! notiming) {
            std::cout << "per file average timings:\n";
            std::cout << "    clock time: " << (elapsed_sec / counter.load()) << "s\n";
            std::cout << "  reading time: " << (read_time / counter.load()) << "s\n";
            std::cout << "    index time: " << (indexer_time / counter.load()) << "s\n";
            std::cout << "   refine time: " << (refine_time / counter.load()) << "s\n";
        }
    } catch (std::exception& ex) {
        std::cerr << "indexing failed: " << ex.what() << '\n' << failure;
    }

    std::cout << success;
}
