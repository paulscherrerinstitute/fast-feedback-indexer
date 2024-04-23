#ifndef FFBIDX_C_API_H
#define FFBIDX_C_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float float_type;

struct input final {
    struct {
        float_type* x;  // x coordinates, pinned memory
        float_type* y;  // y coordinates, pinned memory
        float_type* z;  // z coordinates, pinned memory
    } cell;
    struct {
        float_type* x;  // x coordinates, pinned memory
        float_type* y;  // y coordinates, pinned memory
        float_type* z;  // z coordinates, pinned memory
    } spot;
    unsigned n_cells;   // number of given unit cells (must be before n_spots in memory, see copy_in())
    unsigned n_spots;   // number of spots (must be after n_cells in memory, see copy_in())
    bool new_cells;     // set to true if cells are new or have changed
    bool new_spots;     // set to true if spots are new or have changed
};

struct output final {
    float_type* x;      // x coordinates, pinned memory
    float_type* y;      // y coordinates, pinned memory
    float_type* z;      // z coordinates, pinned memory
    float_type* score;  // per cell score, pinned memory
    unsigned n_cells=1u;// number of unit cells
};

struct config_runtime {
    float_type length_threshold=1e-9;       // threshold for determining equal vector length (|va| - threshold < |vb| < |va| + threshold)
    float_type triml=0.001;                 // lower trim value for distance to nearest integer objective value - 0 < triml < trimh
    float_type trimh=0.3;                   // higher trim value for distance to nearest integer objective value - triml < trimh < 0.5
    float_type delta=0.1;                   // log2 curve position: score = log2(trim(dist(x)) + delta)
    float_type dist1=0.2;                   // maximum distance to int for single coordinate
    float_type dist3=0.15;                  // maximum distance to int for tripple coordinates
    unsigned num_halfsphere_points=32*1024; // number of sample points on half sphere for finding vector candidates
    unsigned num_angle_points=0;            // number of sample points in rotation space for finding cell candidates (0: auto)
};

struct config_persistent {
    unsigned max_output_cells=1;            // maximum number of output unit cells
    unsigned max_input_cells=1;             // maximum number of input unit cells, (must be before max_spots in memory, see copy_in())
    unsigned max_spots=200;                 // maximum number of input spots, (must be after max_input_cells in memory, see copy_in())
    unsigned num_candidate_vectors=32;      // number of candidate vectors (per input cell vector)
    bool redundant_computations=false;      // compute candidates for all three cell vectors instead of just one
};

struct config_ifss {
    float_type threshold_contraction=.8;    // contract error threshold by this value in every iteration
    float_type max_distance=.01;            // max distance to integer for inliers
    unsigned min_spots=6;                   // minimum number of spots to fit against
    unsigned max_iter=15;                   // max number of iterations
};

struct config_ifssr {
    float_type threshold_contraction=.8;    // contract error threshold by this value in every iteration
    float_type max_distance=.00075;         // max distance to reciprocal spots for inliers
    unsigned min_spots=8;                   // minimum number of spots to fit against
    unsigned max_iter=32;                   // max number of iterations
};

typedef enum {
    raw = 0,    // no cell refinement
    ifss = 1,   // iterative fit to selected spots, distance measured in coordinate space
    ifssr = 2   // iterative fit to selected spots, distance measured in reciprocal space
} refinement_type;

struct error {
    char* message = NULL;   // initialized pointer to a char array of length at least msg_len
    unsigned msg_len;       // length of char array pointed to by message
};

typedef void(*callback_func)(void*);

// return -1 (or NULL) on error, 0 (or pointer) on success
int create_indexer(const config_persistent* cfg_persistent,
                   error* err,          // error message will be put here
                   void* data);         // arbitrary user data
int drop_indexer(int handle);           // destroy indexer object, but neither error pointer nor data pointer
error* error_message(int handle);       // error message for handle
void* user_data(int handle);            // user data for handle
int index_start(int handle,
                const input* in,
                output* out,
                const config_runtime* cfg_runtime,
                callback_func callback, // if not NULL, don't call index_end before this function is called
                void* data);            // data for callback function
int index_end(int handle,
              output* out);
int refine(int handle,
           const input* in,
           output* out,
           refinement_type refinement,
           const void* cfg_refinement,
           unsigned block, unsigned nblocks);   // handle one block out of nblocks output cells, threadsafe
                                                // (set to 0,1 for all cells)
int index(int handle,   // just do it all
          const input* in,
          output* out,
          const config_runtime* cfg_runtime,
          refinement_type refinement,
          const void* cfg_refinement);
int best_cell(int handle,
              const output* out);
int crystals(int handle,
             const input* in,
             const output* out,
             float_type threshold,
             unsigned min_spots,
             unsigned* indices,
             unsigned indices_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FFBIDX_C_API_H
