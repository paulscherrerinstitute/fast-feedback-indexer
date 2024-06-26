#ifndef FFBIDX_C_API_H
#define FFBIDX_C_API_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float float_type;

// When first used, the C API pinns the cell and spot coordinates
// using 3*n_cells and n_spots as array sizes.
// The user is responsible to keep the pinned memory area valid
// during use!
// The idea is to preallocate coordinate data space and then
// keep eorking inside this area.
// It's possible to change coordinate pointers before indexing starts.
// This is detected by (!!) pointer comparison (!!). The old pointer is then
// unpinned, and the area of a size given with 3*n_cells or n_spots
// of memory under the corresponding pointer is then pinned.
// Pinning takes place in indexer_start() or indexer_op().
struct input {
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

// See comments related to pinning for input.
// The coordinate pointers here are pinned with a size of 3*n_cells,
// the score pointer with a size of n_cells.
struct output {
    float_type* x;      // x coordinates, pinned memory
    float_type* y;      // y coordinates, pinned memory
    float_type* z;      // z coordinates, pinned memory
    float_type* score;  // per cell score, pinned memory
    unsigned n_cells;   // number of unit cells
};

// See comments related to pinning for input.
// This structure is pinned with a size of sizeof(config_runtime)
struct config_runtime {             // pinned memory
    float_type length_threshold;    // threshold for determining equal vector length (|va| - threshold < |vb| < |va| + threshold)
    float_type triml;               // lower trim value for distance to nearest integer objective value - 0 < triml < trimh
    float_type trimh;               // higher trim value for distance to nearest integer objective value - triml < trimh < 0.5
    float_type delta;               // log2 curve position: score = log2(trim(dist(x)) + delta)
    float_type dist1;               // maximum distance to int for single coordinate
    float_type dist3;               // maximum distance to int for tripple coordinates
    unsigned num_halfsphere_points; // number of sample points on half sphere for finding vector candidates
    unsigned num_angle_points;      // number of sample points in rotation space for finding cell candidates (0: auto)
};

// Persistent configuration leading to allocation of GPU memory
// Values cannot be changed, instead drop the indexer handle and create a new one
struct config_persistent {
    unsigned max_output_cells;      // maximum number of output unit cells
    unsigned max_input_cells;       // maximum number of input unit cells, (must be before max_spots in memory, see copy_in())
    unsigned max_spots;             // maximum number of input spots, (must be after max_input_cells in memory, see copy_in())
    unsigned num_candidate_vectors; // number of candidate vectors (per input cell vector)
    bool redundant_computations;    // compute candidates for all three cell vectors instead of just one
};

struct config_ifssr {
    float_type threshold_contraction; // contract error threshold by this value in every iteration
    float_type max_distance;        // max distance to reciprocal spots for inliers
    unsigned min_spots;             // minimum number of spots to fit against
    unsigned max_iter;              // max number of iterations
};

// This struct, including the message needs to be preallocated by the user.
// The user is responsible for keeping it alive while used and to deallocate it.
// Such an error struct is stored per indexer handle. The stored pointer can be
// retrieved with error_message()
// If a handle is valid, functions taking an indexer handle may fill in the error
// message up to msg_len bytes and return -1.
// If a handle is invalid, no error message is given and functions just return -1 or NULL
struct error {
    char* message;      // initialized pointer to a char array of length at least msg_len
    unsigned msg_len;   // length of char array pointed to by message
};

// Callback will be called with data given to indexer_start()
typedef void(*callback_func)(void*);

// Fill in these structures with default values
// Pointer arguments can be NULL
void set_defaults(struct config_persistent* cfg_persistent,
                  struct config_runtime* cfg_runtime,
                  struct config_ifssr* cfg_ifssr);

// Create a handle for an indexer using cfg_persistent.
// Store error struct for the handle returned.
// Store data pointer for the handle returned. The data pointer can
// be retrieved with user_data()
// The err and data pointers can be NULL.
// Return:
// - handle (>=0) on success
// - -1 on failure (tries to fill in err->message)
int create_indexer(const struct config_persistent* cfg_persistent,
                   struct error* err,   // error message will be put here
                   void* data);         // arbitrary user data

// Drop indexer, cleaning up
// This doesn't free the data and error struct stored for the handle
// Return:
// - 0 on success
// - -1 on error (tries to fill in error message)
int drop_indexer(int handle);           // destroy indexer object, but neither error pointer nor data pointer

// Retrieve error struct stored when creating the handle
// Return:
// - pointer on success
// - NULL on error (invalid handle) or if pointer is NULL
struct error* error_message(int handle); // error message for handle

// Retrieve user data stored when creating the handle
// Return:
// - pointer on success
// - NULL on error (invalid handle) or if pointer is NULL
void* user_data(int handle);            // user data for handle

// Start GPU part of indexing asynchronously
// This will pin data (see comments for input, output, and cfg_runtime) (!!)
// If callback is not NULL, it will be called with data as argument as soon as the partial
// indexing result is ready on the GPU, and index_end() must not be called before that (!!).
// If callback is NULL, index_end() will block until partial results on GPU are ready.
// Return:
// - 0 on success
// - -1 on error (tries to fill in error message)
int index_start(int handle,
                const struct input* in,
                struct output* out,
                const struct config_runtime* cfg_runtime,
                callback_func callback, // if not NULL, don't call index_end before this function is called
                void* data);            // data for callback function

// Finish GPU part of indexing.
// Return:
// - 0 on success
// - -1 on failure (tries to fill in error message)
int index_end(int handle,
              struct output* out);

// Refine output cells on CPU using the iterative fit to selected
// spots reciprocal method.
// It's possible to call this from multiple threads on multually exclusive
// cell blocks (all parallel calls must give the same nblocks value, but
// different block values). All cells are handled at once by passing
// block=0 and nblocks=1
// This will modify output cells and scores.
// If refinement is done, a cell viability check for cell i can be done like this:
// cell_ok = (out->score[i] < cfg_ifssr->max_distance)
// meaning that at least cfg_ifssr->min_spots induced spots are within distance
// cfg_ifssr->max_distance of the measured spots given by in->spots
// Return:
// - 0 on success
// - -1 on error (tries to fill in error message)
int refine(int handle,
           const struct input* in,
           struct output* out,
           const struct config_ifssr* cfg_ifssr, // noop if NULL
           unsigned block, unsigned nblocks);   // handle one block out of nblocks output cells, threadsafe
                                                // (set to 0,1 for all cells)

// Combined index_start(), index_end(), refine() and best_cell()
// This will pin data (see comments for input, output, and cfg_runtime) (!!)
// Check comments for refine() about cell viability checking.
// Return:
// - index (>=0) of cell with best score
// - -1 on failure (tries  to fill in error message)
int indexer_op(int handle,   // just do it all
               const struct input* in,
               struct output* out,
               const struct config_runtime* cfg_runtime,
               const struct config_ifssr* cfg_ifssr);   // no refinement if NULL

// Find index of cell with best score
// Return:
// - index (>=0) of cell with best score
// - -1 on failure (tries to fill in error message)
int best_cell(int handle,
              const struct output* out);

// Find indices of cells belonging to different crystals
// The indices are filled in and the number of crystals returned.
// Return:
// - number of crystals found (>=0)
// - -1 on failure (tries to fill in error message)
int crystals(int handle,
             const struct input* in,
             const struct output* out,
             float_type threshold,      // distance threshold for matched spots
             unsigned min_spots,        // minimum number of new matches for a new crystal
             unsigned* indices,         // preallocated array
             unsigned indices_size);    // size of preallocated array

// Foreign convenience wrappers, use cuda_runtime.h directly if possible
int num_gpus();
int select_gpu(int gpu);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FFBIDX_C_API_H
