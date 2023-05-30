## Simplistic Python Module

The idea is to provide simple access to the fast feedback indexer via python, mainly as a means to make more people evaluate the fast feedback indexer and find deficiencies to determine future coding action.

The python module is built and installed into a specific path by setting the *PYTHON_MODULE* option and the *PYTHON_MODULE_PATH* path variable for cmake.

Since the python module uses the fast feedback indexer library, it must be in a default library search location, in the same directory as the module itself, or the *LD_LIBRARY_PATH* has to be set. To avoid that the module RUNPATH elf entry can be set to the fast feedback indexer library installation location by switching on the *PYTHON_MODULE_RPATH* cmake option. RUNPATH will be set to a relative path, unless the *INSTALL_RELOCATABLE* cmake option is switched off to make RUNPATH an absolute path.

The python module also uses libpython. Only if the *PYTHON_MODULE_RPATH* cmake option is switched on, the module RUNPATH elf entry will be set to the absolute directory of the libpython that was used when building the python module.

### Interface

The module has a primitive non pythonic interface, sorry for that.

#### import ffbidx

Imports the module.

#### ffbidx.indexer(max_output_cells, max_input_cells, max_spots, num_candidate_vectors, redundant_calculations)

**Return**:

Handle to the indexer object

**Arguments**:

- **max_output_cells** is the maximum number of output cells generated
- **max_input_cells** is the maximum number of input cells considered
- **max_spots** is the maximum number of spots considered
- **num_candidate_vectors** is the number of candidates (best sample vectors of a specific length) computed per length
- **redundant_calculations** makes the code compute candidates for all vectors per input cell instead of just one

This allocates space on the GPU for all the data structures used in the computation. The GPU device is parsed from the *INDEXER_GPU_DEVICE* environment variable. If it is not set, the current GPU device is used.

#### ffbidx.index(handle, spots, input_cells, method='ifss', length_threshold=1e-9, triml=.05, trimh=.15, delta=0.1, num_sample_points=32*1024, n_output_cells=1, contraction=.8, min_spots=6, n_iter=15)

Run the fast feedback indexer on given 3D real space input cells and reciprocal spots packed in the **input_cells** and **spots** numpy array and return oriented cells and their scores. The still experimental *'raw'* method first finds candidate vectors according to the score $\sum_{s \in spots} \log_2(trim_l^h(dist(s, clp)) + delta))$, which are then used as rotation axes for the input cell. The cell score for the *'raw'* method is
$-| \\{ s \in spots: dist(s, clp) < h \\} | + 2^{\frac{\sum_{s \in spots} \log_2(trim_l^h(dist(s, clp)) + delta))}{|spots|}} - delta$, where $trim$ stands for trimming, $dist(s, clp)$ for the distance of a spot to the closest lattice point, and $l,h$ are the lower and higher trimming thresholds.

**Return**:

A tuple of numpy arrays *(output_cells, scores)*

- **output_cells** is an array with *N* computed cells in 3D space with shape *(3, 3N), order='C'*. The first cell is *\[:,:3\]* and it's first vector is *\[:,0\]*.
- **scores** is a one dimensional numpy array of shape *(N,)* containing the score (objective function value) for each output cell.

**Arguments**:

- **handle** is the indexer object handle
- **spots** is a numpy array of spot coordinates in reciprocal space. In memory, all x coordinates followed by all y coordinates and finally all z coordinates. If there are *K* spots, the array shape is either *(3, K), order='C'*, or *(K, 3), order='F'*.
- **input_cells** is a numpy array of input cell vector coordinates in real space. In memory, all x coordinates followed by all y coordinates and finally all z coordinates in consecutive packs of 3 coordinates. If there are *M* input cells, the array shape is either *(3, 3M), order='C'*, or *(3M, 3), order='F'*.
- **method** refinement method: one of *'raw'* (no refinement), *'ifss'* (iterative fit to selected spots), *'ifse'* (iterative fit to selected errors)
- **length_threshold**: consider input cell vector length the same if they differ by less than this
- **triml**: >= 0, low trim value, 0 means no trimming
- **trimh**: <= 0.5, high trim value, 0.5 means no trimming
- **delta**: > 0 - triml, $\log_2$ curve position, lower values will be more selective in choosing close spots
- **num_sample_points** is the number of sampling points per sample vector length on the half sphere
- **n_output_cells** is the number of desired output cells
- **contraction** threshold contraction parameter for methods *'ifss'* and *'ifse'*
- **min_spots** minimum number of spots to fit against for methods *'ifss'* and *'ifse'*
- **n_iter** maximum number of iterations for methods *'ifss'* and *'ifse'*

**Refinement Methods**:

After running the *'raw'* method, there's the possibility to refine the cells using two experimental methods currently.
Both methods use the normalized sum of logarithms part from the *'raw'* cell score as the initial threshold $t$.

*'ifss'*: Iteratively fit a new cell to the spots $\\{ s \in spots: dist(s, clp) < t \\}$ and contract the threshold. Stop when the maximum number of iterations is reached, or the spot set size is below the minimum number of spots.

*'ifse'*: Iteratively fit an additive delta to the errors $\\{ dist(s, clp) : s \in spots \land dist(s, clp) < t \\}$ and contract the threshold. Stop when the maximum number of iterations is reached, or the errors set size is below the minimum number of spots.

#### ffbidx.release(handle)

Release the indexer object and associated GPU memory. The handle must not be used after this.

**Arguments**:

- **handle** is the indexer object handle

### Issues

   * The module sets the logging level just once on loading, so the *INDEXER_LOG_LEVEL* environment variable has to be set before the import statement.
   * The handles are taken from an increasing 32 bit counter that wraps around

### Note

To avoid a `make install`, there's a script `pythonlib.sh` at the top level for installing the python module after building it.
