## Simple Python Module

The idea is to provide simple access to the fast feedback indexer via python, mainly as a means to make more people evaluate the fast feedback indexer and find deficiencies to determine future coding action.

The python module is built and installed into a specific path by setting the *PYTHON_MODULE* option and the *PYTHON_MODULE_PATH* path variable for cmake.

Since the python module uses the fast feedback indexer library, it must be in a default library search location, in the same directory as the module itself, or the *LD_LIBRARY_PATH* has to be set. The module RUNPATH elf entry can be set to the fast feedback indexer library installation location by switching on the *PYTHON_MODULE_RPATH* cmake option. RUNPATH will be set to a relative path, unless the *INSTALL_RELOCATABLE* cmake option is switched off to make RUNPATH an absolute path.

The python module also uses libpython. Only if the *PYTHON_MODULE_RPATH* cmake option is switched on, the module RUNPATH elf entry will be set to the absolute directory of the libpython that was used when building the python module.

### Interface

The module provides an Indexer object used to keep state on the GPU and run indexing operations.
Only basic indexing and refinement methods are implemented so far.

#### import ffbidx

Imports the module.

#### indexer = ffbidx.Indexer(max_output_cells=32, max_input_cells=1, max_spots=300, num_candidate_vectors=32, redundant_calculations=True)

Create indexer object and including state allocated on the GPU.

**Return**:

Indexer object

**Arguments**:

- **max_output_cells** is the maximum number of output cells generated
- **max_input_cells** is the maximum number of input cells considered
- **max_spots** is the maximum number of spots considered
- **num_candidate_vectors** is the number of candidates (best sample vectors of a specific length) computed per length
- **redundant_calculations** makes the code compute candidates for all vectors per input cell instead of just one

This allocates space on the GPU for all the data structures used in the computation. The GPU device is parsed from the *INDEXER_GPU_DEVICE* environment variable. If it is not set, the current GPU device is used.

#### output_cells, output_scores = indexer.run(spots, input_cells, method='ifssr', length_threshold=1e-9, triml=.001, trimh=.3, delta=0.1, dist1=.0, dist3=.15, num_halfsphere_points=32*1024, num_angle_points=0, n_output_cells=32, contraction=.8, max_dist=.00075, min_spots=8, n_iter=32)

Run the fast feedback indexer on given 3D real space input cells and reciprocal spots packed in the **input_cells** and **spots** numpy array and return oriented cells and their scores. The still experimental *'raw'* method first finds candidate vectors according to the score $\sqrt[|spots|]{\prod_{s \in spots} trim_l^h(dist(s, clp)) + delta} - delta - c - 1$, which are then used as rotation axes for the input cell. The cell score for the *'raw'* method is
the same. Here, $trim$ stands for trimming, $dist(s, clp)$ for the distance of a spot to the closest lattice point, $l,h$ are the lower and higher trimming thresholds, and $c$ is the number of close spots contributing to the score.

**Return**:

A tuple of numpy arrays *(output_cells, scores)*

- **output_cells** is an array with *N* computed cells in 3D space with shape *(3, 3N), order='C'*. The first cell is *\[:,:3\]* and it's first vector is *\[:,0\]*.
- **scores** is a one dimensional numpy array of shape *(N,)* containing the score (objective function value) for each output cell.

**Arguments**:

- **spots** is a numpy array of spot coordinates in reciprocal space. In memory, all x coordinates followed by all y coordinates and finally all z coordinates. If there are *K* spots, the array shape is either *(3, K), order='C'*, or *(K, 3), order='F'*.
- **input_cells** is a numpy array of input cell vector coordinates in real space. In memory, all x coordinates followed by all y coordinates and finally all z coordinates in consecutive packs of 3 coordinates. If there are *M* input cells, the array shape is either *(3, 3M), order='C'*, or *(3M, 3), order='F'*.
- **method** refinement method: one of *'raw'* (no refinement), *'ifss'* (iterative fit to selected spots), *'ifse'* (iterative fit to selected errors), *'ifssr'* (iterative fit to selected spots reciprocal).
- **length_threshold**: consider input cell vector length the same if they differ by less than this
- **triml**: >= 0, low trim value, 0 means no trimming
- **trimh**: <= 0.5, high trim value, 0.5 means no trimming
- **delta**: > 0 - triml, $\log_2$ curve position, lower values will be more selective in choosing close spots
- **dist1**: spots within this distance are contributing to the score in the vector sampling step (set to *trimh* if <=0)
- **dist3**: spots within this distance are contributing to the score in the cell sampling step (set to *trimh* if <=0)
- **num_halfsphere_points** is the number of sampling points per sample vector length on the half sphere for vector sampling
- **num_angle_points** is the number of angular sampling points for cell rotation sampling (0 for heuristic)
- **n_output_cells** is the number of desired output cells
- **contraction** threshold contraction parameter for methods *'ifss'* and *'ifse'*
- **max_dist** maximum distance parameter for methods *'ifss'* and *'ifse'*
- **min_spots** minimum number of spots to fit against for methods *'ifss'* and *'ifse'*
- **n_iter** maximum number of iterations for methods *'ifss'* and *'ifse'*

**Refinement Methods**:

After running the *'raw'* method, there's the possibility to refine the cells using two experimental methods currently.
Both methods use the normalized sum of logarithms part from the *'raw'* cell score as the initial threshold $t$.

*'ifss'*: Iteratively fit a new cell to the spots $\\{ s \in spots: dist(s, clp) < t \\}$ and contract the threshold. Stop when the maximum number of iterations is reached, or the maximum distance has been reached, or the spot set size is below the minimum number of spots.

*'ifse'*: Iteratively fit an additive delta to the errors $\\{ dist(s, clp) : s \in spots \land dist(s, clp) < t \\}$ and contract the threshold. Stop when the maximum number of iterations is reached, or the maximum distance has been reached, or the errors set size is below the minimum number of spots.

*'ifssr'*: Iteratively fit a new cell to the spots $\\{ s \in spots: ||s, is|| < t \\}$, where *'is'* is the induced spot and contract the threshold. Stop when the maximum number of iterations is reached, or the maximum distance has been reached, or the spot set size is below the minimum number of spots.

#### cell_indices = indexer.crystals(output_cells, spots, output_scores, method='ifssr', threshold=.00075, min_spots=8)

Calculate crystals contained in output candidate cells.

**Return**:

Numpy array with indices of the cells representing separate crystals, or *None* for no crystals found. If *i* is returned in *cell_indices*, *output_cells*[3 * *i*: 3 * *i* + 3] (for *output_cells* in order='C') represents a separate crystal.

**Arguments**:

- **output_cells**: is a numpy array of candidate cell vector coordinates in real space. In memory, all x coordinates followed by all y coordinates and finally all z coordinates in consecutive packs of 3 coordinates. If there are *M* candidate cells, the array shape is either *(3, 3M), order='C'*, or *(3M, 3), order='F'*.
- **spots**: is a numpy array of spot coordinates in reciprocal space. In memory, all x coordinates followed by all y coordinates and finally all z coordinates. If there are *K* spots, the array shape is either *(3, K), order='C'*, or *(K, 3), order='F'*.
- **output_scores**: is a numpy array with the scores for the output cells.
- **method**: how to measure the distance, one of *'raw'* (no refinement), *'ifss'* (iterative fit to selected spots), *'ifse'* (iterative fit to selected errors). For *'ifssr'* the distance between measured and induced spot is compaed to the threshold, otherwise the distance to integer coordinates is compared to threshold.
- **threshold**: Distance threshold for considering a spot covered.
- **min_spots**: Minimal number of extra covered spots for a separate crystal.

#### del indexer

Release the indexer object and associated GPU memory.

### Issues

   * The module sets the logging level just once on loading, so the *INDEXER_LOG_LEVEL* environment variable has to be set before the import statement.
   * Internally used handles taken from an increasing 32 bit counter may wrap around.
