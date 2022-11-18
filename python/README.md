## Simplistic Python Module

The idea is to provide simple access to the fast feedback indexer via python, mainly as a means to make more people evaluate the fast feedback indexer and find deficiencies to determine future coding action.

The python module is built and installed into a specific path by setting the *PYTHON_MODULE* option and the *PYTHON_MODULE_PATH* path variable for cmake.

### Interface

The module has a primitive non pythonic interface, sorry for that.

#### import ffbidx

Imports the module.

#### ffbidx.indexer(max_output_cells, max_input_cells, max_spots, num_candidate_vectors)

**Return**:

Handle to the indexer object

**Arguments**:

- **max_output_cells** is the maximum number of output cells generated
- **max_input_cells** is the maximum number of input cells considered
- **max_spots** is the maximum number of spots considered
- **num_candidate_vectors** is number of candidates (best sample vectors of a specific length) computed per length

This allocates space on the GPU for all the data structures used in the computation.

#### ffbidx.index(handle, length_threshold, num_sample_points, n_output_cells, n_input_cells, data)

Run the fast feedback indexer on given reciprocal space input cells and spots packed in the **data** numpy array and return oriented cells and their scores.

**Return**:

A tuple of numpy arrays *(output_cells, scores)*

- **output_cells** is an array with *N* computed cells in reciprocal space and shape *(3, 3N)*. The first cell is *\[:,:3\]* and it's first vector is *\[:,0\]*.
- **scores** is a one dimensional numpy array of shape *(N,)* containing the score (objective function value) for each output cell.

**Arguments**:

- **handle** is the indexer object handle
- **length_threshold**: consider input cell vector length the same if they differ by less than this
- **num_sample_points** is the number of sampling points per length on the half sphere
- **n_output_cells** is the number of desired output cells
- **n_input_cells** is the number of given unit cells *N* in the data array
- **data** array of vectors with shape *(3,3N+S)*, the first *3N* vectors are the given unit cells, the rest are spots (all in reciprocal space)

#### ffbidx.release(handle)

Release the indexer object. The handle must not be used after this.

**Arguments**:

- **handle** is the indexer object handle

### Issues

   * If the indexer is not realeased properly, ugly memory deallocation errors appear.
   * The module sets the logging level just once on loading, so the *INDEXER_LOG_LEVEL* environment variable has to be set before the import statement.
   * The GPU device is determined once when creating the first indexer handle, so the *INDEXER_GPU_DEVICE* environment variable has to be set before that.

### Note

To avoid a `make install`, there's a script `pythonlib.sh` at the top level for installing the python module after building it. `LD_LIBRARY_PATH` still has to be set, but to the build artefact (`<clone dir>/build/indexer/src` if you more or less followed the build instructions).
