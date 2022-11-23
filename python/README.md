## Simplistic Python Module

The idea is to provide simple access to the fast feedback indexer via python, mainly as a means to make more people evaluate the fast feedback indexer and find deficiencies to determine future coding action.

The python module is built and installed into a specific path by setting the *PYTHON_MODULE* option and the *PYTHON_MODULE_PATH* path variable for cmake.

Since the python module uses the fast feedback indexer library, it must be in a default library search location, in the same directory as the module itself, or the *LD_LIBRARY_PATH* has to be set. To avoid that the module RUNPATH elf entry can be set to the fast feedback indexer library installation location by switching on the *PYTHON_MODULE_RPATH* cmake option. RUNPATH will be set to a relative path, unless the *INSTALL_RELOCATABLE* cmake option is switched off to make RUNPATH an absolute path.

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

This allocates space on the GPU for all the data structures used in the computation. The GPU device is parsed from the *INDEXER_GPU_DEVICE* environment variable. If it is not set, the current GPU device is used.

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

Release the indexer object and associated GPU memory. The handle must not be used after this.

**Arguments**:

- **handle** is the indexer object handle

### Issues

   * The module sets the logging level just once on loading, so the *INDEXER_LOG_LEVEL* environment variable has to be set before the import statement.
   * The handles are taken from an increasing 32 bit counter that wraps around

### Note

To avoid a `make install`, there's a script `pythonlib.sh` at the top level for installing the python module after building it.
