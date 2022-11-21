## Test Code

Here's the place for ctest tests and other tests.

### Tests for ctest

   * **TEST_INDEXER_SIMPLE** Run two indexers in parallel on a simple data file
   * **TEST_INDEXER_EXCEPTION** Excercise *fast_feedback::exception* functionality
   * **TEST_INDEXER_OBJ** Check *fast_feedback::indexer* state handling
   * **TEST_SIMPLE_DATA_READER** Read a simple data file

### Other test code

   * **SIMPLE_DATA_INDEXER** Index simple data with the raw brute force sampling indexer
      * Arguments
         * *file name*: simple data file
         * *max number of spots*: only use the first spots up to this max
         * *max number of output cells*: never produce more than this max
         * *number of kept candidate vectors*: this is per vector length for sampling a first vector and per cell in the rotation step
         * *number of half sphere sample points*: $N$ - sample so many first vectors per length, and $1.5 \sqrt{N}$ cells around every such vector
      * Output
         * Indexing parameters
         * Input cell vectors
         * List of output cells
            * Cell score
            * Cell vectors
   * **REFINED_SIMPLE_DATA_INDEXER** same as *SIMPLE_DATA_INDEXER*, but adding a subsequent least squares refinement step on CPU using the Eigen library
      * Additional arguments
         * *lsq spot filter threshold*: cell will be fitted using least squares to spots approximated to this threshold
         * *score filter threshold*: count how many spots are approximated to this threshold after refinement
      * Output
         * Indexing parameters
         * Input cell
         * List of output cells
         * Per cell unified score list (number of spots $S$)
            * Original score / $-S$
            * Well approximated spots / $S$
         * Timings for
            * Preparation: creating the indexer object and allocating GPU memory
            * Indexing: brute force sampling indexer time
            * Refinement: least squares refinement of indexer output cells on CPU
            * Indexing + Refinement

Since these executables need the fast feedback indexer library, it has to be installed in a default library search location, or the *LD_LIBRARY_PATH* has to be set. To avoid that, the module RUNPATH elf entry can be set to the fast feedback indexer library installation location by switching on the *TESTS_RPATH* cmake option.