"""
The ffbidx python module allows to acces the Fast Feedback Indexer library functionality from Python.

To understand the workings of this library, look at the algorithm description at

https://github.com/paulscherrerinstitute/fast-feedback-indexer/blob/main/doc/algorithm/fast-feedback-indexer.tex

To get more information on this python module, look at

https://github.com/paulscherrerinstitute/fast-feedback-indexer/blob/main/python/README.md

This module exports the Indexer class of which several instances (the limit depends mainly on the GPU memory) can be created.
Every instance maintains state (allocates memory) on the GPU, which is relatively expensive. Instance creation and destruction
rates should therefore be kept at a minimum.
Instances may be used by several threads, but not at the same time (methods are not thread safe).

REFERENCES:

Please reference the relevant papers in scientific work done with this software. You may find a BibTeX file at
https://github.com/paulscherrerinstitute/fast-feedback-indexer/blob/main/BIBTeX.bib

BUGS:

Let us know of any bugs you find at
https://github.com/paulscherrerinstitute/fast-feedback-indexer/issues

IMPROVEMENTS:

Let us know how we can improve this software or just give your opinion at 
https://github.com/paulscherrerinstitute/fast-feedback-indexer/discussions

LICENCE:

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
"""

from ffbidx.ffbidx_impl import indexer, index, crystals, release

class Indexer:
    def __init__(self,
            max_output_cells:int=32,
            max_input_cells:int=1,
            max_spots:int=300,
            num_candidate_vectors:int=32,
            redundant_computations:bool=True):
        """Initialize fast feedback indexer object.

        This initializer allocates state on the GPU.
        The current cuda device is used, unless the
        environment variable INDEXER_GPU_DEVICE is set,
        then the device specified as its value is used.

        Parameters
        ----------
        max_output_cells : int
            Maximum number of potential unit cell candidates calculated.
        max_input_cells : int
            Maximum number of input unit cells.
        max_spots : int
            Maximum number of spots considered. The spot array will be
            truncated to this size if it has more spots.
        num_candidate_vectors : int
            Calculate this number of candidate vectors in the initial
            vector sampling stage.
        redundant_computations : bool
            Calculate vector candidates for all three input unit cell
            vectors, instead of for the biggest one only.
        
        Returns
        -------
        An indexer object with state allocated on the GPU.

        Raises
        ------
        ValueError
            If a parameter value is not acceptable.
        RuntimeError
            If the indexer object cannot be created.
        """
        self.initialized = False
        self.handle = indexer(max_output_cells=max_output_cells,
            max_input_cells=max_input_cells,
            max_spots=max_spots,
            num_candidate_vectors=num_candidate_vectors,
            redundant_computations=redundant_computations)
        self.initialized = True
    
    def run(self,
            spots, input_cells,
            method:str="ifssr",
            length_threshold:float=1e-9,
            triml:float=.001,
            trimh:float=.3,
            delta:float=.1,
            dist1:float=.1,
            dist3:float=.15,
            num_halfsphere_points:int=32*1024,
            num_angle_points:int=0,
            n_output_cells:int=32,
            contraction:float=.8,
            max_dist:float=.00075,
            min_spots:int=8,
            n_iter:int=32):
        """Run indexing operation.

        Parameters
        ----------
        spots : Numpy array
            The input spots in reciprocal space, will be truncated to max_spots spots.
            Must be C_CONTIGUOUS with X coordinates as rows, or F_CONTIGUOUS with
            X coordinates as columns.
        input_cells : Numpy array
            The input unit cells, will be truncated to max_input_cells unit cells.
            Unit cells are given in sets of three consecutive vectors.
            Must be C_CONTIGUOUS with X coordinates as rows, or F_CONTIGUOUS with
            X coordinates as columns.
        method : str
            Refinement method:
            - "raw" : No refinement.
            - "ifss" : Iterative fit to selected spots in reciprocal space.
            - "ifse" : Iterative fit to selected errors in reciprocal space.
            - "ifssr" : Iterative fit to selected spots in primal space.
        length_threshold : float
            Consider two length' the same if they differ less than this value.
        triml : float
            Lower trim value for log of error.
        trimh : float
            Higher trim value for log of error.
        delta : float
            Log error curve shift value.
        dist1 : float
            Count spots approximated more closely than this value in the vector
            sampling stage.
        dist3 : float
            Count spots approximated more closely than this value in the cell
            sampling stage.
        num_halfsphere_points : int
            Number of sampling points on the half sphere for vector sampling.
        num_angle_points : int
            Number of sampling points on the circle for cell sampling.
            If zero, set it as to achieve approximately the same density
            as for vector sampling.
        n_output_cells : int
            Limit the number of unit cell candidates to less than the max.
        contraction : float
            Error threshold contraction factor for selecting close spots
            in refinement.
        max_dist : float
            Stop refinement if the error threshold is below this value.
        min_spots : int
            Stop refinement if less than this number of spots are selected.
        n_iter : int
            Stop refinement if this iteration count has been reached.
        
        Returns
        -------
        output_cells, scores : tuple of Numpy arrays
            The unit cell candidates in C_CONTIGUOUS order with X coordinates
            as rows, and the score values for each unit cell. The score is the
            distance to the worst of the best approximated min_spots spots.
        
        Raises
        ------
        ValueError 
            If a parameter is unacceptable.
        RuntimeError
            If the indexing operation cannot be run.
        """
        return index(self.handle,
            spots=spots,
            input_cells=input_cells,
            method=method,
            length_threshold=length_threshold,
            triml=triml,
            trimh=trimh,
            delta=delta,
            dist1=dist1,
            dist3=dist3,
            num_halfsphere_points=num_halfsphere_points,
            num_angle_points=num_angle_points,
            n_output_cells=n_output_cells,
            contraction=contraction,
            max_dist=max_dist,
            min_spots=min_spots,
            n_iter=n_iter)

    def crystals(self,
            cells, spots, scores,
            method="ifssr",
            threshold=.00075,
            min_spots=8):
        """Select crystals.

        Parameters
        ----------
        cells : Numpy array
            Cell array from index().
        spots : Numpy array
            Spots array from index().
        scores : Numpy array
            Scores array from index().
        method : str
            Measure distance as for this refinement method:
            - "raw" : to integer coordinate
            - "ifss" : to integer coordinate
            - "ifse" : to integer coordinate
            - "ifssr" : induced versus measured spot
        threshold : float
            Distance threshold for considering spots covered.
        min_spots : int
            Minimum number of new spots covered.

        Returns
        -------
        Numpy array with indices of cells considered new crystals,
        or None if there aren't any crystals.

        Raises
        ------
        ValueError
            If parameters are unacceptable.
        RuntimeError
            If the crystals cannot be selected.
        """
        return crystals(cells, spots, scores, method, threshold, min_spots)

    def __del__(self):
        """Release the state allocated on GPU."""
        if (self.initialized):
            release(self.handle)
