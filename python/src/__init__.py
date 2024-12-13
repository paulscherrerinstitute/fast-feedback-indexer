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

from ffbidx.ffbidx_impl import indexer, index, release
from ffbidx.ffbidx_cl import Indexer

ffbidx_cl.__version__ = ffbidx_impl.__version__
__version__ = ffbidx_impl.__version__
