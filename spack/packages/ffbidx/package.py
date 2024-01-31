# Copyright 2013-2022 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install ffbidx
#
# You can edit this file again by typing:
#
#     spack edit ffbidx
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class Ffbidx(CMakePackage, CudaPackage):
    """Develop an indexer for fast feedback"""

    homepage = "https://github.com/paulscherrerinstitute/fast-feedback-indexer.git"
    git = "https://github.com/paulscherrerinstitute/fast-feedback-indexer.git"

    maintainers = ["hcstadler"]

    version('main')

    variant('build_type', default='Release',
            description='CMake build type',
            values=('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel'))
    variant("cuda", default=True, description="Build with CUDA")
    variant("fast_indexer", default=True, description="Build the fast feedback indexer library" )
    variant("python", default=False, description="Build python module")
    variant("simple_data_files", default=False, description="Install simple data files")
    variant("simple_data_indexer", default=False, description="Build simple data reader library and indexer")
    variant("test_all", default=False, description="Enable testing")

    depends_on("python@3:", when='+python')
    depends_on("py-numpy", when='+python')
    depends_on("eigen@3.4:", type=('build', 'link'), when='+simple_data_indexer')
    depends_on("eigen@3.4:", type=('build', 'link'), when='+fast_indexer')
    depends_on("cmake@3.21:", type='build')

    conflicts('cuda_arch=none', when='+fast_indexer',
              msg='CUDA architecture is required when building the fast feedback indexer library')
    conflicts("~cuda", when='+fast_indexer',
              msg="The fast feedback indexer library requires CUDA'")
    conflicts('+python', when='~fast_indexer',
              msg='Python module needs the fast feedback indexer library')
    conflicts('+test_all', when='~simple_data_indexer~fast_indexer',
              msg='Tests need the build of both simple data reader and fast feedback indexer library')
    conflicts('+simple_data_indexer', when='~fast_indexer',
              msg='Simple data indexer needs the fast feedback indexer library')

    # Conflicts for compilers without C++17 support
    conflicts('gcc@:6.5.0')
    conflicts('intel@:18.0.5')

    # Add ctest stage if +test_all in spec
    @run_after('build')
    def test(self):
        if '+test_all' in self.spec:
            with working_dir(self.build_directory):
                ctest = Executable("ctest")
                ctest()

    def cmake_args(self):
        args = [
            self.define('CMAKE_INSTALL_LIBDIR', 'lib'),
            self.define_from_variant('BUILD_FAST_INDEXER', 'fast_indexer'),
            self.define_from_variant('BUILD_FAST_INDEXER_STATIC', 'fast_indexer'),
            self.define_from_variant('SIMPLE_DATA_INDEXER', 'simple_data_indexer'),
            self.define_from_variant('REFINED_SIMPLE_DATA_INDEXER', 'simple_data_indexer'),
            self.define_from_variant('BUILD_SIMPLE_DATA_READER', 'simple_data_indexer'),
            self.define_from_variant('CMAKE_CUDA_ARCHITECTURES', 'cuda_arch'),
            self.define_from_variant('INSTALL_SIMPLE_DATA_FILES', 'simple_data_files'),
            self.define_from_variant('PYTHON_MODULE', 'python'),
            self.define_from_variant('TEST_ALL', 'test_all'),
        ]

        if 'python' in self.spec:
            python_root_dir = self.spec['python'].prefix
            args.append(self.define('Python3_ROOT_DIR', python_root_dir))
            print(f'using python from {python_root_dir}')

        return args

    # Set PATHS for run time
    def setup_run_environment(self, env):
        env.append_path('PATH', self.prefix.bin)
        env.append_path('CPATH', self.prefix.include)
        env.append_path('LD_LIBRARY_PATH', self.prefix.lib)
        env.append_path('LIBRARY_PATH', self.prefix.lib)
        env.append_path('PKG_CONFIG_PATH', self.prefix.share.ffbidx.pkgconfig)
        if '+python' in self.spec:
            env.append_path('PYTHONPATH', self.prefix.lib.ffbidx)
