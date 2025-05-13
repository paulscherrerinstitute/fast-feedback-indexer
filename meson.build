project('ffbidx',
    ['cpp', 'cuda'],
    version: '1.1.3',
    meson_version: '>=1.1',
    license: 'BSD-3-Clause',
    license_files: ['LICENSE.md'])

# Shell script variable settings for setup-env.sh
fast_feedback_indexer_ENV_SETUP  = ''

nvcc_compiler_flags = '-Wall,-Wextra,-Ofast'
cpu_arch = ''
if get_option('cpu-arch') != 'ignore'
    cpu_arch = '-march='+get_option('cpu-arch')
    nvcc_compiler_flags += ','+cpu_arch
endif

nvcc_flags = [ '--forward-unknown-to-host-compiler', '--std=c++17', '--ftz=true', '--fmad=true' ]
if get_option('buildtype') == 'release'
    nvcc_flags += [
        '--compiler-options', nvcc_compiler_flags,
        '-DNDEBUG',
        '--extra-device-vectorization',
        '-O3'
    ]
endif
if get_option('gpu-arch') != 'ignore'
    nvcc_flags += [ '-gencode', get_option('gpu-arch') ]
    add_project_link_arguments(
        nvcc_flags,
        language: 'cuda')
endif

cxx_flags = [ '--std=c++17' ]
extra_warning_flags = [ '-Wall', '-Wextra' ]
if get_option('buildtype') == 'release'
    cxx_flags += [
        '-O3',
        '-DNDEBUG'
    ]
    if cpu_arch != ''
        cxx_flags += cpu_arch
    endif
endif

if get_option('all-warnings').enabled()
    cxx_flags += extra_warning_flags
endif

nvcc_link_flags = '--forward-unknown-to-host-compiler'

subdir('data/simple')
subdir('indexer/src')
subdir('python/src')
subdir('tests/src')
subdir('examples/cpp-simple-data-bulk-indexer')

conf_data = configuration_data()
conf_data.set('fast_feedback_indexer_ENV_SETUP', fast_feedback_indexer_ENV_SETUP)
configure_file(input: 'setup-env.sh.in',
        output: 'setup-env.sh',
        configuration: conf_data,
        install_dir: get_option('datadir')+'/ffbidx',
        install_mode: 'r--r--r--')
