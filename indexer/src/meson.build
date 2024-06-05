if get_option('build-ffbidx').enabled()
    eigen3 = dependency('eigen3')

    conf_data = configuration_data()
    conf_data.set('PROJECT_VERSION', '0.1.0')
    conf_data.set('CMAKE_INSTALL_PREFIX', get_option('prefix'))
    conf_data.set('CMAKE_INSTALL_INCLUDEDIR', get_option('includedir'))
    conf_data.set('CMAKE_INSTALL_LIBDIR', get_option('libdir'))
    conf_data.set('PROJECT_NAME', 'fast_indexer')
    conf_data.set('PROJECT_DESCRIPTION', 'Library for fast feedback indexing using brute force sampling')

    configure_file(input: 'fast_indexer.pc.in',
        output: 'fast_indexer.pc',
        configuration: conf_data,
        install_dir: get_option('datadir')+'/ffbidx/pkgconfig',
        install_mode: 'r--r--r--')
    
    git_exe = find_program('git')
    if git_exe.found()
        version_txt = run_command(git_exe, 'log', '--pretty=format:%H: %aD', '-1', check: true).stdout()
    else
        version_txt = 'unknown'
    endif
    conf_data = configuration_data()
    conf_data.set('VERSION_TXT', version_txt)

    version_file = configure_file(input: 'version.h.in',
        output: 'version.h',
        configuration: conf_data)
    version_dep = declare_dependency(sources: version_file)

    thread_dep = dependency('threads')

    cuda_dep = dependency('cuda', version : '>=10')

    sources = ['log.cpp', 'indexer.cpp', 'indexer_gpu.cu']
    if get_option('include-c-api').enabled()
        sources += 'c_api.cpp'
    endif

    cpu_arch = ''
    if get_option('cpu-arch') != 'ignore'
        cpu_arch = '-march='+get_option('cpu-arch')
    endif

    nvcc_flags = [ '--std=c++17', '--ftz=true', '--fmad=true' ]
    if get_option('buildtype') == 'release'
        nvcc_flags += [
            '--compiler-options', cpu_arch+',-Wall,-Wextra,-Ofast',
            '-DNDEBUG',
            '--extra-device-vectorization',
            '-O3'
        ]
    endif
    if get_option('gpu-arch') != 'ignore'
        nvcc_flags += [ '-gencode', get_option('gpu-arch') ]
    endif

    cxx_flags = [ '--std=c++17' ]
    if get_option('buildtype') == 'release'
        cxx_flags += [
            '-Ofast',
            '-Wall',
            '-Wextra',
            '-DNDEBUG'
        ]
        if cpu_arch != ''
            cxx_flags += cpu_arch
        endif
    endif

    ffbidx = library('fast_indexer',
        sources,
        include_directories: 'ffbidx',
        dependencies: [eigen3, cuda_dep, version_dep, thread_dep],
        install: true,
        install_mode: 'r-xr-xr-x',
        cuda_args: nvcc_flags,
        cpp_args: cxx_flags)
    ffbidx_dep = declare_dependency(
        dependencies: eigen3,
        include_directories: '.',
        link_with: ffbidx,
    )

    headers = [
        'ffbidx/envvar.h',
        'ffbidx/exception.h',
        'ffbidx/indexer.h',
        'ffbidx/log.h',
        'ffbidx/refine.h',
        version_file]
    if get_option('include-c-api').enabled()
        headers += 'ffbidx/c_api.h'
    endif

    install_headers(headers, subdir: 'ffbidx')
endif
