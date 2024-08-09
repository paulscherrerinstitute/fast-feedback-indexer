if get_option('build-ffbidx').enabled()
    subdir('ffbidx')

    pkg_config_dir = get_option('datadir')+'/ffbidx/pkgconfig'

    fs = import('fs')
    if get_option('eigen-source-dir') != 'ignore'
        eigen_dir = join_paths('../..', get_option('eigen-source-dir'))
        if fs.is_dir(eigen_dir+'/Eigen')
            eigen3_dep = declare_dependency( include_directories: [
                eigen_dir,
                eigen_dir+'/unsupported'
            ])
            message('Using Eigen source in '+eigen_dir)
        else
            error('Eigen include directory not found under eigen-source-dir:', get_option('eigen-source-dir'))
        endif
    else
        eigen3_dep = dependency('eigen3')
    endif

    thread_dep = dependency('threads')

    cuda_dep = dependency('cuda', version : '>=10')

    sources = ['log.cpp', 'indexer.cpp', 'indexer_gpu.cu']
    if get_option('include-c-api').enabled()
        sources += 'c_api.cpp'
    endif

    conf_data = configuration_data()
    conf_data.set('PROJECT_VERSION', '1.0.0')
    conf_data.set('CMAKE_INSTALL_PREFIX', get_option('prefix'))
    conf_data.set('CMAKE_INSTALL_INCLUDEDIR', get_option('includedir'))
    conf_data.set('CMAKE_INSTALL_LIBDIR', get_option('libdir'))
    conf_data.set('PROJECT_NAME', 'fast_indexer')
    conf_data.set('PROJECT_DESCRIPTION', 'Library for fast feedback indexing using brute force sampling')

    if get_option('default_library') in [ 'static', 'both' ]
        ffbidx_static = static_library('fast_indexer_static',
            sources,
            include_directories: 'ffbidx',
            dependencies: [eigen3_dep, cuda_dep, version_dep, thread_dep],
            install: true,
            install_mode: 'r-xr-xr-x',
            cuda_args: nvcc_flags,
            cpp_args: cxx_flags,
            link_args: nvcc_link_flags)

        configure_file(input: 'fast_indexer_static.pc.in',
            output: 'fast_indexer_static.pc',
            configuration: conf_data,
            install_dir: pkg_config_dir,
            install_mode: 'r--r--r--')
    endif

    if get_option('default_library') in [ 'shared', 'both' ]
        ffbidx = shared_library('fast_indexer',
            sources,
            include_directories: 'ffbidx',
            dependencies: [eigen3_dep, cuda_dep, version_dep, thread_dep],
            install: true,
            install_mode: 'r-xr-xr-x',
            cuda_args: nvcc_flags,
            cpp_args: cxx_flags,
            link_args: nvcc_link_flags)

        configure_file(input: 'fast_indexer.pc.in',
            output: 'fast_indexer.pc',
            configuration: conf_data,
            install_dir: pkg_config_dir,
            install_mode: 'r--r--r--')
    endif

    if get_option('default_library') in [ 'shared', 'both' ]
        ffbidx_dep = declare_dependency(
            dependencies: eigen3_dep,
            include_directories: '.',
            link_with: ffbidx,
        )
    else
        ffbidx_dep = declare_dependency(
            dependencies: eigen3_dep,
            include_directories: '.',
            link_with: ffbidx_static,
        )
    endif

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

    fast_feedback_indexer_ENV_SETUP += 'ffbidx_PKG_CONFIG_PATH='+join_paths(
        get_option('prefix'), 
        pkg_config_dir
    )+';'
    fast_feedback_indexer_ENV_SETUP += 'ffbidx_CPLUS_INCLUDE_PATH='+join_paths(
        get_option('prefix'),
        get_option('includedir')
    )+';'
    fast_feedback_indexer_ENV_SETUP += 'ffbidx_LIBRARY_PATH='+join_paths(
        get_option('prefix'),
        get_option('libdir')
    )+';'
endif
