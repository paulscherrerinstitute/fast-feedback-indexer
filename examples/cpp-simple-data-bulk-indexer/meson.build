if get_option('build-simple-indexers').enabled()
    if get_option('install-simple-data-reader').disabled()
        error('"-D install-simple-data-reader=enabled" option is required for enabling option "build-simple-indexers"')
    endif
    extra_flags = []
    if meson.get_compiler('cpp').get_id() == 'gcc'
        extra_flags += [ '-fwhole-program' ]
    endif
    executable('simple-data-bulk-indexer',
        sources: 'simple-data-bulk-indexer.cpp',
        dependencies: [ simple_data_dep, ffbidx_dep, cuda_dep ],
        cpp_args: cxx_flags + extra_flags,
        install: true,
        install_rpath: join_paths(get_option('prefix'), get_option('libdir')))

    # This is done in the tests folder for the other simple indexers
    # fast_feedback_indexer_ENV_SETUP += 'ffbidx_PATH='+join_paths(
    #     get_option('prefix'), 
    #     get_option('bindir')
    # )+';'
endif
