if get_option('build-tests').enabled()
    if get_option('install-simple-data-reader').enabled()
        test = executable('test_simple_data_reader',
            'simple_data_test.cpp',
            dependencies: simple_data_dep)
        test('read_file_0',
            test,
            args:[ meson.project_source_root()+'/data/simple/files/image0_local.txt' ])
    endif

    if get_option('build-ffbidx').enabled()
        test = executable('test_indexer_exception',
            'test_indexer_exception.cpp',
            dependencies: ffbidx_dep)
        test('fast_feedback_exception',
            test)
        
        test = executable('test_indexer_obj',
            'test_indexer_obj.cpp',
            dependencies: ffbidx_dep)
        test('indexer_object',
            test)

        test = executable('test_indexer',
            'test_indexer.cpp',
            dependencies: [ffbidx_dep, simple_data_dep])
        test('index_file_0',
            test,
            args:[ meson.project_source_root()+'/data/simple/files/image0_local.txt' ])
        
        test = executable('test_capi',
            'test_capi.cpp',
            dependencies: [ffbidx_dep, simple_data_dep])
        test('capi_index_file_0',
            test,
            args:[ meson.project_source_root()+'/data/simple/files/image0_local.txt' ])
    endif
endif

if get_option('build-simple-indexers').enabled()
    if get_option('install-simple-data-reader').disabled()
        error('"-D install-simple-data-reader=enabled" option is required for enabling option "build-simple-indexers"')
    endif
    executable('simple_data_indexer',
        sources: 'simple_data_indexer.cpp',
        dependencies: [ simple_data_dep, ffbidx_dep ],
        install: true,
        install_rpath: join_paths(get_option('prefix'), get_option('libdir')))
    executable('refined_simple_data_indexer',
        sources: 'refined_simple_data_indexer.cpp',
        dependencies: [ simple_data_dep, ffbidx_dep ],
        install: true,
        install_rpath: join_paths(get_option('prefix'), get_option('libdir')))

    fast_feedback_indexer_ENV_SETUP += 'ffbidx_PATH='+join_paths(
        get_option('prefix'), 
        get_option('bindir')
    )+';'
endif
