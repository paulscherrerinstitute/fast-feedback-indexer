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
    endif
endif
