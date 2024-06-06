if get_option('install-simple-data-files').enabled()
    simple_files_dep = declare_dependency()

    install_subdir('files',
        install_dir: get_option('datadir')+'/ffbidx/data/files',
        install_mode: 'r--r--r--',
        strip_directory: true)
endif

if get_option('install-simple-data-reader').enabled()
    subdir('reader/src')
endif
