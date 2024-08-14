simple_data_dep = declare_dependency(include_directories: '.')

conf_data = configuration_data()
conf_data.set('PROJECT_VERSION', '1.0.1')
conf_data.set('CMAKE_INSTALL_PREFIX', get_option('prefix'))
conf_data.set('CMAKE_INSTALL_INCLUDEDIR', get_option('includedir'))
conf_data.set('PROJECT_NAME', 'simple_data')
conf_data.set('PROJECT_DESCRIPTION', 'Reader for simple data')

configure_file(input: 'simple_data.pc.in',
               output: 'simple_data.pc',
               configuration: conf_data,
               install_dir: get_option('datadir')+'/ffbidx/pkgconfig',
               install_mode: 'r--r--r--')

install_headers('ffbidx/simple_data.h',
    subdir: 'ffbidx',
    install_mode: 'r--r--r--')
