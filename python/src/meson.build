if get_option('include-python-api').enabled()
    pymod = import('python')
    python = pymod.find_installation('python3', modules: 'numpy')
    python_dep = python.dependency()
    incdir_numpy = include_directories(
        run_command(python,
            ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
            check : true
        ).stdout().strip()
    )

    shared_library('ffbidx_impl',
        sources: 'ffbidx_impl.cpp',
        dependencies: [ ffbidx_dep, python_dep ],
        include_directories: incdir_numpy,
        name_prefix: '',
        install_rpath: join_paths(get_option('prefix'), get_option('libdir')),
        install: true,
        install_dir: get_option('libdir')+'/ffbidx',
        cpp_args: cxx_flags,
        link_args: nvcc_link_flags)

    install_data(
        [
            '__init__.py',
            'ffbidx_cl.py'
        ],
        install_dir: get_option('libdir')+'/ffbidx'
    )

    fast_feedback_indexer_ENV_SETUP += 'ffbidx_PYTHONPATH='+join_paths(get_option('prefix'), get_option('libdir'))+';'
endif
