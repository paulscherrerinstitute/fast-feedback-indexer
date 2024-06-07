    git_exe = find_program('git')
    if git_exe.found()
        version_txt = '"'+run_command(git_exe, 'log', '--pretty=format:%H: %aD', '-1', check: true).stdout()+'"'
    else
        version_txt = 'unknown'
    endif
    conf_data = configuration_data()
    conf_data.set('VERSION_TXT', version_txt)

    version_file = configure_file(input: '../version.h.in',
        output: 'version.h',
        configuration: conf_data)
    version_dep = declare_dependency(sources: version_file)
