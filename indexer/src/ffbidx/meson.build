git_exe = find_program('git')
if git_exe.found()
    result = run_command(git_exe, 'log', '--pretty=format:%h', '-1', check: false)
endif

if git_exe.found() and result.returncode() == 0
    version_txt = '"'+meson.project_version()+'-'+result.stdout()+'"'
else
    version_txt = '"'+meson.project_version()+'"'
endif
message('Version string: '+version_txt)

conf_data = configuration_data()
conf_data.set('VERSION_TXT', version_txt)

version_file = configure_file(input: '../version.h.in',
    output: 'version.h',
    configuration: conf_data)
version_dep = declare_dependency(sources: version_file)
