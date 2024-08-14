git_exe = find_program('git')
if git_exe.found()
    result = run_command(git_exe, 'log', '--pretty=format:"%H: %aD"', '-1', check: false)
endif

if git_exe.found() and result.returncode() == 0
    version_txt = result.stdout()
else
    version_txt = '"1.0.0"'
endif
message('Version string: '+version_txt)

conf_data = configuration_data()
conf_data.set('VERSION_TXT', version_txt)

version_file = configure_file(input: '../version.h.in',
    output: 'version.h',
    configuration: conf_data)
version_dep = declare_dependency(sources: version_file)
