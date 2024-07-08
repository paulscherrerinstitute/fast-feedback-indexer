add_subdirectory(reader)

option(INSTALL_SIMPLE_DATA_FILES "Install simple data files" OFF)

if(INSTALL_SIMPLE_DATA_FILES)
    install(DIRECTORY files
            DESTINATION ${CMAKE_INSTALL_DATADIR}/ffbidx/data
            COMPONENT ffbidx_simpledata
            FILES_MATCHING PATTERN "*.txt")
endif(INSTALL_SIMPLE_DATA_FILES)
