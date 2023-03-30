option(BUILD_CRYSTFEL_INTEGRATION "Build fast indexer library CrystFEL C wrapper" OFF)

IF(BUILD_CRYSTFEL_INTEGRATION)
    ADD_LIBRARY(fast_indexer_crystfel SHARED c-wrapper.cpp ffbidx/c-wrapper.h)
    TARGET_LINK_LIBRARIES(fast_indexer_crystfel fast_indexer)

    install(TARGETS fast_indexer_crystfel
            LIBRARY
            COMPONENT ffbidx_common)
    install(FILES ffbidx/c-wrapper.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ffbidx
            COMPONENT ffbidx_development)

    configure_file(fast_indexer_crystfel.pc.in
            fast_indexer_crystfel.pc
            @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fast_indexer_crystfel.pc
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
            COMPONENT ffbidx_development_extra)
ENDIF()
