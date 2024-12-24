add_library(atmosphere STATIC AtmosLocal.C Atmosphere.C)

set_target_properties(atmosphere PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(atmosphere PRIVATE dependencygrid ocean seaice utils)

target_include_directories(atmosphere PUBLIC .)
