add_library(atmosphere AtmosLocal.C Atmosphere.C)

target_link_libraries(atmosphere PRIVATE dependencygrid ocean seaice utils)

target_include_directories(atmosphere PUBLIC .)
