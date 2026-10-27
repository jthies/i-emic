add_library(seaice SeaIce.C)

target_include_directories(seaice PUBLIC .)

target_link_libraries(seaice PRIVATE dependencygrid ocean utils)

install(TARGETS seaice DESTINATION lib)
