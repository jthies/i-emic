add_library(seaice STATIC SeaIce.C)

set_target_properties(seaice PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(seaice PUBLIC .)

target_link_libraries(seaice PRIVATE atmosphere dependencygrid ocean utils)
