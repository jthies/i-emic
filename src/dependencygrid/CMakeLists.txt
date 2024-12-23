add_library(dependencygrid DependencyGrid.C)

target_compile_definitions(dependencygrid PUBLIC ${COMP_IDENT})
target_include_directories(dependencygrid PUBLIC .)
