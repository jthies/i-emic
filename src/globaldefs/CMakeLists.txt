add_library(globaldefs SHARED GlobalDefinitions.C)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/GlobalDefinitions.H.in
               ${CMAKE_CURRENT_BINARY_DIR}/GlobalDefinitions.H)

target_link_libraries(globaldefs PRIVATE
    ${MPI_CXX_LIBRARIES}
    ${Teuchos_LIBRARIES}
    ${Teuchos_TPL_LIBRARIES}
    ${Epetra_LIBRARIES}
    ${Epetra_TPL_LIBRARIES}
)

target_compile_definitions(globaldefs PUBLIC ${COMP_IDENT})
target_include_directories(globaldefs PUBLIC .)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GlobalDefinitions.H DESTINATION include)
install(TARGETS globaldefs DESTINATION lib)
