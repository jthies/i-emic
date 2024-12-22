add_library(globaldefs SHARED GlobalDefinitions.C)

set(HAVE_AMESOS ${Trilinos_Amesos_FOUND})
set(HAVE_ANASAZI ${Trilinos_Anasazi_FOUND})
set(HAVE_AZTECOO ${Trilinos_AztecOO_FOUND})
set(HAVE_BELOS ${Trilinos_Belos_FOUND})
set(HAVE_IFPACK ${Trilinos_Ifpack_FOUND})
set(HAVE_ML ${Trilinos_ML_FOUND})
set(HAVE_NOX ${Trilinos_NOX_FOUND})
set(HAVE_ZOLTAN ${Trilinos_Zoltan_FOUND})
set(HAVE_XDMF ${EpetraExt_ENABLE_HDF5})


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
