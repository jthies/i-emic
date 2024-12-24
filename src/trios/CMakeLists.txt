add_library(trios
  TRIOS_Domain.C
  TRIOS_BlockPreconditioner.C
  TRIOS_Saddlepoint.C
  TRIOS_SolverFactory.C
  TRIOS_Static.C
  TRIOS_MatrixUtils.C
  NOX_Epetra_LinearSystem_Belos.C
)

target_link_libraries(trios PRIVATE ${NOX_LIBRARIES} ${NOX_TPL_LIBRARIES}
                                    ${Belos_LIBRARIES} ${Belos_TPL_LIBRARIES}
                                    ${Amesos_LIBRARIES} ${Amesos_TPL_LIBRARIES}
                                    ${AztecOO_LIBRARIES} ${Ifpack_LIBRARIES} ${Ifpack_TPL_LIBRARIES}
                                    ${ML_LIBRARIES} ${ML_TPL_LIBRARIES}
                                    globaldefs ifpack_mrilu)
target_include_directories(trios PUBLIC .)

install(FILES TRIOS_Domain.H DESTINATION include)
