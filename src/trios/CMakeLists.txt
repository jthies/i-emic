add_library(trios STATIC
  TRIOS_Domain.C
  TRIOS_BlockPreconditioner.C
  TRIOS_HYMLSBlockPreconditioner.C
  TRIOS_FROSchPreconditioner.cpp
  TRIOS_Saddlepoint.C
  TRIOS_SolverFactory.C
  TRIOS_Static.C
  )

find_package(HYMLS)
if (HYMLS_FOUND)
  target_link_libraries(trios PRIVATE ${HYMLS_LIBRARIES})
  target_include_directories(trios PRIVATE ${HYMLS_INCLUDE_DIRS})
endif ()

target_link_libraries(trios PRIVATE ${ShyLU_LIBRARIES})
target_link_libraries(trios PRIVATE ${ShyLU_TPL_LIBRARIES})

target_link_libraries(trios PRIVATE globaldefs ifpack_mrilu)
target_include_directories(trios PUBLIC .)
target_compile_definitions(trios PUBLIC ${COMP_IDENT})

install(FILES TRIOS_Domain.H DESTINATION include)
