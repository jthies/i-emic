set(TEST_INCLUDE_DIRS
  ../continuation/
  ../topo/
  ../lyapunov/
  ../transient/
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

set(TEST_SOURCES
  test_map.C
  test_domain.C
  test_vector.C
  test_jdqz.C
  test_topo.C
  test_ocean.C
  trns_ocean.C
  intt_ocean.C
  reft_ocean.C
  test_parameterlist.C
  intt_2dmoc.C
  test_atmos.C
  test_oceanatmos.C
  test_seaice.C
  test_coupled.C
  intt_coupled.C
  test_integrals.C
  test_matrix.C
  test_ams.C
  )

find_package(GTest QUIET)
if (NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

if (TARGET GTest::gtest)
  set(GTEST_TARGET GTest::gtest)
elseif (TARGET gtest)
  set(GTEST_TARGET gtest)
else()
  set(GTEST_TARGET ${GTEST_LIBRARIES})
endif()

if (GTEST_INCLUDE_DIRS)
  include_directories(BEFORE ${GTEST_INCLUDE_DIRS})
endif()

set(TEST_LIBRARIES
  iemic
  ${MPI_CXX_LIBRARIES}
  ${Belos_LIBRARIES}
  ${Belos_TPL_LIBRARIES}
  ${Epetra_LIBRARIES}
  ${Epetra_TPL_LIBRARIES}
  ${EpetraExt_LIBRARIES}
  ${EpetraExt_TPL_LIBRARIES}
  ${GTEST_TARGET}
  ${CMAKE_THREAD_LIBS_INIT}
)

if (RAILS_FOUND)
  list(APPEND TEST_LIBRARIES ${RAILS_LIBRARIES})
  list(APPEND TEST_INCLUDE_DIRS ${RAILS_INCLUDE_DIRS})
  add_definitions(-DHAVE_RAILS)
endif()

if (NOT MPIEXEC_EXECUTABLE)
  set(MPIEXEC_EXECUTABLE mpirun)
  set(MPIEXEC_NUMPROC_FLAG -np)
endif()

if(APPLE)
  set(MPI_OVERSUBSCRIBE "--oversubscribe")
endif()

foreach(test_source ${TEST_SOURCES})
  get_filename_component(test_name ${test_source} NAME_WE)
  add_executable(${test_name} ${test_source} test_utils.C)

  string(REPLACE "_" ";" test_name_list ${test_name})
  list(GET test_name_list 1 test_base_name)
  file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/test/${test_base_name})
  add_test(NAME ${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${MPI_OVERSUBSCRIBE} $<TARGET_FILE:${test_name}>
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/${test_base_name})

  target_compile_definitions(${test_name} PUBLIC ${COMP_IDENT})
  target_include_directories(${test_name} PUBLIC ${TEST_INCLUDE_DIRS})
  target_link_libraries(${test_name} PUBLIC ${TEST_LIBRARIES})
endforeach()

# additional parallel tests
get_filename_component(test_name test_matrix.C NAME_WE)
add_test(NAME partest_matrix_1 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${MPI_OVERSUBSCRIBE} $<TARGET_FILE:${test_name}>
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/matrix)
add_test(NAME partest_matrix_2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} ${MPI_OVERSUBSCRIBE} $<TARGET_FILE:${test_name}>
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/matrix)
add_test(NAME partest_matrix_4 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${MPI_OVERSUBSCRIBE} $<TARGET_FILE:${test_name}>
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/matrix)
add_test(NAME partest_matrix_8 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8 ${MPIEXEC_PREFLAGS} ${MPI_OVERSUBSCRIBE} $<TARGET_FILE:${test_name}>
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/matrix)

