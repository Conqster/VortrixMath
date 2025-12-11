
set(UNIT_TESTS_DIR ${VX_DIR}/UnitTests)

file(GLOB_RECURSE TEST_SRC ${UNIT_TESTS_DIR}/*.h
						   ${UNIT_TESTS_DIR}/*.cpp)

#source_group(TREE ${VX_MATH_SRC_DIR} PREFIX "VortrixMath" FILES ${MATH_HEADERS})
source_group(TREE ${UNIT_TESTS_DIR} PREFIX "UnitTests_src" FILES ${TEST_SRC})

add_executable(VortrixMathTests 
								${TEST_SRC}
								${MATH_HEADERS})
target_link_libraries(VortrixMathTests PRIVATE VortrixMath)
target_include_directories(VortrixMathTests PRIVATE ${UNIT_TESTS_DIR})


