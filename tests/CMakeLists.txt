


set(TEST_SRC
	test_main.cpp
	Vec4Tests.cpp
	
	#adding more test files 
)


add_executable(VortrixMathTests ${TEST_SRC})
target_link_libraries(VortrixMathTests PRIVATE VortrixMath)
target_include_directories(VortrixMathTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})