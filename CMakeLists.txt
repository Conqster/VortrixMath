cmake_minimum_required(VERSION 3.15)

#generate project
project(VortrixMath VERSION 1.0 LANGUAGES CXX)


#ISO c++ 17 standard required for features like (std::hash, std::unorderedmap, std::filesystem, etc)
#target_compile_features(crrt_compiler_flags INTERFACE cxx_std_17) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_library(VortrixMath INTERFACE)
target_include_directories(VortrixMath INTERFACE ${PROJECT_SOURCE_DIR}/source)


#test program
option(BUILD_TESTS "build the example/test executable" on)
if(BUILD_TESTS)
	file(GLOB_RECURSE VortrixMathTest_SRC "tests/*.cpp" "tests/*.h")
	add_executable(VortrixMathTest ${VortrixMathTest_SRC})
	target_link_libraries(VortrixMathTest PRIVATE VortrixMath)
	set_target_properties(VortrixMathTest PROPERTIES
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		)
endif()

#vs startup project
if(MSVC AND BUILD_TESTS)
	 set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			 PROPERTY VS_STARTUP_PROJECT VortrixMathTest)
 endif()