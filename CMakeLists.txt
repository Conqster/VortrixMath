cmake_minimum_required(VERSION 3.15)

#generate project
project(VoltrixMath VERSION 1.0 LANGUAGES CXX)


#the C++ use standard 
#add_library(crrt_compiler_flags INTERFACE)
#ISO c++ 17 standard required for features like (std::hash, std::unorderedmap, std::filesystem, etc)
#target_compile_features(crrt_compiler_flags INTERFACE cxx_std_17) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_library(VoltrixMath INTERFACE)
target_include_directories(VoltrixMath INTERFACE ${PROJECT_SOURCE_DIR}/source)


#test program
option(BUILD_TESTS "build the example/test executable" on)
if(BUILD_TESTS)
	file(GLOB_RECURSE VoltrixMathTest_SRC "tests/*.cpp" "tests/*.h")
	add_executable(VoltrixMathTest ${VoltrixMathTest_SRC})
	target_link_libraries(VoltrixMathTest PRIVATE VoltrixMath)
	set_target_properties(VoltrixMathTest PROPERTIES
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		)
endif()

#vs startup project
if(MSVC AND BUILD_TESTS)
	 set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			 PROPERTY VS_STARTUP_PROJECT VoltrixMathTest)
 endif()