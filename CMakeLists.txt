cmake_minimum_required(VERSION 3.15)

#generate project
project(VortrixMath VERSION 1.0 LANGUAGES CXX)


#ISO c++ 17 standard required for features like (std::hash, std::unorderedmap, std::filesystem, etc)
#target_compile_features(crrt_compiler_flags INTERFACE cxx_std_17) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VX_DIR ${CMAKE_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")

option(USE_SSE "Enable SSE" ON)
option(BUILD_TESTS "build unit tests" ON)
option(BUILD_PLAYGROUND "build playground" ON)


#maths library 
include(VortrixMath)	


#test program
if(BUILD_TESTS)
	include(UnitTests)
endif()

if(BUILD_PLAYGROUND)
	include(VortrixPlayground)
endif()

#vs startup project
if(MSVC)
			
	if(BUILD_PLAYGROUND)
		set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			 PROPERTY VS_STARTUP_PROJECT VortrixPlayground)
			 
		if(BUILD_TESTS)
		
			add_custom_command(TARGET VortrixPlayground
				PRE_BUILD
				COMMAND $<TARGET_FILE:VortrixMathTests>
				COMMENT "Running unit tests before playground"
				)
		endif()
		
	elseif(BUILD_TESTS)
		set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
					PROPERTY VS_STARTUP_PROJECT VortrixMathTests)
	endif()
	 
 endif()
 
