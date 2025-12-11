set(VX_MATH_SRC_DIR ${VX_DIR}/source)


file(GLOB_RECURSE MATH_HEADERS 
								${VX_MATH_SRC_DIR}/*.h
								${VX_MATH_SRC_DIR}/*.inl)


add_library(VortrixMath INTERFACE)
target_include_directories(VortrixMath INTERFACE ${PROJECT_SOURCE_DIR}/source)
if(USE_SSE)
	target_compile_definitions(VortrixMath INTERFACE VX_USE_SSE)
	#target_compile_definitions(VortrixMath PUBLIC VX_USE_SSE)
endif()

source_group(TREE ${VX_MATH_SRC_DIR} PREFIX "VortrixMath" FILES ${MATH_HEADERS})
