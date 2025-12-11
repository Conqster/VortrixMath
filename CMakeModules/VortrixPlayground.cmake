
set(PLAYGROUND_DIR ${VX_DIR}/Playground)

file(GLOB_RECURSE PLAYGROUND_SRC ${PLAYGROUND_DIR}/*.h
								 ${PLAYGROUND_DIR}/*.cpp)

source_group(TREE ${PLAYGROUND_DIR} PREFIX "Playground_src" FILES ${PLAYGROUND_SRC})

add_executable(VortrixPlayground 
								${PLAYGROUND_SRC}
								${MATH_HEADERS})
target_link_libraries(VortrixPlayground PRIVATE VortrixMath)
target_include_directories(VortrixPlayground PRIVATE ${PLAYGROUND_DIR})


