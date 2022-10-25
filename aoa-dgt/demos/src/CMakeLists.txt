FIND_PACKAGE(cuPoly REQUIRED)

# demo: simple_polyop
CUDA_ADD_EXECUTABLE(
    simple_polyop
    EXCLUDE_FROM_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_polyop.cpp
)

TARGET_LINK_LIBRARIES(
    simple_polyop
    -lntl 
    -lgmp 
    -fopenmp
    ${CUDA_curand_LIBRARY}
    cupolybfv
)

# custom target
ADD_CUSTOM_COMMAND(
    OUTPUT demos.out
    DEPENDS simple_polyop
    COMMAND ${CMAKE_COMMAND} -E touch demos.out
    )
ADD_CUSTOM_TARGET(
    demos
    DEPENDS demos.out
    )