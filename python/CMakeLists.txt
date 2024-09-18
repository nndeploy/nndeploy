
project(pynndeploy)

set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

set(PACKAGE_VERSION ${NNDEPLOY_VERSION})
add_definitions(-DVERSION_INFO="${PACKAGE_VERSION}")

add_subdirectory(pybind11)

include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
pybind11_add_module(pynndeploy src/main.cc)

set_target_properties(pynndeploy PROPERTIES OUTPUT_NAME "nndeploy")
target_link_libraries(pynndeploy PUBLIC nndeploy_framework)
set_target_properties(pynndeploy PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nndeploy")
set_property(TARGET pynndeploy PROPERTY FOLDER "python")

if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    add_custom_command(TARGET pynndeploy POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/nndeploy/nndeploy${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION} 
        ${PROJECT_SOURCE_DIR}/nndeploy/_C/nndeploy${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION})
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")


configure_file(setup.py.i ${PROJECT_SOURCE_DIR}/setup.py)
