
message(STATUS "model/segment")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_segment)

# SOURCE
file(GLOB PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/segment/*.h"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/segment/*.cc"
)

if (ENABLE_NNDEPLOY_PLUGIN_SEGMENT_SEGMENT_ANYTHING)
  file(GLOB_RECURSE SEGMENT_ANYTHING_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/segment/segment_anything/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/segment/segment_anything/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${SEGMENT_ANYTHING_SOURCE})
endif()

## TARGET
add_library(${PLUGIN_BINARY} ${NNDEPLOY_LIB_TYPE} ${PLUGIN_SOURCE} ${PLUGIN_OBJECT})
## DIRECTORY
set_property(TARGET ${PLUGIN_BINARY} PROPERTY FOLDER ${NNDEPLOY_PLUGIN_DIRECTORY})
## DEPEND_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_DEPEND_LIBRARY})
## SYSTEM_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_SYSTEM_LIBRARY}) 
## THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_THIRD_PARTY_LIBRARY}) 
## NNDEPLOY_FRAMEWORK_BINARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_FRAMEWORK_BINARY}) 
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_basic)
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_infer) 
## NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY}) 
## install
if(SYSTEM.Windows)
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
else() 
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)


