message(STATUS "plugin/detect")

# set
set(PLUGIN_SOURCE)
set(PLUGIN_OBJECT)
set(PLUGIN_BINARY nndeploy_plugin_track)

# SOURCE
file(GLOB PLUGIN_SOURCE
  "${PLUGIN_ROOT_PATH}/include/nndeploy/track/*.h"
  "${PLUGIN_ROOT_PATH}/source/nndeploy/track/*.cc"
)

if(ENABLE_NNDEPLOY_PLUGIN_TRACK_FAIRMOT)
  file(GLOB_RECURSE FAIRMOT_SOURCE
    "${PLUGIN_ROOT_PATH}/include/nndeploy/track/fairmot/*.h"
    "${PLUGIN_ROOT_PATH}/source/nndeploy/track/fairmot/*.cc"
  )
  set(PLUGIN_SOURCE ${PLUGIN_SOURCE} ${FAIRMOT_SOURCE})
endif()

# # TARGET
add_library(${PLUGIN_BINARY} ${NNDEPLOY_LIB_TYPE} ${PLUGIN_SOURCE} ${PLUGIN_OBJECT})

# # DIRECTORY
set_property(TARGET ${PLUGIN_BINARY} PROPERTY FOLDER ${NNDEPLOY_PLUGIN_DIRECTORY})

# # DEPEND_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_DEPEND_LIBRARY})

# # SYSTEM_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_SYSTEM_LIBRARY})

# # THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_THIRD_PARTY_LIBRARY})

# # NNDEPLOY_FRAMEWORK_BINARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_FRAMEWORK_BINARY})
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_preprocess)
target_link_libraries(${PLUGIN_BINARY} nndeploy_plugin_infer)

# # NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY
target_link_libraries(${PLUGIN_BINARY} ${NNDEPLOY_PLUGIN_THIRD_PARTY_LIBRARY})

# # install
if(SYSTEM_Windows)
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/track DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
else()
  install(TARGETS ${PLUGIN_BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_LIB_PATH})
  install(DIRECTORY ${PLUGIN_ROOT_PATH}/include/nndeploy/track DESTINATION ${NNDEPLOY_INSTALL_INCLUDE_PATH}/nndeploy)
endif()

# appedn list
set(NNDEPLOY_PLUGIN_LIST ${NNDEPLOY_PLUGIN_LIST} ${PLUGIN_BINARY})

# unset
unset(PLUGIN_SOURCE)
unset(PLUGIN_OBJECT)
unset(PLUGIN_BINARY)
