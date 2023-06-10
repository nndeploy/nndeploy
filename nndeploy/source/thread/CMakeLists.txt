# set
set(SOURCE)
set(OBJECT)
set(BINARY nndeploy_thread)
set(DIRECTORY nndeploy)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

include_directories(${ROOT_PATH})

# SOURCE
file(GLOB_RECURSE SOURCE
  "${ROOT_PATH}/nndeploy/source/thread/*.h"
  "${ROOT_PATH}/nndeploy/source/thread/*.hpp"
  "${ROOT_PATH}/nndeploy/source/thread/*.cc"
  )

# OBJECT

# BINARY
add_library(${BINARY} ${NNDEPLOY_LIB_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY
set_property(TARGET ${BINARY} PROPERTY FOLDER ${DIRECTORY})

# DEPEND_LIBRARY
target_link_libraries(${BINARY} ${DEPEND_LIBRARY})

# SYSTEM_LIBRARY
target_link_libraries(${BINARY} ${SYSTEM_LIBRARY})

# THIRD_PARTY_LIBRARY
target_link_libraries(${BINARY} ${THIRD_PARTY_LIBRARY}) 

# install
install(TARGETS ${BINARY} ${NNDEPLOY_INSTALL_TYPE} DESTINATION ${NNDEPLOY_INSTALL_PATH})

# unset
unset(SOURCE)
unset(OBJECT)
unset(BINARY)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)
