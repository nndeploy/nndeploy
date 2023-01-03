cmake_minimum_required(VERSION 3.10)

# set
set(SOURCE)
set(OBJECT)
set(TARGET nndeploy_base)
set(DIRECTORY nndeploy)
set(DEPEND_LIBRARY)
set(SYSTEM_LIBRARY)
set(THIRD_PARTY_LIBRARY)

include_directories(ROOT_PATH)

# SOURCE
file()

# OBJECT

# TARGET
add_library(${TARGET} ${NNKIT_BUILD_TYPE} ${SOURCE} ${OBJECT})

# DIRECTORY

# DEPEND_LIBRARY

# SYSTEM_LIBRARY

# THIRD_PARTY_LIBRARY

# install

# unset
unset(SOURCE)
unset(OBJECT)
unset(TARGET)
unset(DIRECTORY)
unset(DEPEND_LIBRARY)
unset(SYSTEM_LIBRARY)
unset(THIRD_PARTY_LIBRARY)

