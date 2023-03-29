# --------------------------------------------------------------------
# Template custom cmake config for compiling
#
# This file is used to override the build sets in build.
# If you want to change the config, please use the following
# steps. Assume you are off the root directory. First copy the this
# file so that any local changes will be ignored by git
#
# $ mkdir build
# $ cp cmake/config_nntask.cmake build
#
# Next modify the according entries, and then compile by
#
# $ cd build
# $ cmake ..
#
# Then build in parallel with 8 threads
#
# $ make -j8
# --------------------------------------------------------------------
# nntask
set(NNTASK_ENABLE ON)
## test
set(NNTASK_ENABLE_TEST OFF)
## demo
set(NNTASK_ENABLE_DEMO ON)

## always
set(NNTASK_ENABLE_ALWAYS ON) 