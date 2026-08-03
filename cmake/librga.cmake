include(ExternalProject)

# ======================== librga (Rockchip RGA 2D acceleration) ========================
#
# Usage:
#   -DENABLE_NNDEPLOY_RKRGA=OFF                        # Disabled (default)
#   -DENABLE_NNDEPLOY_RKRGA=ON                         # Use system-installed librga
#   -DENABLE_NNDEPLOY_RKRGA=path/to/librga             # Use bundled librga
#
# On Rockchip devices, RKNN (NPU) and RGA (2D hardware) are independent IP blocks
# but are commonly used together: RGA pre-processes images → RKNN inference → RGA post-processes.
# When RKNN is enabled on aarch64 and third_party/librga exists, librga is auto-enabled.
# To explicitly disable librga on Rockchip, pass -DENABLE_NNDEPLOY_RKRGA=OFF AFTER
# the auto-detection or remove third_party/librga.
# ==========================================================================================

# --- Soft association: auto-detect librga when RKNN is enabled on Rockchip ---
if (ENABLE_NNDEPLOY_RKRGA STREQUAL "OFF")
  if (DEFINED ENABLE_NNDEPLOY_INFERENCE_RKNN
      AND NOT ENABLE_NNDEPLOY_INFERENCE_RKNN STREQUAL "OFF"
      AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64"
      AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/librga")
    message(STATUS "librga: RKNN enabled on aarch64, auto-enabling librga from third_party/")
    set(ENABLE_NNDEPLOY_RKRGA "${CMAKE_SOURCE_DIR}/third_party/librga")
  endif()
endif()

# --- Option handling: OFF / ON (system) / path (bundled) ---
if (ENABLE_NNDEPLOY_RKRGA STREQUAL "OFF")
  # Explicitly disabled — nothing to do.
elseif (ENABLE_NNDEPLOY_RKRGA STREQUAL "ON")
  # ON mode: find system-installed librga
  find_package(librga QUIET)
  if(librga_FOUND)
    message(STATUS "Found system librga")
  else()
    message(FATAL_ERROR "librga not found. Please set ENABLE_NNDEPLOY_RKRGA to the path of librga, or install librga system-wide.")
  endif()
else()
  # Path mode: use specified librga directory
  if(IS_ABSOLUTE ${ENABLE_NNDEPLOY_RKRGA})
    set(LIBRGA_ROOT_PATH ${ENABLE_NNDEPLOY_RKRGA})
    message(STATUS "librga: using absolute path: ${LIBRGA_ROOT_PATH}")
  else()
    set(LIBRGA_ROOT_PATH ${CMAKE_SOURCE_DIR}/${ENABLE_NNDEPLOY_RKRGA})
    message(STATUS "librga: using relative path: ${LIBRGA_ROOT_PATH}")
  endif()

  include_directories(${LIBRGA_ROOT_PATH}/include)
  set(LIB_PATH ${LIBRGA_ROOT_PATH}/${NNDEPLOY_THIRD_PARTY_LIBRARY_PATH_SUFFIX})
  set(LIBS "rga")

  foreach(LIB ${LIBS})
    set(LIB_NAME ${NNDEPLOY_LIB_PREFIX}${LIB}${NNDEPLOY_LIB_SUFFIX})
    set(FULL_LIB_NAME ${LIB_PATH}/${LIB_NAME})
    # Fallback to static (.a) when shared (.so) library not found
    if(NOT EXISTS "${FULL_LIB_NAME}")
      set(STATIC_LIB_NAME "${LIB_PATH}/lib${LIB}.a")
      if(EXISTS "${STATIC_LIB_NAME}")
        message(STATUS "librga: shared not found, using static: lib${LIB}.a")
        set(FULL_LIB_NAME "${STATIC_LIB_NAME}")
      else()
        message(WARNING "librga library not found: ${FULL_LIB_NAME}")
      endif()
    endif()
    if(IS_SYMLINK ${FULL_LIB_NAME})
      get_filename_component(REAL_LIB_NAME ${FULL_LIB_NAME} REALPATH)
      set(FULL_LIB_NAME ${REAL_LIB_NAME})
    endif()
    set(NNDEPLOY_THIRD_PARTY_LIBRARY ${NNDEPLOY_THIRD_PARTY_LIBRARY} ${FULL_LIB_NAME})
  endforeach()

  if(SYSTEM_Windows)
    set(BIN_PATH ${LIBRGA_ROOT_PATH}/bin)
    if(EXISTS "${BIN_PATH}")
      link_directories(${BIN_PATH})
      file(GLOB_RECURSE SET_BIN_PATH ${BIN_PATH}/*.dll)
      foreach(DLL_PATH ${SET_BIN_PATH})
        file(COPY ${DLL_PATH} DESTINATION ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_BUILD_TYPE})
      endforeach()
    endif()
  endif()

  install(DIRECTORY ${LIBRGA_ROOT_PATH} DESTINATION ${NNDEPLOY_INSTALL_THIRD_PARTY_PATH})
endif()
