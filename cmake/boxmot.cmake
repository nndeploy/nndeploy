# cmake/boxmot.cmake
# Integrates boxmot C++ native trackers as static libraries.
#
# Resolution order:
#   1. third_party/boxmot/ (local clone) — preferred
#   2. System-installed boxmot (not supported yet)
#
# Provides:
#   BOXMOT_FOUND          - TRUE if boxmot source is available
#   BOXMOT_INCLUDE_DIRS   - include directories for boxmot headers
#   boxmot_tracker_base   - static lib: base tracker utilities + ReID
#   bytetrack_core        - static lib: ByteTrack tracker
#   botsort_core          - static lib: BotSort tracker
#   ocsort_core           - static lib: OcSort tracker
#   sfsort_core           - static lib: SfSort tracker
#   occluboost_core       - static lib: OccluBoost tracker

set(BOXMOT_ROOT "${PROJECT_SOURCE_DIR}/third_party/boxmot")

if(EXISTS "${BOXMOT_ROOT}/boxmot/native/cpp/trackers")
  set(BOXMOT_FOUND TRUE)

  # Include directories for each tracker's public headers
  set(BOXMOT_INCLUDE_DIRS
    "${BOXMOT_ROOT}/boxmot/native/cpp/trackers/base/include"
    "${BOXMOT_ROOT}/boxmot/native/cpp/trackers/bytetrack/include"
    "${BOXMOT_ROOT}/boxmot/native/cpp/trackers/botsort/include"
    "${BOXMOT_ROOT}/boxmot/native/cpp/trackers/ocsort/include"
    "${BOXMOT_ROOT}/boxmot/native/cpp/trackers/sfsort/include"
    "${BOXMOT_ROOT}/boxmot/native/cpp/trackers/occluboost/include"
  )

  message(STATUS "Found boxmot source: ${BOXMOT_ROOT}")

  # Skip if targets already exist (avoid duplicate add_subdirectory)
  if(NOT TARGET boxmot_tracker_base)
    # Disable boxmot's wheel install rules when used as a subproject
    set(BOXMOT_INSTALL_NATIVE OFF CACHE BOOL "" FORCE)

    # Disable boxmot's built-in ReID ONNX Runtime (nndeploy handles ReID)
    set(BOXMOT_REID_ONNXRUNTIME OFF CACHE BOOL "" FORCE)

    add_subdirectory("${BOXMOT_ROOT}/boxmot/native/cpp")
  endif()
else()
  message(FATAL_ERROR
    "boxmot not found at ${BOXMOT_ROOT}\n"
    "Run: cd third_party && git clone --depth 1 "
    "https://github.com/mikel-brostrom/boxmot.git boxmot")
endif()
