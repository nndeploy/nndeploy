macro(__nndeploy_option variable description value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value} CACHE STRING ${description})
  endif()
endmacro()

set(NNDEPLOY_ALL_OPTIONS)

#######################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   nndeploy_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
macro(nndeploy_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  list(APPEND NNDEPLOY_ALL_OPTIONS ${variable})
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if("${__condition}" STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if("${__value}" MATCHES ";")
      if(${__value})
        __nndeploy_option(${variable} "${description}" ON)
      else()
        __nndeploy_option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        __nndeploy_option(${variable} "${description}" ON)
      else()
        __nndeploy_option(${variable} "${description}" OFF)
      endif()
    else()
      __nndeploy_option(${variable} "${description}" "${__value}")
    endif()
  else()
    unset(${variable} CACHE)
  endif()
endmacro()

function(assign_source_group group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${group}\\${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

function(nndeploy_micro_add_copy_file var src dest)
    get_filename_component(basename "${src}" NAME)
    get_filename_component(dest_parent_dir "${dest}" DIRECTORY)
    add_custom_command(
        OUTPUT "${dest}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${src}" "${dest}"
        DEPENDS "${src}")
    list(APPEND "${var}" "${dest}")
    set("${var}" "${${var}}" PARENT_SCOPE)
endfunction(nndeploy_micro_add_copy_file)

set(MICRONNDEPLOY_TEMPLATE_PROJECTS "${CMAKE_CURRENT_BINARY_DIR}/micronndeploy_template_projects")

# From cmake documentation:
# True if the constant is 1, ON, YES, TRUE, Y, or a non-zero number.
# False if the constant is 0, OFF, NO, FALSE, N, IGNORE, NOTFOUND, the empty string, or ends in the suffix -NOTFOUND.
# Named boolean constants are case-insensitive.
#
# While this regex does contain a check for an empty string that check does not work
# cmake's regex is weak
set(IS_FALSE_PATTERN "^[Oo][Ff][Ff]$|^0$|^[Ff][Aa][Ll][Ss][Ee]$|^[Nn][Oo]$|^[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|.*-[Nn][Oo][Tt][Ff][Oo][Uu][Nn][Dd]$|^$")
set(IS_TRUE_PATTERN "^[Oo][Nn]$|^[1-9][0-9]*$|^[Tt][Rr][Uu][Ee]$|^[Yy][Ee][Ss]$|^[Yy]$")

# Custom file() macro that automatically uses CONFIGURE_DEPENDS if cmake
# supports it. CONFIGURE_DEPENDS scans the globbed directories on each build to
# check if any files have been added/removed. This has a small build overhead,
# but ensures that you don't manually have to rerun cmake if files were added.
# The macro should be used like so:
# nndeploy_file_glob(GLOB VARIABLE_NAME dir/*.cc dir/*c)
# or
# nndeploy_file_glob(GLOB_RECURSE VARIABLE_NAME dir/*/*.cc)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  macro(nndeploy_file_glob glob variable)
    file(${glob} ${variable} CONFIGURE_DEPENDS ${ARGN})
  endmacro()
else()
  macro(nndeploy_file_glob)
    file(${glob} ${variable} ${ARGN})
  endmacro()
endif()
