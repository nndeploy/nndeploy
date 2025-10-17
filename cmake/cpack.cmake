# cpack.cmake

# Copy all contents directly from install directory
install(DIRECTORY ${NNDEPLOY_INSTALL_PATH}/
        DESTINATION .
        USE_SOURCE_PERMISSIONS
        COMPONENT Complete)

set(CPACK_PACKAGE_NAME "nndeploy")
set(CPACK_PACKAGE_VERSION "${NNDEPLOY_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Workflow-based Multi-platform AI Deployment Tool")
set(CPACK_PACKAGE_VENDOR "nndeploy Team")
set(CPACK_PACKAGE_CONTACT "595961667@qq.com")

# Use the defined installation path as the packaging source
set(CPACK_PACKAGE_INSTALL_DIRECTORY "nndeploy-${NNDEPLOY_VERSION}")

# Choose packaging format based on platform
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_NSIS_DISPLAY_NAME "NNDEPLOY ${NNDEPLOY_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "NNDEPLOY")
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin" OR CMAKE_SYSTEM_NAME MATCHES "iOS")
    set(CPACK_GENERATOR "TGZ")
    set(CPACK_DMG_VOLUME_NAME "NNDEPLOY-${NNDEPLOY_VERSION}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Android")
    set(CPACK_GENERATOR "TGZ")
else()
    # Linux system - Check if rpmbuild exists
    find_program(RPMBUILD_EXECUTABLE rpmbuild)
    if(RPMBUILD_EXECUTABLE)
        set(CPACK_GENERATOR "TGZ;DEB;RPM")
    else()
        set(CPACK_GENERATOR "TGZ;DEB")
        message(STATUS "rpmbuild not found, skipping RPM package generation")
    endif()
    
    # DEB package configuration
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "nndeploy Team")
    set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    
    # RPM package configuration (only when rpmbuild exists)
    if(RPMBUILD_EXECUTABLE)
        set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
        set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
    endif()
endif()

# Package filename format - consistent with your installation path
set(CPACK_PACKAGE_FILE_NAME 
    "nndeploy_${NNDEPLOY_VERSION}_${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}_${CMAKE_BUILD_TYPE}_${CMAKE_CXX_COMPILER_ID}")

# Include CPack module
include(CPack)