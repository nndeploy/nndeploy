# nndeploy Java FFI CMake配置文件

cmake_minimum_required(VERSION 3.12)

# 项目信息
project(nndeploy_java_ffi)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找JNI
find_package(JNI REQUIRED)
if(NOT JNI_FOUND)
    message(FATAL_ERROR "JNI not found! Please install JDK and set JAVA_HOME")
endif()

# 查找Java编译器
find_package(Java REQUIRED COMPONENTS Development)
if(NOT Java_FOUND)
    message(FATAL_ERROR "Java Development Kit not found! Please install JDK")
endif()

# 包含JNI头文件
include_directories(${JNI_INCLUDE_DIRS})

# 设置输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Java源文件目录
set(JAVA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nndeploy)
set(JAVA_OUTPUT_DIR ${CMAKE_BINARY_DIR}/java_classes)

# 创建Java输出目录
file(MAKE_DIRECTORY ${JAVA_OUTPUT_DIR})

# 收集Java源文件
file(GLOB_RECURSE JAVA_SOURCES ${JAVA_SOURCE_DIR}/*.java)

# 编译Java类
add_custom_command(
    OUTPUT ${JAVA_OUTPUT_DIR}/com/nndeploy/dag/GraphRunner.class
    COMMAND ${Java_JAVAC_EXECUTABLE} 
        -d ${JAVA_OUTPUT_DIR}
        -cp ${JAVA_OUTPUT_DIR}
        ${JAVA_SOURCES}
    DEPENDS ${JAVA_SOURCES}
    COMMENT "编译Java源文件"
)

# 创建Java编译目标
add_custom_target(compile_java_classes ALL
    DEPENDS ${JAVA_OUTPUT_DIR}/com/nndeploy/dag/GraphRunner.class
)

# 生成JNI头文件（如果需要）
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/jni/com_nndeploy_dag_GraphRunner.h
    COMMAND ${Java_JAVAH_EXECUTABLE}
        -jni
        -cp ${JAVA_OUTPUT_DIR}
        -d ${CMAKE_CURRENT_SOURCE_DIR}/jni
        com.nndeploy.dag.GraphRunner
    DEPENDS compile_java_classes
    COMMENT "生成JNI头文件"
)

# JNI头文件生成目标
add_custom_target(generate_jni_headers
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/jni/com_nndeploy_dag_GraphRunner.h
)

# 收集JNI源文件
file(GLOB_RECURSE JNI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/jni/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/jni/*.cpp
)

file(GLOB_RECURSE JNI_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/jni/*.h
)

# 创建JNI共享库
add_library(nndeploy_jni SHARED ${JNI_SOURCES})

# 设置JNI库属性
set_target_properties(nndeploy_jni PROPERTIES
    VERSION ${NNDEPLOY_VERSION}
    SOVERSION ${NNDEPLOY_VERSION_MAJOR}
    PREFIX ""  # 移除lib前缀
)

# Windows特定设置
if(WIN32)
    set_target_properties(nndeploy_jni PROPERTIES
        SUFFIX ".dll"
    )
endif()

# 包含目录
target_include_directories(nndeploy_jni PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/jni
    ${ROOT_PATH}/framework/include
    ${ROOT_PATH}/framework/source
)

# 链接nndeploy核心库
target_link_libraries(nndeploy_jni
    nndeploy_framework
    ${JNI_LIBRARIES}
)

# 编译定义
target_compile_definitions(nndeploy_jni PRIVATE
    VERSION_INFO="${NNDEPLOY_VERSION}"
)

# 编译选项
if(MSVC)
    target_compile_options(nndeploy_jni PRIVATE /W3)
else()
    target_compile_options(nndeploy_jni PRIVATE -Wall -Wextra)
endif()

# 创建JAR文件
set(JAR_FILE ${CMAKE_BINARY_DIR}/nndeploy-java-${NNDEPLOY_VERSION}.jar)

add_custom_command(
    OUTPUT ${JAR_FILE}
    COMMAND ${Java_JAR_EXECUTABLE} 
        cf ${JAR_FILE}
        -C ${JAVA_OUTPUT_DIR} .
    DEPENDS compile_java_classes
    COMMENT "创建JAR文件: ${JAR_FILE}"
)

add_custom_target(create_jar ALL
    DEPENDS ${JAR_FILE}
)

# 安装规则
install(TARGETS nndeploy_jni
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${JAR_FILE}
    DESTINATION lib
)

install(FILES ${JAVA_SOURCES}
    DESTINATION include/java
)

# 测试目标
if(ENABLE_TESTING)
    # 创建Java测试运行脚本
    set(TEST_SCRIPT ${CMAKE_BINARY_DIR}/run_java_tests.sh)
    
    file(WRITE ${TEST_SCRIPT}
"#!/bin/bash
export LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$DYLD_LIBRARY_PATH
cd ${CMAKE_BINARY_DIR}
java -Djava.library.path=${CMAKE_LIBRARY_OUTPUT_DIRECTORY} \\
     -cp ${JAVA_OUTPUT_DIR} \\
     com.nndeploy.dag.GraphRunnerExample
")
    
    # 设置脚本可执行权限
    file(COPY ${TEST_SCRIPT}
        DESTINATION ${CMAKE_BINARY_DIR}
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE
    )
    
    # 添加测试
    add_test(
        NAME java_graphrunner_test
        COMMAND ${TEST_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    
    # 设置测试环境
    set_tests_properties(java_graphrunner_test PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY};DYLD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )
endif()

# 显示构建信息
message(STATUS "=== nndeploy Java FFI 构建配置 ===")
message(STATUS "Java版本: ${Java_VERSION}")
message(STATUS "JNI包含目录: ${JNI_INCLUDE_DIRS}")
message(STATUS "JNI库: ${JNI_LIBRARIES}")
message(STATUS "Java源文件目录: ${JAVA_SOURCE_DIR}")
message(STATUS "Java输出目录: ${JAVA_OUTPUT_DIR}")
message(STATUS "JNI库输出: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/nndeploy_jni")
message(STATUS "JAR文件: ${JAR_FILE}")
message(STATUS "==============================")

# 帮助信息
add_custom_target(help_java
    COMMAND ${CMAKE_COMMAND} -E echo "nndeploy Java FFI 构建目标:"
    COMMAND ${CMAKE_COMMAND} -E echo "  compile_java_classes - 编译Java类文件"
    COMMAND ${CMAKE_COMMAND} -E echo "  generate_jni_headers - 生成JNI头文件"
    COMMAND ${CMAKE_COMMAND} -E echo "  nndeploy_jni        - 构建JNI共享库"
    COMMAND ${CMAKE_COMMAND} -E echo "  create_jar          - 创建JAR文件"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "运行示例:"
    COMMAND ${CMAKE_COMMAND} -E echo "  ${TEST_SCRIPT}"
)
