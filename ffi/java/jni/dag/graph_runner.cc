
#include "nndeploy/dag/graph_runner.h"

#include <jni.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "nndeploy/base/common.h"
#include "nndeploy/base/status.h"

extern "C" {

// 创建GraphRunner实例
JNIEXPORT jlong JNICALL
Java_com_nndeploy_dag_GraphRunner_createGraphRunner(JNIEnv* env, jobject thiz) {
  nndeploy::dag::GraphRunner* runner = new nndeploy::dag::GraphRunner();
  return reinterpret_cast<jlong>(runner);
}

// 销毁GraphRunner实例
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_destroyGraphRunner(
    JNIEnv* env, jobject thiz, jlong handle) {
  if (handle != 0) {
    nndeploy::dag::GraphRunner* runner =
        reinterpret_cast<nndeploy::dag::GraphRunner*>(handle);
    delete runner;
  }
}
// 运行图计算
JNIEXPORT jboolean JNICALL Java_com_nndeploy_dag_GraphRunner_run(
    JNIEnv* env, jobject thiz, jlong handle, jstring graph_json_str,
    jstring name, jstring task_id) {
  if (handle == 0) {
    return JNI_FALSE;
  }

  nndeploy::dag::GraphRunner* runner =
      reinterpret_cast<nndeploy::dag::GraphRunner*>(handle);

  // 转换Java字符串为C++字符串
  const char* json_str = env->GetStringUTFChars(graph_json_str, nullptr);
  const char* name_str = env->GetStringUTFChars(name, nullptr);
  const char* task_id_str = env->GetStringUTFChars(task_id, nullptr);

  std::string cpp_json_str(json_str);
  std::string cpp_name(name_str);
  std::string cpp_task_id(task_id_str);

  // 执行运行
  auto result = runner->run(cpp_json_str, cpp_name, cpp_task_id);

  // 释放Java字符串
  env->ReleaseStringUTFChars(graph_json_str, json_str);
  env->ReleaseStringUTFChars(name, name_str);
  env->ReleaseStringUTFChars(task_id, task_id_str);

  // 返回运行结果的布尔值
  // return result.isSuccess() ? JNI_TRUE : JNI_FALSE;
  return JNI_TRUE;
}

// 设置是否为JSON文件
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setJsonFile(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_json_file) {
  if (handle != 0) {
    nndeploy::dag::GraphRunner* runner =
        reinterpret_cast<nndeploy::dag::GraphRunner*>(handle);
    runner->set_json_file(static_cast<bool>(is_json_file));
  }
}

// 设置是否转储
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setDump(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_dump) {
  if (handle != 0) {
    nndeploy::dag::GraphRunner* runner =
        reinterpret_cast<nndeploy::dag::GraphRunner*>(handle);
    runner->set_dump(static_cast<bool>(is_dump));
  }
}

// 设置是否启用时间性能分析
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setTimeProfile(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_time_profile) {
  if (handle != 0) {
    nndeploy::dag::GraphRunner* runner =
        reinterpret_cast<nndeploy::dag::GraphRunner*>(handle);
    
    // 使用Android Log打印设置时间性能分析的信息
    __android_log_print(ANDROID_LOG_INFO, "GraphRunner", "设置时间性能分析: %s", 
                       is_time_profile ? "启用" : "禁用");
    
    runner->set_time_profile(static_cast<bool>(is_time_profile));
  } else {
    __android_log_print(ANDROID_LOG_ERROR, "GraphRunner", 
                       "错误: GraphRunner句柄为空，无法设置时间性能分析");
  }
}

// 设置是否启用调试
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setDebug(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_debug) {
  if (handle != 0) {
    nndeploy::dag::GraphRunner* runner =
        reinterpret_cast<nndeploy::dag::GraphRunner*>(handle);
    runner->set_debug(static_cast<bool>(is_debug));
  }
}

// 设置并行类型
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setParallelType(
    JNIEnv* env, jobject thiz, jlong handle, jint parallel_type) {
  if (handle != 0) {
    nndeploy::dag::GraphRunner* runner =
        reinterpret_cast<nndeploy::dag::GraphRunner*>(handle);
    runner->set_parallel_type(
        static_cast<nndeploy::base::ParallelType>(parallel_type));
  }
}

// 设置循环最大标志
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setLoopMaxFlag(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_loop_max_flag) {
  if (handle != 0) {
    nndeploy::dag::GraphRunner* runner =
        reinterpret_cast<nndeploy::dag::GraphRunner*>(handle);
    runner->set_loop_max_flag(static_cast<bool>(is_loop_max_flag));
  }
}

// 设置节点值
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setNodeValue(
    JNIEnv* env, jobject thiz, jlong handle, jstring node_name, jstring key,
    jstring value) {
  if (handle != 0) {
    nndeploy::dag::GraphRunner* runner =
        reinterpret_cast<nndeploy::dag::GraphRunner*>(handle);

    // 转换Java字符串为C++字符串
    const char* node_name_str = env->GetStringUTFChars(node_name, nullptr);
    const char* key_str = env->GetStringUTFChars(key, nullptr);
    const char* value_str = env->GetStringUTFChars(value, nullptr);
    
    std::string cpp_node_name(node_name_str);
    std::string cpp_key(key_str);
    std::string cpp_value(value_str);

    // 设置节点值
    runner->set_node_value(cpp_node_name, cpp_key, cpp_value);

    // 释放Java字符串
    env->ReleaseStringUTFChars(node_name, node_name_str);
    env->ReleaseStringUTFChars(key, key_str);
    env->ReleaseStringUTFChars(value, value_str);
  }
}

}  // extern "C"
