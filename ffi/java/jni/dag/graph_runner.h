#ifndef _NNDEPLOY_FFI_JAVA_JNI_DAG_GRAPH_RUNNER_H_
#define _NNDEPLOY_FFI_JAVA_JNI_DAG_GRAPH_RUNNER_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

// 创建GraphRunner实例
JNIEXPORT jlong JNICALL
Java_com_nndeploy_dag_GraphRunner_createGraphRunner(JNIEnv* env, jobject thiz);

// 销毁GraphRunner实例
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_destroyGraphRunner(
    JNIEnv* env, jobject thiz, jlong handle);

// 运行图计算
JNIEXPORT jobject JNICALL Java_com_nndeploy_dag_GraphRunner_run(
    JNIEnv* env, jobject thiz, jlong handle, jstring graph_json_str,
    jstring name, jstring task_id);

// 设置是否为JSON文件
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setJsonFile(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_json_file);

// 设置是否转储
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setDump(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_dump);

// 设置是否启用时间性能分析
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setTimeProfile(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_time_profile);

// 设置是否启用调试
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setDebug(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_debug);

// 设置并行类型
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setParallelType(
    JNIEnv* env, jobject thiz, jlong handle, jint parallel_type);

// 设置循环最大标志
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setLoopMaxFlag(
    JNIEnv* env, jobject thiz, jlong handle, jboolean is_loop_max_flag);

// 设置节点值
JNIEXPORT void JNICALL Java_com_nndeploy_dag_GraphRunner_setNodeValue(
    JNIEnv* env, jobject thiz, jlong handle, jstring node_name, jstring key,
    jstring value);

#ifdef __cplusplus
}
#endif

#endif  // _NNDEPLOY_FFI_JAVA_JNI_DAG_GRAPH_RUNNER_H_
