package com.nndeploy.dag;

/**
 * GraphRunner - nndeploy图执行器的Java包装类
 * 
 * 提供了图构建、配置和执行的完整接口，支持性能分析、调试等功能。
 * 这是nndeploy框架在Java端的主要入口类。
 * 
 * 使用示例：
 * <pre>
 * GraphRunner runner = new GraphRunner();
 * runner.setTimeProfile(true);
 * runner.setDebug(false);
 * 
 * GraphRunnerResult result = runner.run(jsonGraph, "MyGraph", "task_001");
 * if (result.isSuccess()) {
 *     System.out.println("执行成功，耗时: " + result.getTotalTime() + "ms");
 * } else {
 *     System.err.println("执行失败: " + result.statusMessage);
 * }
 * 
 * runner.close(); // 释放资源
 * </pre>
 * 
 * @author nndeploy team
 * @version 1.0
 */
public class GraphRunner implements AutoCloseable {
    
    // 加载native库
    static {
        try {
            System.loadLibrary("nndeploy_jni");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("无法加载nndeploy_jni库: " + e.getMessage());
            throw e;
        }
    }
    
    /**
     * 并行类型枚举
     */
    public static class ParallelType {
        public static final int NONE = 0;           // 无并行
        public static final int SEQUENCE = 1;       // 顺序执行
        public static final int PIPELINE = 2;       // 流水线并行
        public static final int TASK = 3;          // 任务并行
    }
    
    // Native对象句柄
    private long nativeHandle;
    
    // 是否已初始化
    private boolean initialized;
    
    // 配置参数
    private boolean isJsonFile = true;
    private boolean isDump = true;
    private boolean isTimeProfile = true;
    private boolean isDebug = false;
    private int parallelType = ParallelType.NONE;
    private boolean isLoopMaxFlag = true;
    
    /**
     * 默认构造函数
     * 创建一个新的GraphRunner实例
     * 
     * @throws RuntimeException 如果创建失败
     */
    public GraphRunner() {
        nativeHandle = createGraphRunner();
        if (nativeHandle == 0) {
            throw new RuntimeException("创建GraphRunner失败");
        }
        initialized = true;
    }
    
    /**
     * 运行图计算
     * 
     * @param graphJsonStr 图的JSON字符串描述
     * @param name 图的名称
     * @param taskId 任务ID，用于跟踪和调试
     * @return 执行结果，包含状态、性能数据和输出结果
     * @throws IllegalStateException 如果GraphRunner已关闭
     * @throws IllegalArgumentException 如果参数为空
     */
    public GraphRunnerResult run(String graphJsonStr, String name, String taskId) {
        checkInitialized();
        
        if (graphJsonStr == null || graphJsonStr.isEmpty()) {
            throw new IllegalArgumentException("图JSON字符串不能为空");
        }
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("图名称不能为空");
        }
        if (taskId == null || taskId.isEmpty()) {
            throw new IllegalArgumentException("任务ID不能为空");
        }
        
        return run(nativeHandle, graphJsonStr, name, taskId);
    }
    
    /**
     * 设置是否将输入视为JSON文件
     * 
     * @param isJsonFile true表示输入是JSON文件路径，false表示输入是JSON字符串
     */
    public void setJsonFile(boolean isJsonFile) {
        checkInitialized();
        this.isJsonFile = isJsonFile;
        setJsonFile(nativeHandle, isJsonFile);
    }
    
    /**
     * 获取是否将输入视为JSON文件
     * 
     * @return true表示输入是JSON文件路径，false表示输入是JSON字符串
     */
    public boolean isJsonFile() {
        return isJsonFile;
    }
    
    /**
     * 设置是否启用转储功能
     * 转储功能会输出详细的执行信息，用于调试
     * 
     * @param isDump true启用转储，false禁用
     */
    public void setDump(boolean isDump) {
        checkInitialized();
        this.isDump = isDump;
        setDump(nativeHandle, isDump);
    }
    
    /**
     * 获取是否启用转储功能
     * 
     * @return true表示启用转储，false表示禁用
     */
    public boolean isDump() {
        return isDump;
    }
    
    /**
     * 设置是否启用时间性能分析
     * 
     * @param isTimeProfile true启用性能分析，false禁用
     */
    public void setTimeProfile(boolean isTimeProfile) {
        checkInitialized();
        this.isTimeProfile = isTimeProfile;
        setTimeProfile(nativeHandle, isTimeProfile);
    }
    
    /**
     * 获取是否启用时间性能分析
     * 
     * @return true表示启用性能分析，false表示禁用
     */
    public boolean isTimeProfile() {
        return isTimeProfile;
    }
    
    /**
     * 设置是否启用调试模式
     * 
     * @param isDebug true启用调试模式，false禁用
     */
    public void setDebug(boolean isDebug) {
        checkInitialized();
        this.isDebug = isDebug;
        setDebug(nativeHandle, isDebug);
    }
    
    /**
     * 获取是否启用调试模式
     * 
     * @return true表示启用调试模式，false表示禁用
     */
    public boolean isDebug() {
        return isDebug;
    }
    
    /**
     * 设置并行类型
     * 
     * @param parallelType 并行类型，使用ParallelType常量
     */
    public void setParallelType(int parallelType) {
        checkInitialized();
        this.parallelType = parallelType;
        setParallelType(nativeHandle, parallelType);
    }
    
    /**
     * 获取并行类型
     * 
     * @return 并行类型
     */
    public int getParallelType() {
        return parallelType;
    }
    
    /**
     * 设置循环最大标志
     * 
     * @param isLoopMaxFlag true启用循环最大标志，false禁用
     */
    public void setLoopMaxFlag(boolean isLoopMaxFlag) {
        checkInitialized();
        this.isLoopMaxFlag = isLoopMaxFlag;
        setLoopMaxFlag(nativeHandle, isLoopMaxFlag);
    }
    
    /**
     * 获取循环最大标志
     * 
     * @return true表示启用循环最大标志，false表示禁用
     */
    public boolean isLoopMaxFlag() {
        return isLoopMaxFlag;
    }
    
    /**
     * 设置节点参数值
     * 
     * @param nodeName 节点名称
     * @param key 参数键
     * @param value 参数值
     * @throws IllegalArgumentException 如果参数为空
     */
    public void setNodeValue(String nodeName, String key, String value) {
        checkInitialized();
        
        if (nodeName == null || nodeName.isEmpty()) {
            throw new IllegalArgumentException("节点名称不能为空");
        }
        if (key == null || key.isEmpty()) {
            throw new IllegalArgumentException("参数键不能为空");
        }
        if (value == null) {
            value = ""; // 允许空值，但不允许null
        }
        
        setNodeValue(nativeHandle, nodeName, key, value);
    }
    
    /**
     * 检查GraphRunner是否已初始化
     * 
     * @throws IllegalStateException 如果已关闭
     */
    private void checkInitialized() {
        if (!initialized || nativeHandle == 0) {
            throw new IllegalStateException("GraphRunner已关闭");
        }
    }
    
    /**
     * 关闭GraphRunner并释放资源
     * 调用此方法后，GraphRunner将不能再使用
     */
    @Override
    public void close() {
        if (initialized && nativeHandle != 0) {
            destroyGraphRunner(nativeHandle);
            nativeHandle = 0;
            initialized = false;
        }
    }
    
    /**
     * 析构函数，确保资源被正确释放
     */
    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }
    
    /**
     * 获取GraphRunner的状态信息
     * 
     * @return 状态信息字符串
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("GraphRunner{");
        sb.append("initialized=").append(initialized);
        sb.append(", isJsonFile=").append(isJsonFile);
        sb.append(", isDump=").append(isDump);
        sb.append(", isTimeProfile=").append(isTimeProfile);
        sb.append(", isDebug=").append(isDebug);
        sb.append(", parallelType=").append(parallelType);
        sb.append(", isLoopMaxFlag=").append(isLoopMaxFlag);
        sb.append('}');
        return sb.toString();
    }
    
    // ================== Native方法声明 ==================
    
    /**
     * 创建native GraphRunner实例
     * 
     * @return native对象句柄
     */
    private static native long createGraphRunner();
    
    /**
     * 销毁native GraphRunner实例
     * 
     * @param handle native对象句柄
     */
    private static native void destroyGraphRunner(long handle);
    
    /**
     * 执行图计算
     * 
     * @param handle native对象句柄
     * @param graphJsonStr 图JSON字符串
     * @param name 图名称
     * @param taskId 任务ID
     * @return 执行结果
     */
    private static native GraphRunnerResult run(long handle, String graphJsonStr, String name, String taskId);
    
    /**
     * 设置是否为JSON文件
     * 
     * @param handle native对象句柄
     * @param isJsonFile 是否为JSON文件
     */
    private static native void setJsonFile(long handle, boolean isJsonFile);
    
    /**
     * 设置是否转储
     * 
     * @param handle native对象句柄
     * @param isDump 是否转储
     */
    private static native void setDump(long handle, boolean isDump);
    
    /**
     * 设置是否启用时间性能分析
     * 
     * @param handle native对象句柄
     * @param isTimeProfile 是否启用时间性能分析
     */
    private static native void setTimeProfile(long handle, boolean isTimeProfile);
    
    /**
     * 设置是否启用调试
     * 
     * @param handle native对象句柄
     * @param isDebug 是否启用调试
     */
    private static native void setDebug(long handle, boolean isDebug);
    
    /**
     * 设置并行类型
     * 
     * @param handle native对象句柄
     * @param parallelType 并行类型
     */
    private static native void setParallelType(long handle, int parallelType);
    
    /**
     * 设置循环最大标志
     * 
     * @param handle native对象句柄
     * @param isLoopMaxFlag 是否启用循环最大标志
     */
    private static native void setLoopMaxFlag(long handle, boolean isLoopMaxFlag);
    
    /**
     * 设置节点值
     * 
     * @param handle native对象句柄
     * @param nodeName 节点名称
     * @param key 参数键
     * @param value 参数值
     */
    private static native void setNodeValue(long handle, String nodeName, String key, String value);
}
