package com.nndeploy.dag;

import java.util.List;
import java.util.Map;

/**
 * GraphRunner执行结果类
 * 
 * 封装了图执行的结果信息，包括状态、性能分析数据和结果数据
 */
public class GraphRunnerResult {
    
    /**
     * 状态码，0表示成功，其他值表示失败
     */
    public int statusCode;
    
    /**
     * 状态消息，描述执行状态的详细信息
     */
    public String statusMessage;
    
    /**
     * 时间性能分析数据
     * Key: 节点名称或操作名称
     * Value: 执行时间（毫秒）
     */
    public Map<String, Float> timeProfilerMap;
    
    /**
     * 执行结果数据列表
     * 包含图执行后的输出结果
     */
    public List<Object> results;
    
    /**
     * 默认构造函数
     */
    public GraphRunnerResult() {
        this.statusCode = -1;
        this.statusMessage = "";
        this.timeProfilerMap = new java.util.HashMap<>();
        this.results = new java.util.ArrayList<>();
    }
    
    /**
     * 检查执行是否成功
     * 
     * @return true表示成功，false表示失败
     */
    public boolean isSuccess() {
        return statusCode == 0;
    }
    
    /**
     * 获取总执行时间
     * 
     * @return 总执行时间（毫秒），如果没有性能分析数据则返回-1
     */
    public float getTotalTime() {
        if (timeProfilerMap == null || timeProfilerMap.isEmpty()) {
            return -1.0f;
        }
        
        float total = 0.0f;
        for (Float time : timeProfilerMap.values()) {
            if (time != null) {
                total += time;
            }
        }
        return total;
    }
    
    /**
     * 获取指定节点的执行时间
     * 
     * @param nodeName 节点名称
     * @return 执行时间（毫秒），如果节点不存在则返回-1
     */
    public float getNodeTime(String nodeName) {
        if (timeProfilerMap == null || nodeName == null) {
            return -1.0f;
        }
        
        Float time = timeProfilerMap.get(nodeName);
        return time != null ? time : -1.0f;
    }
    
    /**
     * 获取结果数量
     * 
     * @return 结果数量
     */
    public int getResultCount() {
        return results != null ? results.size() : 0;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("GraphRunnerResult{");
        sb.append("statusCode=").append(statusCode);
        sb.append(", statusMessage='").append(statusMessage).append('\'');
        sb.append(", totalTime=").append(getTotalTime()).append("ms");
        sb.append(", resultCount=").append(getResultCount());
        sb.append('}');
        return sb.toString();
    }
}
