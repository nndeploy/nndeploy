package com.nndeploy.dag;

/**
 * GraphRunner使用示例
 * 
 * 展示了如何使用GraphRunner进行图计算，包括配置设置、执行和结果处理
 */
public class GraphRunnerExample {
    
    /**
     * 基本使用示例
     */
    public static void basicExample() {
        System.out.println("=== GraphRunner 基本使用示例 ===");
        
        // 示例图JSON（实际使用时应该是完整的图定义）
        String graphJson = """
        {
            "name": "SimpleGraph",
            "nodes": [
                {
                    "key": "Input",
                    "name": "input_node",
                    "inputs": [],
                    "outputs": ["data"]
                },
                {
                    "key": "Process", 
                    "name": "process_node",
                    "inputs": ["data"],
                    "outputs": ["result"]
                }
            ],
            "edges": [
                {
                    "name": "data",
                    "producers": ["input_node"],
                    "consumers": ["process_node"]
                },
                {
                    "name": "result", 
                    "producers": ["process_node"],
                    "consumers": []
                }
            ]
        }
        """;
        
        // 使用try-with-resources确保资源正确释放
        try (GraphRunner runner = new GraphRunner()) {
            // 配置GraphRunner
            runner.setTimeProfile(true);
            runner.setDebug(false);
            runner.setDump(true);
            runner.setParallelType(GraphRunner.ParallelType.PIPELINE);
            
            System.out.println("GraphRunner配置: " + runner.toString());
            
            // 执行图计算
            GraphRunnerResult result = runner.run(graphJson, "SimpleGraph", "task_001");
            
            // 处理结果
            if (result.isSuccess()) {
                System.out.println("✓ 执行成功!");
                System.out.println("总耗时: " + result.getTotalTime() + "ms");
                System.out.println("结果数量: " + result.getResultCount());
                
                // 打印性能分析数据
                if (result.timeProfilerMap != null && !result.timeProfilerMap.isEmpty()) {
                    System.out.println("\n性能分析数据:");
                    result.timeProfilerMap.forEach((nodeName, time) -> 
                        System.out.println("  " + nodeName + ": " + time + "ms")
                    );
                }
            } else {
                System.err.println("✗ 执行失败!");
                System.err.println("错误代码: " + result.statusCode);
                System.err.println("错误信息: " + result.statusMessage);
            }
            
        } catch (Exception e) {
            System.err.println("执行过程中发生异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 高级配置示例
     */
    public static void advancedExample() {
        System.out.println("\n=== GraphRunner 高级配置示例 ===");
        
        try (GraphRunner runner = new GraphRunner()) {
            // 高级配置
            runner.setJsonFile(false);  // 输入是JSON字符串而不是文件
            runner.setTimeProfile(true);
            runner.setDebug(true);
            runner.setDump(false);
            runner.setParallelType(GraphRunner.ParallelType.TASK);
            runner.setLoopMaxFlag(true);
            
            // 设置节点参数
            runner.setNodeValue("process_node", "batch_size", "32");
            runner.setNodeValue("process_node", "device", "gpu");
            runner.setNodeValue("process_node", "precision", "fp16");
            
            System.out.println("高级配置完成: " + runner.toString());
            
            // 模拟复杂图JSON
            String complexGraphJson = """
            {
                "name": "ComplexGraph",
                "description": "包含多个处理节点的复杂图",
                "nodes": [
                    {
                        "key": "DataLoader",
                        "name": "data_loader",
                        "params": {
                            "batch_size": 32,
                            "shuffle": true
                        }
                    },
                    {
                        "key": "Preprocessor", 
                        "name": "preprocessor",
                        "params": {
                            "normalize": true,
                            "resize": [224, 224]
                        }
                    },
                    {
                        "key": "Model",
                        "name": "inference_model", 
                        "params": {
                            "model_path": "/path/to/model.onnx",
                            "device": "gpu",
                            "precision": "fp16"
                        }
                    },
                    {
                        "key": "Postprocessor",
                        "name": "postprocessor",
                        "params": {
                            "threshold": 0.5,
                            "nms_threshold": 0.4
                        }
                    }
                ]
            }
            """;
            
            // 执行复杂图
            GraphRunnerResult result = runner.run(complexGraphJson, "ComplexGraph", "advanced_task_001");
            
            // 详细结果分析
            System.out.println("\n执行结果详情:");
            System.out.println("状态: " + (result.isSuccess() ? "成功" : "失败"));
            System.out.println("状态码: " + result.statusCode);
            System.out.println("状态消息: " + result.statusMessage);
            System.out.println("总执行时间: " + result.getTotalTime() + "ms");
            
            if (result.isSuccess()) {
                // 分析各节点性能
                System.out.println("\n各节点性能分析:");
                String[] nodeNames = {"data_loader", "preprocessor", "inference_model", "postprocessor"};
                for (String nodeName : nodeNames) {
                    float nodeTime = result.getNodeTime(nodeName);
                    if (nodeTime >= 0) {
                        System.out.println("  " + nodeName + ": " + nodeTime + "ms");
                    }
                }
                
                // 计算性能占比
                float totalTime = result.getTotalTime();
                if (totalTime > 0) {
                    System.out.println("\n性能占比分析:");
                    for (String nodeName : nodeNames) {
                        float nodeTime = result.getNodeTime(nodeName);
                        if (nodeTime >= 0) {
                            float percentage = (nodeTime / totalTime) * 100;
                            System.out.printf("  %s: %.1f%%\n", nodeName, percentage);
                        }
                    }
                }
            }
            
        } catch (Exception e) {
            System.err.println("高级示例执行失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 批量处理示例
     */
    public static void batchProcessingExample() {
        System.out.println("\n=== GraphRunner 批量处理示例 ===");
        
        try (GraphRunner runner = new GraphRunner()) {
            // 配置为批量处理模式
            runner.setTimeProfile(true);
            runner.setDebug(false);
            runner.setParallelType(GraphRunner.ParallelType.PIPELINE);
            
            String batchGraphJson = """
            {
                "name": "BatchProcessingGraph",
                "batch_size": 10,
                "nodes": [
                    {
                        "key": "BatchInput",
                        "name": "batch_input"
                    },
                    {
                        "key": "BatchProcessor", 
                        "name": "batch_processor"
                    },
                    {
                        "key": "BatchOutput",
                        "name": "batch_output"
                    }
                ]
            }
            """;
            
            // 模拟批量处理多个任务
            int batchCount = 5;
            long totalTime = 0;
            int successCount = 0;
            
            System.out.println("开始批量处理 " + batchCount + " 个任务...");
            
            for (int i = 0; i < batchCount; i++) {
                String taskId = "batch_task_" + String.format("%03d", i + 1);
                
                long startTime = System.currentTimeMillis();
                GraphRunnerResult result = runner.run(batchGraphJson, "BatchProcessingGraph", taskId);
                long endTime = System.currentTimeMillis();
                
                long taskTime = endTime - startTime;
                totalTime += taskTime;
                
                if (result.isSuccess()) {
                    successCount++;
                    System.out.printf("✓ %s 完成 (耗时: %dms, 内部耗时: %.1fms)\n", 
                                    taskId, taskTime, result.getTotalTime());
                } else {
                    System.err.printf("✗ %s 失败: %s\n", taskId, result.statusMessage);
                }
            }
            
            // 统计结果
            System.out.println("\n批量处理统计:");
            System.out.println("总任务数: " + batchCount);
            System.out.println("成功数: " + successCount);
            System.out.println("失败数: " + (batchCount - successCount));
            System.out.println("成功率: " + String.format("%.1f%%", (successCount * 100.0 / batchCount)));
            System.out.println("总耗时: " + totalTime + "ms");
            System.out.println("平均耗时: " + String.format("%.1fms", totalTime * 1.0 / batchCount));
            
        } catch (Exception e) {
            System.err.println("批量处理示例执行失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 主函数，运行所有示例
     */
    public static void main(String[] args) {
        System.out.println("nndeploy GraphRunner Java示例");
        System.out.println("==============================");
        
        try {
            // 运行基本示例
            basicExample();
            
            // 运行高级示例  
            advancedExample();
            
            // 运行批量处理示例
            batchProcessingExample();
            
            System.out.println("\n所有示例执行完成!");
            
        } catch (Exception e) {
            System.err.println("示例执行过程中发生未捕获异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
