---
name: feature_debugging_tools
title: 调试和追踪工具
description: 提供可视化的调试、性能分析和错误追踪工具，支持执行追踪、性能统计、断点管理和可视化输出
category: [feature]
difficulty: medium
priority: P2
status: planned
version: 1.0.0
tags: [dag, debug, profiler, visualizer, tracing, breakpoint]
estimated_time: 8h
files_affected: [framework/include/nndeploy/dag/debugger.h, framework/source/nndeploy/dag/debugger.cc, framework/include/nndeploy/dag/profiler.h, framework/source/nndeploy/dag/profiler.cc, framework/include/nndeploy/dag/visualizer.h, framework/source/nndeploy/dag/visualizer.cc]
---

# Feature: 调试和追踪工具

## 1. 背景（是什么 && 为什么）

### 现状分析
- **当前状态**：nndeploy DAG 模块缺少可视化的调试和追踪能力
- **存在问题**：难以追踪执行流程、分析性能瓶颈、定位错误
- **需求痛点**：需要可视化的调试工具、性能分析、错误追踪

### 设计问题
- 没有执行追踪日志
- 没有可视化调试输出
- 没有性能统计数据收集
- 没有错误诊断工具

## 2. 目标（想做成什么样子）

### 核心目标
- **执行追踪**：记录完整的执行流程和时间线
- **性能分析**：收集各节点的执行时间和资源使用
- **错误诊断**：提供详细的错误信息和堆栈追踪
- **可视化输出**：生成执行树、调用图等可视化结果

### 预期效果
- 用户可以查看详细的执行日志
- 可以导出执行追踪文件（兼容追踪工具）
- 可以可视化图结构和执行路径
- 可以分析性能瓶颈

## 3. 范围明确（需要修改与新增那些文件,不能修改和新增那些文件）

### 需要新增的文件
- `framework/include/nndeploy/dag/debugger.h` - 调试器接口
- `framework/source/nndeploy/dag/debugger.cc` - 调试器实现
- `framework/include/nndeploy/dag/profiler.h` - 性能分析器
- `framework/source/nndeploy/dag/profiler.cc` - 性能分析实现
- `framework/include/nndeploy/dag/visualizer.h` - 可视化工具
- `framework/source/nndeploy/dag/visualizer.cc` - 可视化实现

### 需要修改的文件
- `framework/include/nndeploy/dag/executor.h` - 集成调试触发点
- `framework/include/nndeploy/dag/graph_runner.h` - 集成调试器

### 不能修改的文件
- 现有的执行逻辑保持不变
- 不修改核心数据结构

### 影响范围
- 调试功能可选启用
- 不启用时无性能影响

## 4. 设计方案（大致的方案）

### 新旧方案对比
- **旧方案**：只有简单的日志输出
- **新方案**：提供完整的调试和追踪工具链
- **核心变化**：新增 Debugger、Profiler、Visualizer

### 架构/接口设计

#### 调试器
```cpp
// 调试级别
enum DebugLevel {
    kDebugLevelNone,      // 关闭
    kDebugLevelError,     // 只记录错误
    kDebugLevelWarn,      // 警告及以上
    kDebugLevelInfo,      // 信息及以上
    kDebugLevelDebug,     // 调试及以上
    kDebugLevelTrace      // 追踪（最详细）
};

// 执行追踪事件
struct TraceEvent {
    std::string event_id;         // 事件 ID
    std::string parent_id;        // 父事件 ID
    std::string node_name;        // 节点名称
    std::string event_type;       // 事件类型（start/end/error）
    int64_t timestamp;           // 时间戳（微秒）
    int64_t duration_us;          // 持续时间（微秒）
    nlohmann::json data;          // 事件数据
    std::string thread_id;        // 线程 ID
};

// 调试器
class Debugger {
public:
    Debugger(const std::string& session_id = "");
    ~Debugger();

    // 设置调试级别
    void setLevel(DebugLevel level);
    DebugLevel getLevel() const { return level_; }

    // 会话管理
    std::string getSessionId() const { return session_id_; }
    void setSessionId(const std::string& id) { session_id_ = id; }

    // 追踪事件
    void traceStart(const std::string& node_name, const nlohmann::json& data = {});
    void traceEnd(const std::string& node_name, const nlohmann::json& data = {});
    void traceError(const std::string& node_name, const std::string& error,
                    const nlohmann::json& data = {});

    // 记录日志
    void log(DebugLevel level, const std::string& message);
    void logError(const std::string& message);
    void logWarn(const std::string& message);
    void logInfo(const std::string& message);
    void logDebug(const std::string& message);
    void logTrace(const std::string& message);

    // 获取追踪事件
    std::vector<TraceEvent> getTraceEvents() const;
    std::vector<TraceEvent> getTraceEvents(const std::string& node_name) const;

    // 导出追踪（兼容 Chrome Trace Viewer）
    bool exportTrace(const std::string& filename, const std::string& format = "json");

    // 清空追踪
    void clearTrace();

    // 启用/禁用调试
    void enable(bool enabled = true) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    // 设置输出目标
    void setOutputFile(const std::string& filename);
    void setStdout(bool enable);

private:
    std::string session_id_;
    DebugLevel level_;
    bool enabled_;
    std::vector<TraceEvent> trace_events_;
    std::map<std::string, std::string> event_stack_;  // node_name -> event_id
    std::ofstream output_file_;
    bool stdout_enabled_;
    std::mutex mutex_;
};

// 断点管理
class BreakpointManager {
public:
    // 设置断点
    bool setBreakpoint(const std::string& node_name, const std::string& condition = "");

    // 删除断点
    bool removeBreakpoint(const std::string& node_name);

    // 检查是否触发断点
    bool shouldBreak(const std::string& node_name, const nlohmann::json& context);

    // 获取断点列表
    std::vector<std::string> getBreakpoints() const;

    // 清空所有断点
    void clear();

private:
    struct Breakpoint {
        std::string node_name;
        std::string condition;  // 可选的条件表达式
        int hit_count;
    };
    std::map<std::string, Breakpoint> breakpoints_;
    std::mutex mutex_;
};
```

#### 性能分析器
```cpp
// 性能统计
struct PerformanceStats {
    std::string node_name;
    int call_count;             // 调用次数
    int64_t total_time_us;      // 总时间（微秒）
    int64_t avg_time_us;       // 平均时间
    int64_t max_time_us;       // 最大时间
    int64_t min_time_us;       // 最小时间
    std::vector<int64_t> samples;  // 所有采样

    // 内存使用（可选）
    size_t peak_memory_bytes;
    size_t avg_memory_bytes;

    // 其他指标
    int error_count;
};

// 性能分析器
class Profiler {
public:
    Profiler(const std::string& session_id = "");
    ~Profiler();

    // 启用/禁用性能分析
    void enable(bool enabled = true);
    bool isEnabled() const { return enabled_; }

    // 会话管理
    std::string getSessionId() const { return session_id_; }

    // 开始计时
    void startTiming(const std::string& node_name);

    // 结束计时
    void endTiming(const std::string& node_name);

    // 记录内存使用
    void recordMemory(const std::string& node_name, size_t bytes);

    // 记录错误
    void recordError(const std::string& node_name);

    // 获取性能统计
    PerformanceStats getStats(const std::string& node_name) const;
    std::vector<PerformanceStats> getAllStats() const;

    // 获取瓶颈节点
    std::vector<std::string> getBottlenecks(size_t top_n = 5) const;

    // 导出性能报告
    bool exportReport(const std::string& filename);
    bool exportReportCSV(const std::string& filename);

    // 打印摘要
    void printSummary();

    // 清空统计
    void clear();

private:
    std::string session_id_;
    bool enabled_;
    std::map<std::string, PerformanceStats> stats_;
    std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> timers_;
    std::mutex mutex_;
};
```

#### 可视化工具
```cpp
// 可视化格式
enum VisualizationFormat {
    kVisFormatDot,          // Graphviz DOT
    kVisFormatJson,         // JSON
    kVisFormatMermaid,      // Mermaid
    kVisFormatHTML          // 交互式 HTML
};

// 可视化选项
struct VisualizationOptions {
    bool show_execution_time;    // 显示执行时间
    bool show_data_flow;         // 显示数据流
    bool highlight_executed;     // 高亮已执行节点
    bool show_errors;            // 显示错误节点
    bool color_by_time;          // 按时间着色
    bool show_thread_id;         // 显示线程 ID
    bool show_device_info;       // 显示设备信息
};

// 可视化器
class Visualizer {
public:
    Visualizer(Graph* graph);

    // 生成可视化输出
    bool visualize(const std::string& filename,
                   VisualizationFormat format = kVisFormatDot,
                   const VisualizationOptions& options = {});

    // 生成执行树可视化
    bool visualizeExecutionTree(const std::vector<TraceEvent>& events,
                                 const std::string& filename);

    // 生成性能热力图
    bool generateHeatmap(const std::vector<PerformanceStats>& stats,
                        const std::string& filename);

    // 生成调用图
    bool generateCallGraph(const std::string& filename);

    // 获取 Graphviz DOT 字符串
    std::string toDot(const VisualizationOptions& options = {});

    // 获取 Mermaid 字符串
    std::string toMermaid(const VisualizationOptions& options = {});

    // 获取 JSON
    nlohmann::json toJson(const VisualizationOptions& options = {});

private:
    std::string getColorByTime(int64_t time_us, int64_t max_time_us);
    Graph* graph_;
};
```

#### 集成 API
```cpp
// GraphRunner 集成
class GraphRunner {
public:
    // 获取调试器
    std::shared_ptr<Debugger> getDebugger();
    void setDebugger(std::shared_ptr<Debugger> debugger);

    // 获取性能分析器
    std::shared_ptr<Profiler> getProfiler();
    void setProfiler(std::shared_ptr<Profiler> profiler);

    // 获取可视化器
    std::shared_ptr<Visualizer> getVisualizer();

    // 启用调试模式
    void enableDebugMode(DebugLevel level = kDebugLevelDebug);

    // 生成调试报告
    bool generateDebugReport(const std::string& output_dir);

protected:
    std::shared_ptr<Debugger> debugger_;
    std::shared_ptr<Profiler> profiler_;
    bool debug_enabled_;
};

// 便捷宏
#define NNDEPLOY_DEBUG_TRACE_START(node) \
    if (debugger_) debugger_->traceStart(#node);

#define NNDEPLOY_DEBUG_TRACE_END(node) \
    if (debugger_) debugger_->traceEnd(#node);

#define NNDEPLOY_DEBUG_LOG(level, msg) \
    if (debugger_) debugger_->log(kDebugLevel##level, msg);

#define NNDEPLOY_PROFILE_START(node) \
    if (profiler_) profiler_->startTiming(#node);

#define NNDEPLOY_PROFILE_END(node) \
    if (profiler_) profiler_->endTiming(#node);
```

### 核心操作流程

#### 调试追踪流程
```
┌─────────────────┐
│  GraphRunner    │
│  开始执行       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Debugger       │
│  traceStart()   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  节点执行       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Debugger       │
│  traceEnd()     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  导出追踪文件    │
└─────────────────┘
```

#### 可视化流程
```
┌─────────────────┐
│  Graph          │
│  结构 + 追踪     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualizer     │
│  toDot/Mermaid  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  生成可视化文件  │
│  (dot/svg/html) │
└─────────────────┘
```

### 技术细节
- 使用 Chrome Trace Viewer 格式（JSON trace event format）
- 使用 Graphviz 生成图形（DOT 格式）
- 支持多种可视化格式（DOT、JSON、Mermaid、HTML）
- 导出文件可离线查看
- 调试信息管理：
  - 内存限制（默认 100MB，超出后循环覆盖）
  - 流式导出（避免内存溢出）
  - 采样模式（降低开销）
- 性能分析：
  - 使用高精度时钟（std::chrono::high_resolution_clock）
  - 支持分层统计（节点/图/全局）
  - 瓶颈识别（Top N 耗时节点）

## 5. 实施步骤

### Step 1: 实现调试器
- 实现 Debugger 类
- 实现断点管理器
- 涉及文件：`framework/include/nndeploy/dag/debugger.h`, `framework/source/nndeploy/dag/debugger.cc`

### Step 2: 实现性能分析器
- 实现 Profiler 类
- 实现性能统计和报告生成
- 涉及文件：`framework/include/nndeploy/dag/profiler.h`, `framework/source/nndeploy/dag/profiler.cc`

### Step 3: 实现可视化工具
- 实现 Visualizer 类
- 实现多种可视化格式
- 涉及文件：`framework/include/nndeploy/dag/visualizer.h`, `framework/source/nndeploy/dag/visualizer.cc`

### Step 4: 集成 GraphRunner
- 集成调试器、性能分析器、可视化器
- 实现调试报告生成
- 涉及文件：`framework/include/nndeploy/dag/graph_runner.h`, `framework/source/nndeploy/dag/graph_runner.cc`

### Step 5: 添加调试触发点
- 在 Executor 中添加调试触发点
- 添加性能分析触发点
- 涉及文件：`framework/include/nndeploy/dag/executor.h`, `framework/source/nndeploy/dag/executor.cc`

### Step 6: 测试和文档
- 编写单元测试
- 编写使用示例
- 创建可视化 Demo
- 涉及文件：`test/`, `docs/`, `demo/`

### 兼容性与迁移
- 调试功能可选启用
- 不启用时无性能影响
- 现有代码无需修改

## 6. 验收标准

### 功能测试
- **测试用例 1**：调试器正确记录执行事件
- **测试用例 2**：导出的追踪文件格式正确
- **测试用例 3**：性能分析器正确统计时间
- **测试用例 4**：瓶颈节点识别正确
- **测试用例 5**：可视化生成正确的 DOT/Mermaid
- **测试用例 6**：断点触发正确
- **测试用例 7**：调试报告生成完整
- **测试用例 8**：不启用调试时无性能影响

### 代码质量
- 通过 GoogleTest 单元测试
- 代码覆盖率 > 80%
- 符合项目代码规范
- 头文件注释完善

### 回归测试
- 现有 DAG 执行功能正常运行
- 不启用调试时无影响
- 各种 Executor 仍能正常工作

### 性能与可维护性
- 调试追踪开销 < 5%（启用时）
- 性能分析开销 < 10%（启用时）
- 代码结构清晰，易于扩展

### 文档与示例
- API 文档完整
- 提供使用示例
- 提供可视化 Demo
- 提供追踪文件查看指南

## 7. 其他说明

### 相关资源
- Chrome Trace Viewer
- Graphviz 文档
- OpenTelemetry 追踪规范

### 风险与应对
- **风险**：调试信息占用大量内存
  - **应对**：提供内存限制，支持流式导出
- **风险**：性能分析影响执行速度
  - **应对**：默认关闭，按需启用，使用采样
- **风险**：可视化文件过大
  - **应对**：提供过滤选项，支持分层显示

### 依赖关系
- 依赖：回调系统
- 依赖：Context（用于追踪上下文）

### 扩展方向
- 集成 OpenTelemetry 标准
- 支持分布式追踪
- 支持实时可视化（WebSocket）
- 支持 AI 辅助调试
