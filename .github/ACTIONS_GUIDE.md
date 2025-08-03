# GitHub Actions 配置使用指南

## 概述

本项目配置了完整的CI/CD流水线，包含以下工作流：

1. **CI/CD Pipeline** (`ci.yml`) - 主要的构建和测试流程
2. **Python CI/CD** (`python.yml`) - Python应用的专用流程
3. **Docker Build** (`docker.yml`) - Docker镜像构建和发布
4. **Legacy Workflows** - 原有的平台特定构建

## 工作流详细说明

### 1. 主要CI/CD流程 (ci.yml)

#### 触发条件
- 推送到 `main`, `develop`, `feature/**`, `hotfix/**` 分支
- 针对 `main`, `develop` 分支的Pull Request
- 手动触发（支持参数配置）

#### 包含的任务
- 📋 代码质量检查（格式、静态分析）
- 🏗️ 多平台构建矩阵（Linux/Windows/macOS × Debug/Release）
- 🧪 单元测试和集成测试
- ⚡ 性能基准测试
- 📚 文档自动构建
- 🔒 安全扫描（CodeQL）
- 🚀 自动发布（标签推送时）
- 📢 构建状态通知

#### 手动触发参数
- `run_tests`: 是否运行测试（布尔值）
- `build_type`: 构建类型（Debug/Release/RelWithDebInfo）

### 2. Python流程 (python.yml)

#### 功能特性
- 🐍 多Python版本测试（3.8-3.11）
- 🎨 代码格式检查（black, flake8）
- 🔍 类型检查（mypy）
- 📊 测试覆盖率报告
- 🛡️ 安全扫描（bandit, safety）

### 3. Docker流程 (docker.yml)

#### 功能特性
- 🐳 多架构构建（amd64/arm64）
- 🏷️ 智能标签管理
- 💾 构建缓存优化
- 🔒 镜像安全扫描（Trivy）
- 📦 自动推送到GitHub Container Registry

## 配置要求

### 1. 仓库设置

#### 必需的权限设置
```yaml
permissions:
  contents: read
  packages: write
  security-events: write
```

#### 可选的Secrets
- `CODECOV_TOKEN` - 用于上传测试覆盖率
- `SLACK_WEBHOOK` - 用于Slack通知
- `EMAIL_SERVER_*` - 用于邮件通知

### 2. 分支保护规则

建议为 `main` 分支设置以下保护规则：

```
必需的状态检查：
- 代码质量检查
- 构建-ubuntu-latest-Release
- 构建-windows-latest-Release  
- 构建-macos-latest-Release
- 测试-ubuntu-latest

其他设置：
✅ 要求分支是最新的
✅ 要求线性历史
✅ 包括管理员
```

### 3. 文件结构要求

确保以下文件存在：
```
├── cmake/config.cmake          # CMake配置
├── requirements.txt            # Python依赖
├── .clang-format              # C++格式化配置
├── run_clang_format.py        # 格式化脚本
├── test/                      # 测试目录
└── docs/                      # 文档目录
```

## 使用说明

### 1. 启用工作流

1. 将配置文件推送到仓库
2. 前往 GitHub 仓库的 "Actions" 标签页
3. 启用GitHub Actions（如果未启用）
4. 工作流将在下次推送时自动运行

### 2. 查看构建状态

- **总览**: 仓库主页的绿色✅或红色❌状态图标
- **详细**: Actions标签页 → 选择具体的工作流运行
- **徽章**: 可在README中添加状态徽章

```markdown
![CI Status](https://github.com/用户名/仓库名/workflows/CI%2FCD%20Pipeline/badge.svg)
```

### 3. 手动触发构建

1. 前往 Actions → CI/CD Pipeline
2. 点击 "Run workflow"
3. 选择分支和参数
4. 点击 "Run workflow" 执行

### 4. 发布流程

#### 创建发布版本
```bash
# 创建并推送标签
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

这将自动：
- 触发完整的构建和测试
- 创建GitHub Release
- 上传构建产物
- 构建并推送Docker镜像

## 高级配置

### 1. 自定义构建矩阵

修改 `ci.yml` 中的matrix配置：
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    config: [Debug, Release]
    # 添加更多维度
    compiler: [gcc, clang]
```

### 2. 添加部署环境

在 `ci.yml` 末尾添加部署job：
```yaml
deploy:
  name: 部署到生产环境
  if: github.ref == 'refs/heads/main'
  needs: [build, test]
  runs-on: ubuntu-latest
  environment: production
  steps:
    # 部署步骤
```

### 3. 自定义通知

修改 `notify` job 添加自定义通知逻辑：
```yaml
- name: 发送Slack通知
  if: always()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 故障排除

### 常见问题

1. **构建失败** - 检查依赖项和构建脚本
2. **测试超时** - 增加timeout-minutes设置
3. **权限错误** - 检查GITHUB_TOKEN权限
4. **缓存问题** - 清除Actions缓存

### 调试技巧

1. **启用调试日志**:
   ```yaml
   env:
     ACTIONS_STEP_DEBUG: true
   ```

2. **SSH调试**:
   ```yaml
   - name: SSH调试
     uses: mxschmitt/action-tmate@v3
     if: ${{ failure() }}
   ```

3. **保存构建日志**:
   ```yaml
   - name: 上传日志
     if: always()
     uses: actions/upload-artifact@v4
     with:
       name: build-logs
       path: build/*.log
   ```

## 最佳实践

1. **💡 渐进式采用** - 先启用基础功能，再逐步添加高级特性
2. **🔄 定期更新** - 保持Actions版本最新
3. **📊 监控指标** - 关注构建时间和成功率
4. **🛡️ 安全第一** - 定期检查安全扫描结果
5. **📝 文档维护** - 及时更新配置文档

## 支持的事件类型

| 事件 | 描述 | 使用场景 |
|------|------|----------|
| `push` | 代码推送 | 持续集成 |
| `pull_request` | PR创建/更新 | 代码审查 |
| `workflow_dispatch` | 手动触发 | 按需构建 |
| `schedule` | 定时任务 | 夜间构建 |
| `release` | 发布事件 | 自动部署 |
| `issue` | Issue事件 | 自动化管理 |

---

> 📚 更多信息请参考：[GitHub Actions文档](https://docs.github.com/en/actions) 