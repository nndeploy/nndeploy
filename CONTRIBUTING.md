# CONTRIBUTING

nndeploy is in its development stage. If you love open source and enjoy tinkering, whether for learning purposes or having better ideas, we welcome you to join us.

## How Can You Contribute?

There are several ways you can contribute to nndeploy:

1. **Report Issues:** Help us find and fix bugs
2. **Feature Requests:** Suggest new features and improvements  
3. **Pull Requests:** Contribute code to improve the project
4. **Documentation:** Help improve documentation
5. **Share:** Share this project with others

### Reporting Issues

If you find any issues or bugs, please report them. Here's how to do it:

1. Check if the issue has already been reported in the [Issues](https://github.com/nndeploy/nndeploy/issues) section.
2. If not, click the "New Issue" button and provide a clear title and detailed description.
3. Share relevant information like your environment setup and steps to reproduce the issue.

### Feature Requests

Have a good idea for nndeploy? Share it with us! Here's how to request a feature:

1. Check if your idea is already in the [Issues](https://github.com/nndeploy/nndeploy/issues) section
2. If it's a new feature request, click "New Issue" and use the "Feature Request" template to describe your idea
3. Add WeChat: titian5566 (Note: nndeploy+name), share your ideas with us

### Pull Requests

To submit a pull request, follow these steps:

1. Fork this repository
2. Create a `new-branch` branch for your changes
3. Make your modifications, ensuring your code adheres to the project's coding guidelines
4. Test your changes to verify they work correctly
5. Document your changes and provide clear commit messages
6. Push your changes to your fork
7. Create a pull request to the `new-branch` branch of this repository

We will review your pull request, communicate with you, and work together to merge it into the main repository.

## Coding Guidelines

Please ensure your code follows the nndeploy coding guidelines. Consistent coding style helps maintain the codebase.

Based on Google style guide, with some differences:

- Function naming: First letter lowercase, camelCase
  ```
  int myFunction() {
    return 0;
  }
  ```

- Class and struct naming: First letter uppercase, camelCase 
  ```
  class MyClass {
  }
  struct MyStruct {
  }
  ```

- Variable naming: First letter lowercase, words connected by underscore
  ```
  int my_variable = 0;
  ```

- Member variable naming: First letter lowercase, words connected by underscore, ends with underscore
  ```
  class MyClass {
      int my_variable_;
  }
  ```

- Macro naming: All uppercase, words connected by underscore
  ```
  #define MY_MACRO 1
  ```

- Enum naming: Starts with lowercase 'k', camelCase
  ```
  enum MyEnum {
    kMyEnumValue1,
    kMyEnumValue2, 
    kMyEnumValue3,
  };
  ```

# 贡献指南

nndeploy正处于发展阶段，如果您热爱开源、喜欢折腾，不论是出于学习目的，抑或是有更好的想法，欢迎加入我们。

## 如何贡献？

您可以通过以下方式参与nndeploy项目:

1. **报告问题:** 帮助我们发现和修复bug
2. **功能建议:** 提出新功能和改进建议  
3. **提交代码:** 贡献代码以改进项目
4. **完善文档:** 帮助改进和完善文档
5. **推广项目:** 向他人推荐和分享本项目

### 报告问题

如果您发现任何问题或bug，请按以下步骤报告：

1. 检查该问题是否已在[Issues](https://github.com/nndeploy/nndeploy/issues)页面被报告过
2. 如果没有，点击"New Issue"按钮，提供清晰的标题和详细描述
3. 分享相关信息，如您的环境配置和复现问题的步骤

### 功能建议

有更好的想法? 请分享给我们，以下是提出功能建议的步骤:

1. 检查您的想法是否已在[Issues](https://github.com/nndeploy/nndeploy/issues)页面提出
2. 如果是新的功能建议,点击"New Issue"并使用"Feature Request"模板描述您的想法
3. 添加微信：titian5566 (备注：nndeploy+姓名)，将您的想法分享给我们

### 提交代码

要提交代码，请按以下步骤操作：

1. Fork本仓库
2. 创建一个`new-branch`分支用于您的修改
3. 进行修改，确保代码符合项目的编码规范
4. 测试您的修改以验证其正确性
5. 记录您的修改并提供清晰的提交信息
6. 将修改推送到您的fork仓库
7. 向本仓库的`new-branch`分支创建pull request

我们会审查您的pull request，与您沟通，并一起将其合并到主仓库中。

## 代码规范

请确保您的代码遵循nndeploy的代码规范。统一的代码风格有助于维护代码库。

基于Google代码风格指南，但有以下不同：

- 函数命名：首字母小写，驼峰命名
  ```
  int myFunction() {
    return 0;
  }
  ```

- 类和结构体命名：首字母大写，驼峰命名
  ```
  class MyClass {
  }
  struct MyStruct {
  }
  ```

- 变量命名：首字母小写，单词间下划线连接
  ```
  int my_variable = 0;
  ```

- 成员变量命名：首字母小写，单词间下划线连接，以下划线结尾
  ```
  class MyClass {
      int my_variable_;
  }
  ```

- 宏定义命名：全大写，单词间下划线连接
  ```
  #define MY_MACRO 1
  ```

- 枚举命名：以小写字母k开头，驼峰命名
  ```
  enum MyEnum {
    kMyEnumValue1,
    kMyEnumValue2, 
    kMyEnumValue3,
  };
  ```
