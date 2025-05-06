
# nndeploy Code of Conduct

## Coding Guidelines

Please ensure your code follows the nndeploy coding guidelines. Consistent coding style helps maintain the codebase.

Based on Google style guide, with some differences(perhaps):

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
  struct MyStruct {
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

### Python

Based on the Google Python Style Guide, the basic requirements are as follows:

1. Files and Encoding
   - **File Encoding**: Files should be encoded in UTF-8 and usually do not need an encoding declaration at the top of the file (Python 3 uses UTF-8 by default).
   - **File Naming**: File names should use lowercase letters, with words separated by underscores, e.g., `my_module.py`.

2. Comments
   - **Module Comments**: Each module should start with a module-level docstring describing the module's purpose, functionality, and main classes or functions. For example:
     ```python
     """
     This module provides utility functions for handling strings.

     It includes functions for string formatting, splitting, and joining.
     """
     ```
   - **Function and Method Comments**: Functions and methods also require docstrings explaining their purpose, parameters, return values, and possible exceptions. For example:
     ```python
     def add_numbers(a, b):
         """
         Add two numbers together.

         Args:
             a (int): The first number.
             b (int): The second number.

         Returns:
             int: The sum of a and b.
         """
         return a + b
     ```
   - **Inline Comments**: Use inline comments in the code to explain complex or non-intuitive parts, but don't overuse them. Comments should be concise and clear.

3. Classes and Objects
   - **Class Naming**: Class names should use CapWords convention, e.g., `MyClass`.
   - **Class Docstrings**: Classes should include a docstring describing their purpose and main functionality.
     ```python
     class MyClass:
         """
         A simple class that represents a person.

         Attributes:
             name (str): The name of the person.
             age (int): The age of the person.
         """
         def __init__(self, name, age):
             self.name = name
             self.age = age

         def get_info(self):
             """
             Get the person's information.

             Returns:
                 str: A string containing the person's name and age.
             """
             return f"Name: {self.name}, Age: {self.age}"
     ```
   - **Member Variable and Method Naming**: Member variables and method names should use lowercase with words separated by underscores, e.g., `my_variable` and `my_method`.

4. Functions and Methods
   - **Function Naming**: Function names should use lowercase with words separated by underscores, e.g., `calculate_sum`.
   - **Parameter Naming**: Parameter names should also use lowercase with words separated by underscores and be descriptive.
   - **Function Length**: Functions should be kept short and focused on a single task. Avoid excessively long functions.

5. Code Layout
   - **Indentation**: Use 4 spaces for indentation, not tabs.
   - **Line Length**: Limit each line of code to a maximum of 80 characters. If it exceeds, break the line while maintaining code readability. Usually break after operators. For example:
     ```python
     result = some_function(arg1, arg2,
                            arg3, arg4)
     ```
   - **Blank Lines**: Use blank lines to separate logical sections, e.g., two blank lines between functions and classes, one blank line between methods within a class.

6. Import Statements
   - **Import Order**: Group import statements in the order of standard library, third-party libraries, and local modules, with a blank line between each group. For example:
     ```python
     import os
     import sys

     import requests

     from my_module import my_function
     ```
   - **Avoid Wildcard Imports**: Avoid using wildcard imports like `from module import *` as they can lead to naming conflicts and reduced code readability.

7. Exception Handling
   - **Specific Exception Types**: When catching exceptions, specify the exact exception type instead of using a generic `except` statement. For example:
     ```python
     try:
         result = 1 / 0
     except ZeroDivisionError:
         print("Division by zero occurred.")
     ```

8. Testing
   - **Write Unit Tests**: Write unit tests for your code to ensure correctness and stability. You can use Python's `unittest` or `pytest` testing frameworks.


## 代码规范

请确保您的代码遵循nndeploy的代码规范。统一的代码风格有助于维护代码库。

### C++
基于Google代码风格指南，但可能有以下不同：

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
  struct MyStruct {
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

### Python

基于Google代码风格指南，基本要求如下：

- 1. 文件和编码
  - **文件编码**：文件应使用 UTF - 8 编码，并且通常不需要在文件开头指定编码声明（Python 3 默认使用 UTF - 8）。
  - **文件命名**：文件名应使用小写字母，单词之间可以用下划线分隔，例如 `my_module.py`。

- 2. 注释
  - **模块注释**：每个模块开头应包含一个模块级别的文档字符串，描述模块的功能、用途和包含的主要类或函数。例如：
  ```python
  """
  This module provides utility functions for handling strings.

  It includes functions for string formatting, splitting, and joining.
  """
  ```
  - **函数和方法注释**：函数和方法也需要文档字符串，说明函数的功能、参数、返回值和可能抛出的异常。例如：
  ```python
  def add_numbers(a, b):
      """
      Add two numbers together.

      Args:
          a (int): The first number.
          b (int): The second number.

      Returns:
          int: The sum of a and b.
      """
      return a + b
  ```
  - **行内注释**：在代码中使用行内注释来解释复杂或不直观的代码部分，但不要过度使用，注释应简洁明了。

- 3. 类和对象
  - **类命名**：类名应使用大写字母开头的驼峰命名法，例如 `MyClass`。
  - **类的文档字符串**：类应包含一个文档字符串，描述类的用途和主要功能。
  ```python
  class MyClass:
      """
      A simple class that represents a person.

      Attributes:
          name (str): The name of the person.
          age (int): The age of the person.
      """
      def __init__(self, name, age):
          self.name = name
          self.age = age

      def get_info(self):
          """
          Get the person's information.

          Returns:
              str: A string containing the person's name and age.
          """
          return f"Name: {self.name}, Age: {self.age}"
  ```
  - **成员变量和方法命名**：成员变量和方法名应使用小写字母，单词之间用下划线分隔，例如 `my_variable` 和 `my_method`。

- 4. 函数和方法
  - **函数命名**：函数名应使用小写字母，单词之间用下划线分隔，例如 `calculate_sum`。
  - **参数命名**：参数名也应使用小写字母，单词之间用下划线分隔，参数应具有描述性。
  - **函数长度**：函数应尽量短小，一个函数最好只完成一个明确的任务，避免函数过长。

- 5. 代码布局
  - **缩进**：使用 4 个空格进行缩进，而不是制表符。
  - **行长度**：每行代码的长度尽量不超过 80 个字符，如果超过可以进行换行。换行时应保持代码的可读性，通常在运算符之后换行。例如：
  ```python
  result = some_function(arg1, arg2,
                       arg3, arg4)
  ```
  - **空行**：在不同的逻辑块之间使用空行分隔，例如函数之间、类之间使用两个空行分隔，类的方法之间使用一个空行分隔。

- 6. 导入语句
  - **导入顺序**：导入语句应按照标准库、第三方库和本地库的顺序进行分组，每组之间用一个空行分隔。例如：
  ```python
  import os
  import sys

  import requests

  from my_module import my_function
  ```
  - **避免使用通配符导入**：尽量避免使用 `from module import *` 这种通配符导入方式，因为它可能会导致命名冲突和代码可读性降低。

- 7. 异常处理
  - **异常类型明确**：在捕获异常时，应明确指定要捕获的异常类型，而不是使用通用的 `except` 语句。例如：
  ```python
  try:
      result = 1 / 0
  except ZeroDivisionError:
      print("Division by zero occurred.")
  ```

- 8. 测试
  - **编写单元测试**：为代码编写单元测试，确保代码的正确性和稳定性。可以使用 Python 的 `unittest` 或 `pytest` 等测试框架。