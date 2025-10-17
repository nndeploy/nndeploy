@ECHO OFF
SET DIR=%~dp0
SET JAVA_EXE=%JAVA_HOME%\bin\java.exe
IF EXIST "%JAVA_EXE%" (
) ELSE (
  SET JAVA_EXE=java
)
"%JAVA_EXE%" -Xmx64m -cp "%DIR%\gradle\wrapper\gradle-wrapper.jar" org.gradle.wrapper.GradleWrapperMain %*
