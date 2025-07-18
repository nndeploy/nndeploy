@echo off
setlocal enabledelayedexpansion
rem Automatically set third-party library environment variables and establish executable and dynamic library connection relationships
rem Usage: .\set_install_env.bat

set WORKSPACE=%cd%
set THIRDPARTY_DIR=%WORKSPACE%\third_party

echo Checking third-party library directory: %THIRDPARTY_DIR%

rem Check if third-party library directory exists
if not exist "%THIRDPARTY_DIR%" (
    echo Warning: Third-party library directory not found: %THIRDPARTY_DIR%
    exit /b 1
)

rem Traverse all subdirectories in the third-party library directory
for /d %%i in ("%THIRDPARTY_DIR%\*") do (
    rem Check lib directory (static libraries and import libraries)
    if exist "%%i\lib" (
        set LIB=%%i\lib;!LIB!
        echo Found library path: %%i\lib
    )
    
    rem Check bin directory (dynamic libraries and executables)
    if exist "%%i\bin" (
        set PATH=%%i\bin;!PATH!
        echo Found executable path: %%i\bin
    )
)

rem set nndeploy library paths
set NNDEPLOY_LIB_PATH=%WORKSPACE%\lib
set NNDEPLOY_BIN_PATH=%WORKSPACE%\bin

rem Add nndeploy library path to LIB environment variable
if exist "%NNDEPLOY_LIB_PATH%" (
    set LIB=!NNDEPLOY_LIB_PATH!;!LIB!
    echo Added nndeploy library path: %NNDEPLOY_LIB_PATH%
) else (
    echo Warning: nndeploy library path not found: %NNDEPLOY_LIB_PATH%
)

rem Add nndeploy executable path to PATH environment variable
if exist "%NNDEPLOY_BIN_PATH%" (
    set PATH=!NNDEPLOY_BIN_PATH!;!PATH!
    echo Added nndeploy executable path: %NNDEPLOY_BIN_PATH%
) else (
    echo Warning: nndeploy executable path not found: %NNDEPLOY_BIN_PATH%
)

echo.
echo Environment variables setup completed, executable and dynamic library connection relationships established
echo Current PATH: !PATH!
echo Current LIB: !LIB!


echo run demo...
cd demo
@REM run demo
@REM To run other demos, replace the execution command below
@REM Available demo examples:
@REM nndeploy_demo_dag.exe
@REM nndeploy_demo_detect.exe --name nndeploy::detect::YoloGraph --inference_type kInferenceTypeDefault --device_type kDeviceTypeCodeAscendCL:0 --model_type kModelTypeDefault --is_path --model_value /home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx.json,/home/ascenduserdg01/model/nndeploy/detect/yolo11s.sim.onnx.safetensors --codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential --yolo_version 11 --model_inputs images --model_outputs output0 --input_path ../docs/image/demo/detect/sample.jpg --output_path yolo_nndeploy_acl_sample_output.jpg
nndeploy_demo_dag.exe