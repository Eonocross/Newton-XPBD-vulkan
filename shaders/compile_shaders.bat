@echo off
setlocal enabledelayedexpansion

where glslc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    if defined VULKAN_SDK (
        set "PATH=%VULKAN_SDK%\bin;%PATH%"
    ) else (
        echo [ERROR] glslc not found. Please install Vulkan SDK or add glslc to PATH.
        exit /b 1
    )
)

echo Compiling GLSL compute shaders to SPIR-V...
set "OUT_DIR=..\addons\modules\vkxpbd_shaders"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

for %%f in (*.comp) do (
    echo Compiling %%f -> %OUT_DIR%\%%~nf.spv
    glslc --target-env=vulkan1.3 -O "%%f" -o "%OUT_DIR%\%%~nf.spv"
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to compile %%f
        exit /b 1
    )
)

echo All shaders compiled successfully!
