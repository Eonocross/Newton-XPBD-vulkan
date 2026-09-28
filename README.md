# Newton-XPBD-vulkan
Aiming bit-for-bit identical results of the originally cuda Newton XPBD solver across different hardwares using vulkan. Currently only has plugins for Blender (tested only on Blender 5.x) with only cached simulation/playback (no realtime)

# Quick Setup Guide (Pre-built Release)

### Downloads

| Hardware | Status | Latest Binaries Download Link |
| :--- | :---: | :--- |
| **NVIDIA** | **Effectively Bit-exact** | [Windows x64 / Linux x86_64](https://github.com/Eonocross/Newton-XPBD-vulkan/releases) |
| **AMD (discrete)**  | **Untested** | *Should use standard Windows / Linux binaries* |
| **AMD (integrated)** | **Untested** | *Should use standard Windows / Linux binaries* |
| **Intel UHD / Iris Xe** | **Compat\*** | *Should use standard Windows / Linux binaries* |
| **Apple Silicon** | **Untested** | *not sure* |
| **CPU (x86 / arm64) & Mobile** | **Not a target yet** | — |

> **\* Compatibility mode (`Compat`)**: runs on hardware without native float atomics. Results will likely not be bit-exact against CUDA.

---


### Blender Plugin

[Download Plugin ZIP](https://github.com/Eonocross/Newton-XPBD-vulkan/releases) (`newton_vulkan_plugin.py`)

---


### Step 1: Install Vulkan Modules & Shaders
Download the binaries package for your OS and extract the `modules/` folder directly into your Blender `addons` directory:
- **Windows**: `%APPDATA%\Blender Foundation\Blender\<version>\scripts\addons\`
- **Linux**: `~/.config/blender/<version>/scripts/addons/`

The resulting folder layout must look like:
```text
scripts/addons/
└── modules/
    ├── vkxpbd.pyd (or vkxpbd.so on Linux)
    └── vkxpbd_shaders/
        ├── *.spv (17 shader files)
```

### Step 2: Install the Blender Plugin
Download the plugin package and place `newton_vulkan_plugin.py` into your Blender `addons` directory:
- **Windows**: `%APPDATA%\Blender Foundation\Blender\<version>\scripts\addons\`
- **Linux**: `~/.config/blender/<version>/scripts/addons/`

*(Alternatively, you can drag and drop `newton_vulkan_plugin.py` directly into the Blender viewport).*

### Step 3: Enable the Addon
1. In Blender, open **Edit** > **Preferences** > **Add-ons**.
2. Search for and enable **Newton Vulkan Direct**.
3. In the 3D Viewport sidebar (`N` panel under **Newton Vulkan**), verify your NVIDIA GPU appears in the **Hardware & Output > GPU** dropdown.



---

# Build Guide: (NVIDIA Target)

> **Hardware Target**: Tested and verified on **NVIDIA GPUs** (GeForce RTX / Quadro / Tesla).  
> Driver requirement: NVIDIA Display Driver **510.xx or newer** with Vulkan 1.3 support.

### Index
- [1. Clone & Prerequisites](#1-clone--prerequisites)
  - [A. Clone Repository](#a-clone-repository)
  - [B. Hardware & Drivers](#b-hardware--drivers)
  - [C. Development Toolchain](#c-development-toolchain)
- [2. Windows Build Instructions](#2-windows-build-instructions)
  - [Method A: Visual Studio 2022 (Recommended)](#method-a-visual-studio-2022-recommended)
  - [Method B: MinGW-w64 GCC (CLI)](#method-b-mingw-w64-gcc-cli)
- [3. Linux Build Instructions](#3-linux-build-instructions-nvidia-drivers)

---

## 1. Clone & Prerequisites

### A. Clone Repository
```bash
git clone https://github.com/Eonocross/Newton-XPBD-vulkan.git
cd Newton-XPBD-vulkan
```

### B. Hardware & Drivers
- **NVIDIA GPU** with up-to-date Game Ready or Studio Drivers:
  - Download: [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)
  - Verify Vulkan 1.3 capability in terminal:
    - **Windows**: `nvidia-smi` and `vulkaninfo --summary`
    - **Linux**: `nvidia-smi` and `vulkaninfo --summary`

### C. Development Toolchain
- **LunarG Vulkan SDK** (v1.3.250 or newer):
  - Includes official Vulkan headers and the `glslc` SPIR-V compiler.
  - Download: [vulkan.lunarg.com/sdk/home](https://vulkan.lunarg.com/sdk/home)
- **CMake** (v3.20 or newer):
  - Download: [cmake.org/download](https://cmake.org/download/)
- **Python (64-bit)**:
  - Must match the Python version embedded in your target Blender installation (e.g. Python 3.13 for Blender 5.2).
- **pybind11**:
  - Header library used for C++ Python bindings:
    ```bash
    pip install pybind11
    ```

---

## 2. Windows Build Instructions

### Method A: Visual Studio 2022 (Recommended)

- **Step A.1: Toolchain Setup**
  1. Install **Visual Studio 2022** with the **"Desktop development with C++"** workload enabled (MSVC v143 toolset with C++20).
  2. Install the **LunarG Vulkan SDK**. Ensure `%VULKAN_SDK%\Bin` is added to your environment `PATH`.

- **Step A.2: Compile Compute Shaders**  
  From Command Prompt or PowerShell in the repository root:
  ```bat
  cd shaders
  compile_shaders.bat
  ```
  This compiles all 17 `.comp` compute shaders via `glslc --target-env=vulkan1.3 -O` directly into `addons/modules/vkxpbd_shaders/*.spv`.

- **Step A.3: Build C++ Extension via CMake**
  ```powershell
  mkdir build
  cd build

  # Point to your Python directory (e.g. Blender's Python or system Python)
  cmake .. -G "Visual Studio 17 2022" -A x64 -DPython_ROOT_DIR="<path to python directory>"

  # Compile Release build
  cmake --build . --config Release
  ```

- **Step A.4: Deploy Binary**  
  Copy the output binary `build/Release/vkxpbd.pyd` into `addons/modules/vkxpbd.pyd`.

---

### Method B: MinGW-w64 GCC (CLI)

If compiling via MinGW-w64 GCC 13+ in PowerShell:
```powershell
g++ -O3 -shared -std=c++20 -DVK_NO_PROTOTYPES `
  src/bindings.cpp src/scene.cpp third_party/volk.c `
  -I"src" -I"third_party" `
  -I"$env:VULKAN_SDK/Include" `
  -I"<path to python directory>/include" `
  -I"<path to pybind11 include directory>" `
  -L"<path to python directory>/libs" `
  -lpython313 -static-libgcc -static-libstdc++ `
  "-Wl,-Bstatic" -lwinpthread "-Wl,-Bdynamic" `
  -o addons/modules/vkxpbd.pyd
```

---

## 3. Linux Build Instructions (NVIDIA Drivers)

Tested on Linux systems with proprietary NVIDIA drivers (`nvidia-driver-535+` / `nvidia-driver-550+`).

- **Step 3.1: Install Build Dependencies**
  ```bash
  sudo apt update
  sudo apt install -y build-essential cmake git \
      libvulkan-dev vulkan-tools glslc \
      python3-dev python3-pip python3-pybind11
  ```

- **Step 3.2: Compile Compute Shaders**
  ```bash
  mkdir -p addons/modules/vkxpbd_shaders

  for shader in shaders/*.comp; do
      base=$(basename "$shader" .comp)
      glslc --target-env=vulkan1.3 -O "$shader" -o "addons/modules/vkxpbd_shaders/${base}.spv"
  done
  ```

- **Step 3.3: Build C++ Extension**
  ```bash
  mkdir build && cd build

  # Configure CMake targeting your system/Blender Python
  cmake .. -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="<path to python directory>"

  # Build
  cmake --build . --parallel $(nproc)
  ```

- **Step 3.4: Deploy Shared Library**
  ```bash
  cp vkxpbd.so ../addons/modules/vkxpbd.so
  ```