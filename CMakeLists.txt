cmake_minimum_required(VERSION 3.20)
project(vkxpbd C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# vkxpbd Python C++ extension module
pybind11_add_module(vkxpbd MODULE
    src/scene.cpp
    src/bindings.cpp
    third_party/volk.c
)

target_include_directories(vkxpbd PRIVATE
    src
    third_party
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vkxpbd PRIVATE
    ${CMAKE_DL_LIBS}
)

if(WIN32)
    set_target_properties(vkxpbd PROPERTIES
        PREFIX ""
        SUFFIX ".pyd"
    )
endif()
