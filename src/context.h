// Vulkan context initialization: instance, physical device, logical device, queue, and command pool.
#pragma once

#define VK_ENABLE_BETA_EXTENSIONS
#include "volk.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#define VK_CHECK(x)                                                                                \
    do {                                                                                           \
        VkResult _r = (x);                                                                         \
        if (_r != VK_SUCCESS) {                                                                    \
            char _msg[256];                                                                        \
            snprintf(_msg, sizeof(_msg), "Vulkan error %d at %s:%d", (int)_r, __FILE__, __LINE__); \
            throw std::runtime_error(_msg);                                                        \
        }                                                                                          \
    } while (0)

namespace vkx {

struct Context {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t compute_queue_family = 0;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE; // primary, rerecorded per use
    VkPhysicalDeviceProperties props{};
    bool supports_subgroup_size_32 = false;
    bool supports_float_controls = false;

    ~Context() { destroy(); }

    static std::vector<std::string> enumerate_devices() {
        volkInitialize();
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "vkxpbd";
        app.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;

        VkInstance temp_instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&ci, nullptr, &temp_instance) != VK_SUCCESS) {
            return {};
        }
        volkLoadInstance(temp_instance);

        uint32_t nd = 0;
        vkEnumeratePhysicalDevices(temp_instance, &nd, nullptr);
        std::vector<VkPhysicalDevice> devs(nd);
        if (nd > 0) {
            vkEnumeratePhysicalDevices(temp_instance, &nd, devs.data());
        }

        std::vector<std::string> names;
        names.reserve(nd);
        for (uint32_t i = 0; i < nd; i++) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(devs[i], &p);
            names.push_back(std::string(p.deviceName));
        }

        return names;
    }

    void init(bool enable_validation = false, int device_index = -1, bool compat_mode = false) {
        volkInitialize();

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "vkxpbd";
        app.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ci.pApplicationInfo = &app;

        const char* layers[1] = {"VK_LAYER_KHRONOS_validation"};
        bool layer_available = false;
        if (enable_validation) {
            uint32_t layer_count = 0;
            vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
            if (layer_count > 0) {
                std::vector<VkLayerProperties> available(layer_count);
                vkEnumerateInstanceLayerProperties(&layer_count, available.data());
                for (const auto& l : available) {
                    if (strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                        layer_available = true;
                        break;
                    }
                }
            }
            if (layer_available) {
                ci.enabledLayerCount = 1;
                ci.ppEnabledLayerNames = layers;
            } else {
                printf("[Newton Vulkan] Notice: Vulkan validation layers requested but not installed. Continuing simulation without validation.\n");
            }
        }
        VK_CHECK(vkCreateInstance(&ci, nullptr, &instance));
        volkLoadInstance(instance);

        uint32_t nd = 0;
        vkEnumeratePhysicalDevices(instance, &nd, nullptr);
        std::vector<VkPhysicalDevice> devs(nd ? nd : 1);
        vkEnumeratePhysicalDevices(instance, &nd, devs.data());
        if (nd == 0) throw std::runtime_error("no Vulkan physical devices");

        uint32_t chosen_idx = 0;
        if (device_index >= 0 && (uint32_t)device_index < nd) {
            chosen_idx = (uint32_t)device_index;
        } else {
            for (uint32_t i = 0; i < nd; i++) {
                VkPhysicalDeviceProperties p{};
                vkGetPhysicalDeviceProperties(devs[i], &p);
                if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    chosen_idx = i;
                    if (p.vendorID == 0x10DE) {
                        break;
                    }
                }
            }
        }

        physical_device = devs[chosen_idx];
        vkGetPhysicalDeviceProperties(physical_device, &props);

        // find a compute queue (graphics|compute preferred, plain compute ok)
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &nq, qs.data());
        for (uint32_t i = 0; i < nq; i++) {
            if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_family = i;
                if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) break; // prefer combined
            }
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = compute_queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;

        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.synchronization2 = VK_TRUE;

        // Check for VK_EXT_subgroup_size_control / Vulkan 1.3 subgroupSizeControl
        VkPhysicalDeviceSubgroupSizeControlFeaturesEXT ssc{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
        VkPhysicalDeviceSubgroupSizeControlPropertiesEXT ssp{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};
        
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &ssp;
        vkGetPhysicalDeviceProperties2(physical_device, &p2);

        VkPhysicalDeviceFeatures2 f2_check{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2_check.pNext = &ssc;
        vkGetPhysicalDeviceFeatures2(physical_device, &f2_check);

        if (ssc.subgroupSizeControl && (ssp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
            ssp.minSubgroupSize <= 32 && ssp.maxSubgroupSize >= 32) {
            ssc.subgroupSizeControl = VK_TRUE;
            supports_subgroup_size_32 = true;
            f13.pNext = &ssc;
        }

        printf("[Newton Vulkan] Device [%u] Subgroup Query: control=%d, reqStages=0x%x, min=%u, max=%u, active32=%d\n",
               chosen_idx, (int)ssc.subgroupSizeControl, (unsigned)ssp.requiredSubgroupSizeStages,
               ssp.minSubgroupSize, ssp.maxSubgroupSize, (int)supports_subgroup_size_32);

        std::vector<const char*> dev_exts;
        if (supports_subgroup_size_32) {
            dev_exts.push_back("VK_EXT_subgroup_size_control");
        }

        // Check for VK_KHR_shader_float_controls
        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> available_exts(ext_count);
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, available_exts.data());
        for (const auto& e : available_exts) {
            if (strcmp(e.extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME) == 0) {
                supports_float_controls = true;
                dev_exts.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
                break;
            }
        }

        VkPhysicalDeviceFloatControlsProperties fc_props{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        if (supports_float_controls) {
            VkPhysicalDeviceProperties2 p2_fc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2_fc.pNext = &fc_props;
            vkGetPhysicalDeviceProperties2(physical_device, &p2_fc);
        }

        const char* atomics_ext = "VK_EXT_shader_atomic_float";
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};

        if (!compat_mode) {
            VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            f2.pNext = &af;
            vkGetPhysicalDeviceFeatures2(physical_device, &f2);
            if (!af.shaderBufferFloat32AtomicAdd)
                throw std::runtime_error("device lacks shaderBufferFloat32AtomicAdd (required)");

            af.pNext = f13.pNext;
            f13.pNext = &af;
            dev_exts.push_back(atomics_ext);
        }

        dci.pNext = &f13;
        dci.enabledExtensionCount = (uint32_t)dev_exts.size();
        dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();

        VK_CHECK(vkCreateDevice(physical_device, &dci, nullptr, &device));
        volkLoadDevice(device);

        vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = compute_queue_family;
        VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &command_pool));

        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = command_pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));

        printf("[Newton Vulkan] Using device [%u]: %s (queue family %u%s%s%s)\n",
               chosen_idx, props.deviceName, compute_queue_family,
               compat_mode ? ", compatibility mode" : "",
               supports_subgroup_size_32 ? ", subgroup_size=32" : "",
               supports_float_controls ? ", float_controls" : "");
    }

    void destroy() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (command_pool) {
                vkDestroyCommandPool(device, command_pool, nullptr);
                command_pool = VK_NULL_HANDLE;
            }
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        // Do NOT destroy instance: Blender itself uses Vulkan for UI / viewport rendering,
        // and vkDestroyInstance tears down NVIDIA's process-wide driver dispatch tables,
        // crashing Blender's DrvPresentBuffers presentation loop.
        instance = VK_NULL_HANDLE;
    }
};

} // namespace vkx
