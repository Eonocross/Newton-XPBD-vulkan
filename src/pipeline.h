// Compute pipeline creation from a compiled SPIR-V file.
// One descriptor set layout per shader, declared explicitly by the caller
// (mirrors the Warp kernel signature order — fidelity over cleverness).
#pragma once

#include "context.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

namespace vkx {

inline std::vector<char> read_spirv(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open SPIR-V: " + path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz % 4 != 0) throw std::runtime_error("SPIR-V size not multiple of 4: " + path);
    std::vector<char> code((size_t)sz);
    if (fread(code.data(), 1, (size_t)sz, f) != (size_t)sz)
        throw std::runtime_error("short read: " + path);
    fclose(f);
    return code;
}

struct ComputePipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    Context* ctx = nullptr;

    ~ComputePipeline() { destroy(); } // null-safe

    // bindings: (binding, dtype size unused here) — just count of storage buffers
    // plus optional uniform buffer at binding 0..n; caller passes full list.
    void create(Context* c, const std::string& spirv_path,
                const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                size_t push_constant_size) {
        ctx = c;
        auto code = read_spirv(spirv_path);

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = code.size();
        smci.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module;
        VK_CHECK(vkCreateShaderModule(c->device, &smci, nullptr, &module));

        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = (uint32_t)bindings.size();
        lci.pBindings = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(c->device, &lci, nullptr, &set_layout));

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pc.size = (uint32_t)push_constant_size;
        pc.offset = 0;

        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &set_layout;
        if (push_constant_size > 0) {
            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pc;
        }
        VK_CHECK(vkCreatePipelineLayout(c->device, &plci, nullptr, &layout));

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";

        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci.stage = stage;
        pci.layout = layout;
        VK_CHECK(vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline));

        vkDestroyShaderModule(c->device, module, nullptr);
    }

    void destroy() {
        if (ctx && ctx->device) {
            if (pipeline) { vkDestroyPipeline(ctx->device, pipeline, nullptr); pipeline = VK_NULL_HANDLE; }
            if (layout) { vkDestroyPipelineLayout(ctx->device, layout, nullptr); layout = VK_NULL_HANDLE; }
            if (set_layout) { vkDestroyDescriptorSetLayout(ctx->device, set_layout, nullptr); set_layout = VK_NULL_HANDLE; }
        }
    }
};

// Descriptor pool creation for compute pipeline storage buffers
inline VkDescriptorPool make_pool(Context* c, uint32_t max_sets, uint32_t max_storage) {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_storage};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets = max_sets;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    VkDescriptorPool pool;
    VK_CHECK(vkCreateDescriptorPool(c->device, &pci, nullptr, &pool));
    return pool;
}

// RAII guard: destroys pool + its sets at scope exit
struct PoolGuard {
    Context* c;
    VkDescriptorPool p;
    std::vector<VkDescriptorSet> sets;
    PoolGuard(Context* c, VkDescriptorPool p) : c(c), p(p) {}
    ~PoolGuard() {
        if (p) {
            for (auto s : sets) vkFreeDescriptorSets(c->device, p, 1, &s);
            vkDestroyDescriptorPool(c->device, p, nullptr);
        }
    }
};

inline VkDescriptorSet alloc_set(Context* c, VkDescriptorPool pool, VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    VkDescriptorSet set;
    VK_CHECK(vkAllocateDescriptorSets(c->device, &ai, &set));
    return set;
}

inline void write_set(Context* c, VkDescriptorSet set, uint32_t binding, VkBuffer buffer,
                      VkDeviceSize size) {
    VkDescriptorBufferInfo info{buffer, 0, size};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &info;
    vkUpdateDescriptorSets(c->device, 1, &w, 0, nullptr);
}

} // namespace vkx
