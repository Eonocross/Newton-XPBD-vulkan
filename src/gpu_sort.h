// GPU radix sort (32-bit keys, 32-bit values), workgroup-parallel.
// Four 8-bit passes: histogram -> scan (bin prefix + per-WG scatter bases) -> scatter.
// Stable: preserves relative order of elements with equal keys.
#pragma once

#include "buffer.h"
#include "pipeline.h"
#include "context.h"

#include <cstdint>
#include <vector>

namespace vkx {

// Helper for descriptor bindings
std::vector<VkDescriptorSetLayoutBinding> gpu_sort_make_bindings(uint32_t n);

struct GpuSort {
    ComputePipeline histogram;
    ComputePipeline scan;
    ComputePipeline scatter;

    Buffer keys_a, vals_a;   // ping-pong pair (final result lands in keys_a/vals_a)
    Buffer keys_b, vals_b;
    Buffer global_hist;      // 256 bins * groups counts (u32), rewritten by scan
    Buffer scanned;          // 256 bin offsets (u32)

    uint32_t groups = 0;

    void create(Context* ctx, size_t n, VkDescriptorPool pool, PoolGuard& guard, const std::string& shader_dir = "shaders") {
        auto mb = [](uint32_t cnt) {
            std::vector<VkDescriptorSetLayoutBinding> v;
            for (uint32_t b = 0; b < cnt; b++)
                v.push_back({b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
            return v;
        };
        std::string prefix = shader_dir.empty() ? "" : (shader_dir + "/");
        histogram.create(ctx, prefix + "sort_histogram.spv", mb(4), 256);
        scan.create(ctx, prefix + "sort_scan.spv", mb(2), 256);
        scatter.create(ctx, prefix + "sort_scatter.spv", mb(7), 256);

        keys_a.create(ctx, n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        vals_a.create(ctx, n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        keys_b.create(ctx, n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        vals_b.create(ctx, n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        groups = uint32_t((n + 255) / 256);
        global_hist.create(ctx, size_t(groups) * 256 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        scanned.create(ctx, 256 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        hist_src = alloc_set(ctx, pool, histogram.set_layout);
        hist_a = alloc_set(ctx, pool, histogram.set_layout);
        hist_b = alloc_set(ctx, pool, histogram.set_layout);
        set_scan = alloc_set(ctx, pool, scan.set_layout);
        sc_src = alloc_set(ctx, pool, scatter.set_layout);
        sc_a = alloc_set(ctx, pool, scatter.set_layout);
        sc_b = alloc_set(ctx, pool, scatter.set_layout);
        for (VkDescriptorSet s : {hist_src, hist_a, hist_b, set_scan, sc_src, sc_a, sc_b})
            guard.sets.push_back(s);
    }

    void destroy() {
        histogram.destroy();
        scan.destroy();
        scatter.destroy();
        keys_a.destroy();
        vals_a.destroy();
        keys_b.destroy();
        vals_b.destroy();
        global_hist.destroy();
        scanned.destroy();
    }

    struct PC { uint32_t n; uint32_t shift; uint32_t groups; uint32_t pad; };

    // Record all 4 passes into an open command buffer. src_keys/src_vals hold
    // (cell_index, particle_id) on entry; after the passes the sorted pair is
    // in keys_a/vals_a (ping-pong lands there after an even number of passes).
    // One barrier after each pass.
    // Pre-binds fixed inputs for each pass at initialization time.
    void record(Context* ctx, VkCommandBuffer cmd, VkBuffer src_keys, VkBuffer src_vals, uint32_t n) {
        PC pc{n, 0, groups, 0};

        for (uint32_t pass = 0; pass < 4; pass++) {
            pc.shift = pass * 8;
            // Clear histogram before every pass
            vkCmdFillBuffer(cmd, global_hist.buffer, 0, global_hist.size, 0);
            barrier(cmd);
            // pass flow: 0: src->b, 1: b->a, 2: a->b, 3: b->a  (final in a)
            VkBuffer in_k = (pass == 0) ? src_keys : (pass == 1 ? keys_b.buffer : keys_a.buffer);
            VkBuffer in_v = (pass == 0) ? src_vals : (pass == 1 ? vals_b.buffer : vals_a.buffer);
            VkBuffer out_k = (pass % 2 == 0) ? keys_b.buffer : keys_a.buffer;
            VkBuffer out_v = (pass % 2 == 0) ? vals_b.buffer : vals_a.buffer;

            // histogram: select per-pass descriptor set
            VkDescriptorSet hs = (pass == 0) ? hist_src : (pass % 2 == 1 ? hist_b : hist_a);
            dispatch(cmd, histogram, hs, &pc);
            barrier(cmd);

            // scan: single workgroup prefix scan over bins
            dispatch(cmd, scan, set_scan, &pc, 1);
            barrier(cmd);

            // scatter: write sorted pairs
            VkDescriptorSet sc = (pass == 0) ? sc_src : (pass % 2 == 1 ? sc_b : sc_a);
            dispatch(cmd, scatter, sc, &pc);
            barrier(cmd);
        }
    }

    // pre-bind all per-pass descriptor sets (call once after create, before any
    // recording; safe because these bindings never change afterwards)
    void bind_static(Context* ctx, VkBuffer src_keys, VkBuffer src_vals) {
        auto wb = [&](VkDescriptorSet s, uint32_t binding, VkBuffer b) {
            write_set(ctx, s, binding, b, keys_a.size);
        };
        // histogram variants: binding0 = input keys; 1..3 fixed
        for (int v = 0; v < 3; v++) {
            VkDescriptorSet hs = (v == 0) ? hist_src : (v == 1 ? hist_b : hist_a);
            VkBuffer k = (v == 0) ? src_keys : (v == 1 ? keys_b.buffer : keys_a.buffer);
            wb(hs, 0, k);
            write_set(ctx, hs, 1, global_hist.buffer, global_hist.size);
            write_set(ctx, hs, 2, scanned.buffer, scanned.size);
            write_set(ctx, hs, 3, keys_a.buffer, keys_a.size); // pad binding (unused)
        }
        // scan: fixed
        write_set(ctx, set_scan, 0, global_hist.buffer, global_hist.size);
        write_set(ctx, set_scan, 1, scanned.buffer, scanned.size);
        // scatter variants: 0=in_k,1=in_v, 3=out_k,4=out_v, 5=wg bases, 6 pad
        auto bind_sc = [&](VkDescriptorSet s, VkBuffer ik, VkBuffer iv, VkBuffer ok, VkBuffer ov) {
            wb(s, 0, ik); wb(s, 1, iv);
            write_set(ctx, s, 2, scanned.buffer, scanned.size);
            wb(s, 3, ok); wb(s, 4, ov);
            write_set(ctx, s, 5, global_hist.buffer, global_hist.size);
            write_set(ctx, s, 6, keys_a.buffer, keys_a.size); // pad binding (unused)
        };
        bind_sc(sc_src, src_keys, src_vals, keys_b.buffer, vals_b.buffer);
        bind_sc(sc_b, keys_b.buffer, vals_b.buffer, keys_a.buffer, vals_a.buffer);
        bind_sc(sc_a, keys_a.buffer, vals_a.buffer, keys_b.buffer, vals_b.buffer);
    }

    // Debug: record a SINGLE pass (0..3). Caller submits + waits between calls.
    void record_pass(Context* ctx, VkCommandBuffer cmd, uint32_t pass, uint32_t n) {
        PC pc{n, pass * 8, groups, 0};
        vkCmdFillBuffer(cmd, global_hist.buffer, 0, global_hist.size, 0);
        barrier(cmd);
        VkDescriptorSet hs = (pass == 0) ? hist_src : (pass % 2 == 1 ? hist_b : hist_a);
        dispatch(cmd, histogram, hs, &pc);
        barrier(cmd);
        dispatch(cmd, scan, set_scan, &pc, 1);
        barrier(cmd);
        VkDescriptorSet sc = (pass == 0) ? sc_src : (pass % 2 == 1 ? sc_b : sc_a);
        dispatch(cmd, scatter, sc, &pc);
        barrier(cmd);
    }

private:
    void dispatch(VkCommandBuffer cmd, ComputePipeline& p, VkDescriptorSet set, void* pc,
                  uint32_t n_groups_override = 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), pc);
        vkCmdDispatch(cmd, n_groups_override ? n_groups_override : groups, 1, 1);
    }

    static void barrier(VkCommandBuffer cmd) {
        // Pipeline barrier covering compute and transfer memory access.
        VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                          VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

public:
    VkDescriptorSet set_scan = VK_NULL_HANDLE;
    VkDescriptorSet hist_src = VK_NULL_HANDLE, hist_a = VK_NULL_HANDLE, hist_b = VK_NULL_HANDLE;
    VkDescriptorSet sc_src = VK_NULL_HANDLE, sc_a = VK_NULL_HANDLE, sc_b = VK_NULL_HANDLE;
};

} // namespace vkx
