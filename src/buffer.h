// SSBO helpers: device-local storage buffers with staging upload and download.
// Storage buffers mirror Structure-of-Arrays (SoA) layout.
#pragma once

#include "context.h"

#include <cstring>
#include <vector>

namespace vkx {

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    Context* ctx = nullptr;

    // movable, not copyable: transfers handle ownership to prevent double-free
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept { *this = std::move(o); }
    Buffer& operator=(Buffer&& o) noexcept {
        destroy();
        buffer = o.buffer;
        memory = o.memory;
        size = o.size;
        ctx = o.ctx;
        staging_buf = o.staging_buf;
        staging_mem = o.staging_mem;
        staging_map = o.staging_map;
        staging_cap = o.staging_cap;

        o.buffer = VK_NULL_HANDLE;
        o.memory = VK_NULL_HANDLE;
        o.size = 0;
        o.staging_buf = VK_NULL_HANDLE;
        o.staging_mem = VK_NULL_HANDLE;
        o.staging_map = nullptr;
        o.staging_cap = 0;
        return *this;
    }

    ~Buffer() { destroy(); } // null-safe: destroy() clears handles after freeing

    void create(Context* c, VkDeviceSize bytes, VkBufferUsageFlags usage) {
        ctx = c;
        size = bytes;
        if (bytes == 0) return;

        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes;
        bi.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(c->device, &bi, nullptr, &buffer));

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(c->device, buffer, &req);

        uint32_t mem_type = find_memory_type(c, req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = mem_type;
        VK_CHECK(vkAllocateMemory(c->device, &ai, nullptr, &memory));
        VK_CHECK(vkBindBufferMemory(c->device, buffer, memory, 0));
    }

    static uint32_t find_memory_type(Context* c, uint32_t type_bits, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(c->physical_device, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("no suitable memory type");
    }

    // upload host data (may be nullptr to leave undefined / rely on later fill)
    void upload(const void* data, VkDeviceSize bytes) {
        if (bytes == 0) return;
        Staging s = acquire_staging(bytes);

        memcpy(s.map, data, bytes);
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        r.memory = s.mem;
        r.size = VK_WHOLE_SIZE;
        vkFlushMappedMemoryRanges(ctx->device, 1, &r); // NOT coherent memory

        copy_buffer(s.buf, buffer, bytes);
        release_staging();
    }

    // download whole buffer to host
    std::vector<uint8_t> download(VkDeviceSize bytes) {
        Staging s = acquire_staging(bytes);
        copy_buffer(buffer, s.buf, bytes);

        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        r.memory = s.mem;
        r.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(ctx->device, 1, &r); // Invalidate non-coherent memory before host read

        std::vector<uint8_t> out((size_t)bytes);
        memcpy(out.data(), s.map, (size_t)bytes);
        release_staging();
        return out;
    }

    void fill_zero(VkDeviceSize bytes) {
        // record a vkCmdFillBuffer on the shared cmd buffer and submit
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(ctx->cmd, &bi));
        vkCmdFillBuffer(ctx->cmd, buffer, 0, bytes, 0);
        VK_CHECK(vkEndCommandBuffer(ctx->cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &ctx->cmd;
        VK_CHECK(vkQueueSubmit(ctx->compute_queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(ctx->compute_queue));
        vkResetCommandBuffer(ctx->cmd, 0);
    }

    // device-local buffer-to-buffer copy (whole range)
    void copy_from(const Buffer& src, VkDeviceSize bytes) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(ctx->cmd, &bi));
        VkBufferCopy c{0, 0, bytes};
        vkCmdCopyBuffer(ctx->cmd, src.buffer, buffer, 1, &c);
        VK_CHECK(vkEndCommandBuffer(ctx->cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &ctx->cmd;
        VK_CHECK(vkQueueSubmit(ctx->compute_queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(ctx->compute_queue));
        vkResetCommandBuffer(ctx->cmd, 0);
    }

    void free_persistent() {
        if (staging_buf && ctx && ctx->device) {
            vkDestroyBuffer(ctx->device, staging_buf, nullptr);
            vkFreeMemory(ctx->device, staging_mem, nullptr);
            staging_buf = VK_NULL_HANDLE;
            staging_mem = VK_NULL_HANDLE;
            staging_map = nullptr;
            staging_cap = 0;
        }
    }

    void destroy() {
        free_persistent();
        if (buffer && ctx && ctx->device) vkDestroyBuffer(ctx->device, buffer, nullptr);
        if (memory && ctx && ctx->device) vkFreeMemory(ctx->device, memory, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        size = 0;
    }

  private:
    // legacy per-call staging handles (unused since persistent staging)

    // PERSISTENT staging buffer (grow-only): create+allocate+free of an 80 MB
    // HOST_VISIBLE allocation per frame cost ~450 ms at 5M particles (driver-
    // side allocation path), dwarfing the actual PCIe copy (~6 ms). Caching it
    // makes every upload/download a map-free memcpy + copy + submit.
    struct Staging {
        VkBuffer buf;
        VkDeviceMemory mem;
        void* map;
    };
    VkBuffer staging_buf = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem = VK_NULL_HANDLE;
    void* staging_map = nullptr;
    VkDeviceSize staging_cap = 0;

    Staging acquire_staging(VkDeviceSize bytes) {
        if (bytes > staging_cap) {
            if (staging_buf) {
                vkDestroyBuffer(ctx->device, staging_buf, nullptr);
                vkFreeMemory(ctx->device, staging_mem, nullptr);
                staging_buf = VK_NULL_HANDLE;
            }
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VK_CHECK(vkCreateBuffer(ctx->device, &bi, nullptr, &staging_buf));
            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(ctx->device, staging_buf, &req);
            // prefer HOST_CACHED: plain HOST_VISIBLE can be write-combined, and
            // CPU READS from WC memory run ~25x slower (~450 ms per 80 MB read
            // at 5M particles — measured). Cached host memory reads at RAM speed.
            uint32_t mt = UINT32_MAX;
            {
                VkPhysicalDeviceMemoryProperties mp;
                vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &mp);
                for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
                    if ((req.memoryTypeBits & (1u << i)) &&
                        (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                        (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) {
                        mt = i;
                        break;
                    }
                }
                if (mt == UINT32_MAX)
                    mt = find_memory_type(ctx, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            }
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mt;
            VK_CHECK(vkAllocateMemory(ctx->device, &ai, nullptr, &staging_mem));
            VK_CHECK(vkBindBufferMemory(ctx->device, staging_buf, staging_mem, 0));
            VK_CHECK(vkMapMemory(ctx->device, staging_mem, 0, req.size, 0, &staging_map));
            staging_cap = req.size;
        }
        return {staging_buf, staging_mem, staging_map};
    }

    void release_staging() {} // kept for symmetry; buffer is persistent now

    void copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize bytes) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(ctx->cmd, &bi));
        VkBufferCopy c{0, 0, bytes};
        vkCmdCopyBuffer(ctx->cmd, src, dst, 1, &c);
        VK_CHECK(vkEndCommandBuffer(ctx->cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &ctx->cmd;
        VK_CHECK(vkQueueSubmit(ctx->compute_queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(ctx->compute_queue));
        vkResetCommandBuffer(ctx->cmd, 0);
    }

};

} // namespace vkx
