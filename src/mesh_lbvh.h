#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdio>

// Linear Bounding Volume Hierarchy (LBVH) builder using Morton codes
struct MeshLbvhBuilder {
    static inline uint32_t part1by2(uint32_t n) {
        n &= 0x000003ff;
        n = (n ^ (n << 16)) & 0xff0000ff;
        n = (n ^ (n << 8))  & 0x0300f00f;
        n = (n ^ (n << 4))  & 0x030c30c3;
        n = (n ^ (n << 2))  & 0x09249249;
        return n;
    }

    static inline uint32_t morton3_1024(float x, float y, float z) {
        int ix = int(x * 1024.0f);
        int iy = int(y * 1024.0f);
        int iz = int(z * 1024.0f);
        uint32_t ux = uint32_t(std::max(0, std::min(1023, ix)));
        uint32_t uy = uint32_t(std::max(0, std::min(1023, iy)));
        uint32_t uz = uint32_t(std::max(0, std::min(1023, iz)));
        return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
    }

    static inline int clz64(uint64_t x) {
        if (x == 0) return 64;
#if defined(_MSC_VER)
        unsigned long idx;
        _BitScanReverse64(&idx, x);
        return 63 - int(idx);
#elif defined(__GNUC__) || defined(__clang__)
        return __builtin_clzll(x);
#else
        int count = 0;
        while ((x & 0x8000000000000000ULL) == 0) {
            count++;
            x <<= 1;
        }
        return count;
#endif
    }

    static void build(
        const std::vector<float>& pts,
        const std::vector<uint32_t>& tris,
        std::vector<PackedNodeHalf>& lowers,
        std::vector<PackedNodeHalf>& uppers,
        std::vector<uint32_t>& primitive_indices,
        uint32_t& root
    ) {
        const uint32_t n_items = uint32_t(tris.size() / 3);
        assert(n_items > 0);

        struct Bounds {
            float lo[3], hi[3];
        };
        std::vector<Bounds> item_bounds(n_items);
        float total_lo[3] = {1e30f, 1e30f, 1e30f};
        float total_hi[3] = {-1e30f, -1e30f, -1e30f};

        for (uint32_t t = 0; t < n_items; t++) {
            Bounds& b = item_bounds[t];
            b.lo[0] = b.hi[0] = pts[tris[t*3+0]*3+0];
            b.lo[1] = b.hi[1] = pts[tris[t*3+0]*3+1];
            b.lo[2] = b.hi[2] = pts[tris[t*3+0]*3+2];
            for (int v = 1; v < 3; v++) {
                const float* p = &pts[tris[t*3+v]*3];
                for (int a = 0; a < 3; a++) {
                    b.lo[a] = std::min(b.lo[a], p[a]);
                    b.hi[a] = std::max(b.hi[a], p[a]);
                }
            }
            for (int a = 0; a < 3; a++) {
                total_lo[a] = std::min(total_lo[a], b.lo[a]);
                total_hi[a] = std::max(total_hi[a], b.hi[a]);
            }
        }

        float total_inv_edges[3];
        for (int a = 0; a < 3; a++) {
            float e = (total_hi[a] - total_lo[a]) + 0.0001f;
            total_inv_edges[a] = 1.0f / e;
        }

        struct MortonItem {
            uint64_t key;
            uint32_t index;
        };
        std::vector<MortonItem> items(n_items);
        for (uint32_t t = 0; t < n_items; t++) {
            float c[3];
            for (int a = 0; a < 3; a++) c[a] = 0.5f * (item_bounds[t].lo[a] + item_bounds[t].hi[a]);
            float loc[3];
            for (int a = 0; a < 3; a++) loc[a] = (c[a] - total_lo[a]) * total_inv_edges[a];
            uint32_t m = morton3_1024(loc[0], loc[1], loc[2]);
            items[t] = {uint64_t(m), t};
        }

        // Radix sort items by Morton key (stable sort)
        std::stable_sort(items.begin(), items.end(), [](const MortonItem& a, const MortonItem& b) {
            return a.key < b.key;
        });

        primitive_indices.resize(n_items);
        std::vector<uint64_t> keys(n_items);
        for (uint32_t t = 0; t < n_items; t++) {
            keys[t] = items[t].key;
            primitive_indices[t] = items[t].index;
        }

        std::vector<int> deltas(n_items > 1 ? n_items - 1 : 1);
        for (uint32_t i = 0; i + 1 < n_items; i++) {
            deltas[i] = clz64(keys[i] ^ keys[i+1]);
        }

        const uint32_t max_nodes = 2 * n_items;
        lowers.assign(max_nodes, PackedNodeHalf{});
        uppers.assign(max_nodes, PackedNodeHalf{});

        std::vector<int> parents(max_nodes, -1);
        std::vector<int> num_children(max_nodes, 0);
        std::vector<int> range_lefts(max_nodes, 0);
        std::vector<int> range_rights(max_nodes, 0);

        for (uint32_t i = 0; i < n_items; i++) {
            uint32_t item = primitive_indices[i];
            lowers[i].x = item_bounds[item].lo[0];
            lowers[i].y = item_bounds[item].lo[1];
            lowers[i].z = item_bounds[item].lo[2];
            lowers[i].packed = i | (1u << 30); // leaf

            uppers[i].x = item_bounds[item].hi[0];
            uppers[i].y = item_bounds[item].hi[1];
            uppers[i].z = item_bounds[item].hi[2];
            uppers[i].packed = i + 1;

            range_lefts[i] = int(i);
            range_rights[i] = int(i);
        }

        const int internal_offset = int(n_items);
        for (uint32_t i = 0; i < n_items; i++) {
            int curr = int(i);
            while (true) {
                int left = range_lefts[curr];
                int right = range_rights[curr];
                if (left == 0 && right == int(n_items - 1)) {
                    root = uint32_t(curr);
                    parents[curr] = -1;
                    break;
                }

                bool parent_right = false;
                if (left == 0) {
                    parent_right = true;
                } else {
                    if (right != int(n_items - 1) && deltas[right] >= deltas[left - 1]) {
                        if (deltas[right] == deltas[left - 1]) {
                            parent_right = bool((primitive_indices[left - 1] % 2) ^ (primitive_indices[right] % 2));
                        } else {
                            parent_right = true;
                        }
                    } else {
                        parent_right = false;
                    }
                }

                int parent;
                if (parent_right) {
                    parent = right + internal_offset;
                    parents[curr] = parent;
                    lowers[parent].packed = uint32_t(curr); // left child
                    range_lefts[parent] = left;
                    num_children[parent]++;
                } else {
                    parent = left + internal_offset - 1;
                    parents[curr] = parent;
                    uppers[parent].packed = uint32_t(curr); // right child
                    range_rights[parent] = right;
                    num_children[parent]++;
                }

                if (num_children[parent] == 1) {
                    break; // first child done
                } else {
                    int lc = int(lowers[parent].packed & 0x3FFFFFFFu);
                    int rc = int(uppers[parent].packed & 0x3FFFFFFFu);
                    lowers[parent].x = std::min(lowers[lc].x, lowers[rc].x);
                    lowers[parent].y = std::min(lowers[lc].y, lowers[rc].y);
                    lowers[parent].z = std::min(lowers[lc].z, lowers[rc].z);
                    uppers[parent].x = std::max(uppers[lc].x, uppers[rc].x);
                    uppers[parent].y = std::max(uppers[lc].y, uppers[rc].y);
                    uppers[parent].z = std::max(uppers[lc].z, uppers[rc].z);
                    curr = parent;
                }
            }
        }

        // mark_packed_leaf_nodes (leaf_size = 4)
        const int leaf_size = 4;
        for (uint32_t node_index = 0; node_index < max_nodes; node_index++) {
            if (num_children[node_index] > 0 || node_index < n_items) {
                int depth = 1;
                int p = parents[node_index];
                while (p != -1) {
                    p = parents[p];
                    depth++;
                }
                int left = range_lefts[node_index];
                int right = range_rights[node_index] + 1;
                if (right - left <= leaf_size || depth >= 32) {
                    lowers[node_index].packed = uint32_t(left) | (1u << 30);
                    uppers[node_index].packed = uint32_t(right);
                }
            }
        }
    }
};
