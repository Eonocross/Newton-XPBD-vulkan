// CPU BVH builder over triangle meshes.
// Builds a binary BVH over triangles with packed 16-byte node halves.
#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdio>

namespace vkx {

#pragma pack(push, 1)
struct PackedNodeHalf {
    float x, y, z;
    uint32_t packed;  // lower: i (30 bits) | leaf flag (2 bits); upper: right index
};
#pragma pack(pop)
static_assert(sizeof(PackedNodeHalf) == 16, "packed node must be 16 bytes");

#include "mesh_lbvh.h"

struct MeshBvh {
    // inputs
    std::vector<float> points;          // 3 * n_verts
    std::vector<uint32_t> indices;      // 3 * n_tris
    // bvh
    std::vector<PackedNodeHalf> lowers, uppers;
    std::vector<uint32_t> primitive_indices;
    uint32_t root = 0;
    static constexpr uint32_t LEAF_SIZE = 4;

    struct Tri {
        uint32_t id;
        float cx, cy, cz;   // centroid
        float lo[3], hi[3]; // bounds
    };

    void build(const std::vector<float>& pts, const std::vector<uint32_t>& tris) {
        points = pts;
        indices = tris;
        const uint32_t n_tris = uint32_t(tris.size() / 3);
        assert(n_tris > 0);

        std::vector<Tri> tris_info(n_tris);
        for (uint32_t t = 0; t < n_tris; t++) {
            Tri& T = tris_info[t];
            T.id = t;
            T.lo[0] = T.hi[0] = pts[tris[t*3+0]*3+0];
            T.lo[1] = T.hi[1] = pts[tris[t*3+0]*3+1];
            T.lo[2] = T.hi[2] = pts[tris[t*3+0]*3+2];
            for (int v = 1; v < 3; v++) {
                const float* p = &pts[tris[t*3+v]*3];
                for (int a = 0; a < 3; a++) {
                    T.lo[a] = std::min(T.lo[a], p[a]);
                    T.hi[a] = std::max(T.hi[a], p[a]);
                }
            }
            T.cx = 0.5f*(T.lo[0]+T.hi[0]);
            T.cy = 0.5f*(T.lo[1]+T.hi[1]);
            T.cz = 0.5f*(T.lo[2]+T.hi[2]);
        }

        primitive_indices.resize(n_tris);
        for (uint32_t t = 0; t < n_tris; t++) primitive_indices[t] = t;

        MeshLbvhBuilder::build(pts, tris, lowers, uppers, primitive_indices, root);
    }

private:
    uint32_t make_leaf(std::vector<Tri>& T, uint32_t lo, uint32_t hi) {
        uint32_t node = uint32_t(lowers.size());
        PackedNodeHalf l{}, u{};
        // bounds over the leaf's primitives
        float lof[3] = {1e30f,1e30f,1e30f}, hif[3] = {-1e30f,-1e30f,-1e30f};
        for (uint32_t t = lo; t < hi; t++) {
            const Tri& tr = T[t];
            for (int a = 0; a < 3; a++) {
                lof[a] = std::min(lof[a], tr.lo[a]);
                hif[a] = std::max(hif[a], tr.hi[a]);
            }
        }
        l.x = lof[0]; l.y = lof[1]; l.z = lof[2];
        u.x = hif[0]; u.y = hif[1]; u.z = hif[2];
        // re-map the primitive range so leaves own a contiguous slice
        // (we sort T ids into position as we recurse)
        l.packed = lo | (1u << 30);
        u.packed = hi;
        // map the leaf's slice of the SORTED triangle array back to original
        // triangle ids — the shader reads prim[pc] to get the real triangle
        for (uint32_t t = lo; t < hi; t++) primitive_indices[t] = T[t].id;
        lowers.push_back(l); uppers.push_back(u);
        return node;
    }

    // returns node index; sorts T[lo..hi) so leaves own contiguous slices
    uint32_t build_range(std::vector<Tri>& T, uint32_t lo, uint32_t hi) {
        if (hi - lo <= LEAF_SIZE) {
            return make_leaf(T, lo, hi);
        }
        // bounds of centroids
        float clo[3] = {1e30f,1e30f,1e30f}, chi[3] = {-1e30f,-1e30f,-1e30f};
        for (uint32_t t = lo; t < hi; t++) {
            clo[0]=std::min(clo[0],T[t].cx); chi[0]=std::max(chi[0],T[t].cx);
            clo[1]=std::min(clo[1],T[t].cy); chi[1]=std::max(chi[1],T[t].cy);
            clo[2]=std::min(clo[2],T[t].cz); chi[2]=std::max(chi[2],T[t].cz);
        }
        int axis = 0;
        float ext = chi[0]-clo[0];
        if (chi[1]-clo[1] > ext) { axis = 1; ext = chi[1]-clo[1]; }
        if (chi[2]-clo[2] > ext) { axis = 2; ext = chi[2]-clo[2]; }
        uint32_t mid = lo + (hi - lo) / 2;
        if (ext > 0.0f) {
            std::nth_element(T.begin()+lo, T.begin()+mid, T.begin()+hi,
                [axis](const Tri& a, const Tri& b) {
                    float ka = (axis==0)? a.cx : (axis==1)? a.cy : a.cz;
                    float kb = (axis==0)? b.cx : (axis==1)? b.cy : b.cz;
                    return ka < kb;
                });
        }
        uint32_t left = build_range(T, lo, mid);
        uint32_t right = build_range(T, mid, hi);
        uint32_t node = uint32_t(lowers.size());
        PackedNodeHalf l{}, u{};
        l.x = lowers[left].x < lowers[right].x ? lowers[left].x : lowers[right].x;
        l.y = lowers[left].y < lowers[right].y ? lowers[left].y : lowers[right].y;
        l.z = lowers[left].z < lowers[right].z ? lowers[left].z : lowers[right].z;
        u.x = uppers[left].x > uppers[right].x ? uppers[left].x : uppers[right].x;
        u.y = uppers[left].y > uppers[right].y ? uppers[left].y : uppers[right].y;
        u.z = uppers[left].z > uppers[right].z ? uppers[left].z : uppers[right].z;
        l.packed = left;
        u.packed = right;
        lowers.push_back(l); uppers.push_back(u);
        return node;
    }

public:
    // Re-expand all node AABBs after vertex positions changed (rigid animation).
    // The tree topology (prim_indices, left/right child links) is unchanged;
    // only the xyz min/max of each node is updated. Call after updating points[].
    void refit_recursive(uint32_t ni) {
        uint32_t lower_packed = lowers[ni].packed;
        uint32_t is_leaf = (lower_packed >> 30u) & 0x3u;
        if (is_leaf != 0u) {
            // Leaf: recompute bounds from actual vertex positions
            uint32_t prim_lo = lower_packed & 0x3FFFFFFFu;
            uint32_t prim_hi = uppers[ni].packed & 0x3FFFFFFFu;
            float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
            for (uint32_t pi = prim_lo; pi < prim_hi; pi++) {
                uint32_t t = primitive_indices[pi];
                for (int v = 0; v < 3; v++) {
                    uint32_t idx = indices[t * 3 + v];
                    for (int a = 0; a < 3; a++) {
                        lo[a] = std::min(lo[a], points[idx * 3 + a]);
                        hi[a] = std::max(hi[a], points[idx * 3 + a]);
                    }
                }
            }
            lowers[ni].x = lo[0]; lowers[ni].y = lo[1]; lowers[ni].z = lo[2];
            uppers[ni].x = hi[0]; uppers[ni].y = hi[1]; uppers[ni].z = hi[2];
        } else {
            // Inner node: recursively refit children first (post-order traversal)
            uint32_t left  = lower_packed & 0x3FFFFFFFu;
            uint32_t right = uppers[ni].packed;
            refit_recursive(left);
            refit_recursive(right);
            lowers[ni].x = std::min(lowers[left].x, lowers[right].x);
            lowers[ni].y = std::min(lowers[left].y, lowers[right].y);
            lowers[ni].z = std::min(lowers[left].z, lowers[right].z);
            uppers[ni].x = std::max(uppers[left].x, uppers[right].x);
            uppers[ni].y = std::max(uppers[left].y, uppers[right].y);
            uppers[ni].z = std::max(uppers[left].z, uppers[right].z);
        }
    }

    void refit() {
        if (!lowers.empty()) {
            refit_recursive(root);
        }
    }


    // BVH-traversal closest-point query — mirrors the GLSL shader exactly
    // (same pruning, same packed node interpretation). Used to validate the
    // traversal on CPU against query_closest brute force.
    bool query_closest_bvh(const float* p, float max_dist, uint32_t& out_face,
                           float* out_point) const {
        float min_dist = max_dist;
        bool found = false;
        uint32_t best = 0;
        uint32_t stack[32];
        uint32_t sp = 0;
        stack[sp++] = root;
        const float eps = 1e-6f; // NOT the shader's avg_edge*1e-3; test pure traversal
        while (sp > 0) {
            uint32_t ni = stack[--sp];
            float nds = dist_sq_aabb(p, lowers[ni], uppers[ni]);
            if (nds > (min_dist + eps) * (min_dist + eps)) continue;
            uint32_t lower_packed = lowers[ni].packed;
            uint32_t left_i = lower_packed & 0x3FFFFFFFu;
            uint32_t is_leaf = (lower_packed >> 30u) & 0x3u;
            uint32_t right_i = uppers[ni].packed;
            if (is_leaf != 0u) {
                for (uint32_t pc = left_i; pc < right_i; pc++) {
                    uint32_t t = primitive_indices[pc];
                    const float* a = &points[indices[t*3+0]*3];
                    const float* b = &points[indices[t*3+1]*3];
                    const float* c = &points[indices[t*3+2]*3];
                    float cp[3];
                    closest_point_on_triangle(p, a, b, c, cp);
                    float dx = cp[0]-p[0], dy = cp[1]-p[1], dz = cp[2]-p[2];
                    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (dist < min_dist + eps) {
                        if (dist < min_dist) { min_dist = dist; }
                        best = t; found = true;
                        out_point[0]=cp[0]; out_point[1]=cp[1]; out_point[2]=cp[2];
                    }
                }
            } else {
                float lds = dist_sq_aabb(p, lowers[left_i], uppers[left_i]);
                float rds = dist_sq_aabb(p, lowers[right_i], uppers[right_i]);
                float thresh = (min_dist + eps) * (min_dist + eps);
                if (lds < rds) {
                    if (rds < thresh) stack[sp++] = right_i;
                    if (lds < thresh) stack[sp++] = left_i;
                } else {
                    if (lds < thresh) stack[sp++] = left_i;
                    if (rds < thresh) stack[sp++] = right_i;
                }
            }
        }
        out_face = best;
        return found;
    }

    static float dist_sq_aabb(const float* p, const PackedNodeHalf& lo,
                              const PackedNodeHalf& hi) {
        float dx = std::max(std::max(lo.x - p[0], 0.0f), p[0] - hi.x);
        float dy = std::max(std::max(lo.y - p[1], 0.0f), p[1] - hi.y);
        float dz = std::max(std::max(lo.z - p[2], 0.0f), p[2] - hi.z);
        return dx*dx + dy*dy + dz*dz;
    }

    // ---- CPU closest-point query (reference implementation, used for the
    // shader self-test AND as the brute-force comparison target) ----
    static void closest_point_on_triangle(const float* p, const float* a,
                                          const float* b, const float* c, float* out) {
        float ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
        float ac[3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
        float ap[3] = {p[0]-a[0], p[1]-a[1], p[2]-a[2]};
        float d1 = ab[0]*ap[0]+ab[1]*ap[1]+ab[2]*ap[2];
        float d2 = ac[0]*ap[0]+ac[1]*ap[1]+ac[2]*ap[2];
        float bp[3] = {p[0]-b[0], p[1]-b[1], p[2]-b[2]};
        float d3 = ab[0]*bp[0]+ab[1]*bp[1]+ab[2]*bp[2];
        float d4 = ac[0]*bp[0]+ac[1]*bp[1]+ac[2]*bp[2];
        float cp[3] = {p[0]-c[0], p[1]-c[1], p[2]-c[2]};
        float d5 = ab[0]*cp[0]+ab[1]*cp[1]+ab[2]*cp[2];
        float d6 = ac[0]*cp[0]+ac[1]*cp[1]+ac[2]*cp[2];
        float va = d3*d6 - d5*d4;
        float vb = d5*d2 - d1*d6;
        float vc = d1*d4 - d3*d2;
        float denom = va+vb+vc;
        if (denom > 1e-12f && va >= 0 && vb >= 0 && vc >= 0) {
            float inv = 1.0f/denom;
            out[0] = a[0] + (ab[0]*vb + ac[0]*vc)*inv;
            out[1] = a[1] + (ab[1]*vb + ac[1]*vc)*inv;
            out[2] = a[2] + (ab[2]*vb + ac[2]*vc)*inv;
            return;
        }
        // Exact triangle region tests for closest point computation:
        if (d3 >= 0 && d4 <= d3) { out[0]=b[0]; out[1]=b[1]; out[2]=b[2]; return; }
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
            float t = d1/(d1-d3);
            out[0]=a[0]+ab[0]*t; out[1]=a[1]+ab[1]*t; out[2]=a[2]+ab[2]*t; return;
        }
        if (d6 >= 0 && d5 <= d6) { out[0]=c[0]; out[1]=c[1]; out[2]=c[2]; return; }
        if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            float t = d2/(d2-d6);
            out[0]=a[0]+ac[0]*t; out[1]=a[1]+ac[1]*t; out[2]=a[2]+ac[2]*t; return;
        }
        if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
            float t = (d4-d3)/((d4-d3)+(d5-d6));
            float cb[3] = {c[0]-b[0], c[1]-b[1], c[2]-b[2]};
            out[0]=b[0]+cb[0]*t; out[1]=b[1]+cb[1]*t; out[2]=b[2]+cb[2]*t; return;
        }
        if (d5 <= 0 && d1 >= 0) {
            float t = d1/(d1-d5);
            out[0]=a[0]+ab[0]*t; out[1]=a[1]+ab[1]*t; out[2]=a[2]+ab[2]*t; return;
        }
        float t = d5/(d5-d6);
        float cb2[3] = {c[0]-b[0], c[1]-b[1], c[2]-b[2]};
        out[0]=b[0]+cb2[0]*t; out[1]=b[1]+cb2[1]*t; out[2]=b[2]+cb2[2]*t;
    }

    bool query_closest(const float* p, float max_dist, uint32_t& out_face,
                       float* out_point) const {
        float min_dist_sq = max_dist * max_dist;
        bool found = false;
        uint32_t best = 0;
        // brute-force over all triangles (CPU reference; the GLSL shader uses
        // the BVH — this validates RESULTS, not traversal)
        const uint32_t n_tris = uint32_t(indices.size() / 3);
        for (uint32_t t = 0; t < n_tris; t++) {
            const float* a = &points[indices[t*3+0]*3];
            const float* b = &points[indices[t*3+1]*3];
            const float* c = &points[indices[t*3+2]*3];
            float cp[3];
            closest_point_on_triangle(p, a, b, c, cp);
            float dx = cp[0]-p[0], dy = cp[1]-p[1], dz = cp[2]-p[2];
            float dsq = dx*dx + dy*dy + dz*dz;
            if (dsq < min_dist_sq) {
                min_dist_sq = dsq;
                found = true;
                best = t;
                out_point[0]=cp[0]; out_point[1]=cp[1]; out_point[2]=cp[2];
            }
        }
        out_face = best;
        return found;
    }
};

} // namespace vkx
