#include "scene.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <chrono>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace vkx {

static constexpr int GRID_DIM = 128;
static constexpr int NUM_CELLS = GRID_DIM * GRID_DIM * GRID_DIM;
static constexpr int GRID_ORIGIN = 64;

static auto make_bindings = [](uint32_t n) {
    std::vector<VkDescriptorSetLayoutBinding> v;
    for (uint32_t b = 0; b < n; b++)
        v.push_back({b, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    return v;
};

XpbdSolver::XpbdSolver() = default;
XpbdSolver::~XpbdSolver() {
    if (m_ctx.device) {
        vkDeviceWaitIdle(m_ctx.device);
        if (m_frame_fence) {
            vkDestroyFence(m_ctx.device, m_frame_fence, nullptr);
            m_frame_fence = VK_NULL_HANDLE;
        }
        if (m_profiling_enabled) {
            print_profiler_summary();
            if (m_query_pool) {
                vkDestroyQueryPool(m_ctx.device, m_query_pool, nullptr);
                m_query_pool = VK_NULL_HANDLE;
            }
        }
        m_substep_cmds.clear();
    }
}

void XpbdSolver::record_dispatch(VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                                 void* pc, size_t pc_size, uint32_t groups) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(m_ctx.cmd, &bi));
    vkCmdBindPipeline(m_ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(m_ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (pc_size) vkCmdPushConstants(m_ctx.cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pc_size, pc);
    vkCmdDispatch(m_ctx.cmd, groups, 1, 1);
    VK_CHECK(vkEndCommandBuffer(m_ctx.cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_ctx.cmd;
    VK_CHECK(vkQueueSubmit(m_ctx.compute_queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(m_ctx.compute_queue));
    vkResetCommandBuffer(m_ctx.cmd, 0);
}

void XpbdSolver::init(const SolverConfig& config, size_t num_particles, size_t max_contacts) {
    m_config = config;
    m_n_parts = num_particles;
    m_contact_max = max_contacts ? max_contacts : (num_particles * 2);

    m_ctx.init(m_config.enable_validation, m_config.device_index, m_config.compat_mode);

    // Create particle & state buffers
    b_q.create(&m_ctx, m_n_parts * 4 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_qd.create(&m_ctx, m_n_parts * 4 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_f.create(&m_ctx, m_n_parts * 4 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_invm.create(&m_ctx, m_n_parts * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_radius.create(&m_ctx, m_n_parts * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_flags.create(&m_ctx, m_n_parts * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_world.create(&m_ctx, m_n_parts * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_grav.create(&m_ctx, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    std::vector<float> f0(m_n_parts * 4, 0.0f);
    b_f.upload(f0.data(), f0.size() * 4);
    std::vector<int32_t> wo(m_n_parts, 0);
    b_world.upload(wo.data(), m_n_parts * 4);
    float grav_pad[4] = {m_config.gravity[0], m_config.gravity[1], m_config.gravity[2], 0.0f};
    b_grav.upload(grav_pad, 16);

    b_point_ids.create(&m_ctx, m_n_parts * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_cell_starts.create(&m_ctx, NUM_CELLS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_cell_ends.create(&m_ctx, NUM_CELLS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    b_cc.create(&m_ctx, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_cpart.create(&m_ctx, m_contact_max * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_cshape.create(&m_ctx, m_contact_max * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_cbodypos.create(&m_ctx, m_contact_max * 3 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_cbodyvel.create(&m_ctx, m_contact_max * 3 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_cnormal.create(&m_ctx, m_contact_max * 3 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    b_delta.create(&m_ctx, m_n_parts * 3 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_q_out.create(&m_ctx, m_n_parts * 4 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_qd_out.create(&m_ctx, m_n_parts * 4 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_q_init.create(&m_ctx, m_n_parts * 4 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_qd_init.create(&m_ctx, m_n_parts * 4 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    g_cells.create(&m_ctx, m_n_parts * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    g_ids.create(&m_ctx, m_n_parts * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Allocate diagnostic buffers: header [0]=pair_count, [1]=overflow_count
    b_diag_header.create(&m_ctx, 2 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::vector<uint32_t> diag_hdr_zero(2, 0);
    b_diag_header.upload(diag_hdr_zero.data(), 8);

    uint32_t max_pairs = m_config.max_diag_pairs ? m_config.max_diag_pairs : 100000;
    b_diag_pairs.create(&m_ctx, max_pairs * sizeof(GPUPairRecord), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Full shape-contact diagnostic buffer: only allocate full size if diagnostics enabled
    size_t shape_diag_floats = m_config.enable_diagnostics ? (4 * (m_contact_max > 0 ? m_contact_max : 100000) * 32) : 32;
    b_p11_diag.create(&m_ctx, shape_diag_floats * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    b_p11_diag.fill_zero(shape_diag_floats * sizeof(float));


    b_p11_apply_diag.create(&m_ctx, 32 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::vector<uint32_t> p11_zero(32, 0u);
    b_p11_apply_diag.upload(p11_zero.data(), 32 * 4);

    b_mesh_diag.create(&m_ctx, 32 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_mesh_diag.upload(p11_zero.data(), 32 * 4);



    m_pool = make_pool(&m_ctx, 64, 256);
    m_pool_guard = std::make_unique<PoolGuard>(&m_ctx, m_pool);

    m_sorter.create(&m_ctx, m_n_parts, m_pool, *m_pool_guard, m_config.shader_dir);
    m_sorter.bind_static(&m_ctx, g_cells.buffer, g_ids.buffer);

    const char* prof_env = std::getenv("VKXPBD_PROFILE");
    bool env_enabled = prof_env && (std::strcmp(prof_env, "1") == 0 || std::strcmp(prof_env, "true") == 0);
    if (m_config.enable_profile || env_enabled) {
        m_profiling_enabled = true;
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 64; // Max queries recorded per substep command buffer
        VK_CHECK(vkCreateQueryPool(m_ctx.device, &qpci, nullptr, &m_query_pool));
        printf("[Newton Vulkan] GPU Profiler ENABLED (timestampPeriod = %.4f ns)\n", m_ctx.props.limits.timestampPeriod);
    } else {
        m_profiling_enabled = false;
    }

    m_initialized = true;
}

void XpbdSolver::add_mesh(const float* verts, size_t num_verts, const uint32_t* tris, size_t num_tris,
                          float shape_margin, const float* shape_rot_xyzw, int watertight_override) {
    MeshData md;
    md.vertices.assign(verts, verts + num_verts * 3);
    md.indices.assign(tris, tris + num_tris * 3);
    md.shape_margin = shape_margin;
    if (shape_rot_xyzw) {
        md.shape_rot[0] = shape_rot_xyzw[0];
        md.shape_rot[1] = shape_rot_xyzw[1];
        md.shape_rot[2] = shape_rot_xyzw[2];
        md.shape_rot[3] = shape_rot_xyzw[3];
    } // else defaults to identity {0,0,0,1}

    if (watertight_override >= 0) {
        md.is_watertight = (watertight_override != 0);
    } else {
        // Auto-detect watertightness: exactly 2 triangles must share every geometric edge.
        // Matches Warp / Newton: newton.Mesh.is_watertight definition.
        if (num_tris == 0 || num_verts == 0) {
            md.is_watertight = false;
        } else {
            std::vector<std::pair<uint32_t, uint32_t>> edges;
            edges.reserve(num_tris * 3);
            for (size_t t = 0; t < num_tris; t++) {
                uint32_t i0 = tris[t * 3 + 0];
                uint32_t i1 = tris[t * 3 + 1];
                uint32_t i2 = tris[t * 3 + 2];
                edges.push_back({std::min(i0, i1), std::max(i0, i1)});
                edges.push_back({std::min(i1, i2), std::max(i1, i2)});
                edges.push_back({std::min(i2, i0), std::max(i2, i0)});
            }
            std::sort(edges.begin(), edges.end());
            bool closed_manifold = true;
            size_t i = 0;
            while (i < edges.size()) {
                size_t j = i + 1;
                while (j < edges.size() && edges[j] == edges[i]) j++;
                if ((j - i) != 2) {
                    closed_manifold = false;
                    break;
                }
                i = j;
            }
            md.is_watertight = closed_manifold;
        }
    }
    m_meshes.push_back(std::move(md));
    m_meshes_finalized = false;
}

void XpbdSolver::finalize_meshes() {
    if (!m_initialized) throw std::runtime_error("XpbdSolver::init must be called before finalize_meshes");
    size_t n_cubes = m_meshes.size();
    size_t n_shapes = 1 + n_cubes; // shape 0 = plane, 1..n = meshes
    size_t n_bodies = n_cubes;

    std::vector<int32_t> shape_body(n_shapes);
    std::vector<float> mat_mu(n_shapes, m_config.shape_material_mu);
    shape_body[0] = -1; // ground plane has body -1
    for (size_t c = 0; c < n_cubes; c++) shape_body[1 + c] = int32_t(c);

    b_shape_body.create(&m_ctx, n_shapes * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_shape_body.upload(shape_body.data(), n_shapes * 4);
    b_mu.create(&m_ctx, n_shapes * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_mu.upload(mat_mu.data(), n_shapes * 4);

    if (n_bodies > 0) {
        b_body_q.create(&m_ctx, n_bodies * 8 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_qd.create(&m_ctx, n_bodies * 8 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_com.create(&m_ctx, n_bodies * 3 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_invI.create(&m_ctx, n_bodies * 9 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_invm.create(&m_ctx, n_bodies * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_flags.create(&m_ctx, n_bodies * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        std::vector<float> bodyq(n_bodies * 8, 0.0f);
        for (size_t b = 0; b < n_bodies; b++) bodyq[b * 8 + 7] = 1.0f; // identity quaternion w=1
        b_body_q.upload(bodyq.data(), bodyq.size() * 4);

        std::vector<float> zeros(n_bodies * 9, 0.0f);
        b_body_qd.upload(zeros.data(), n_bodies * 8 * 4);
        b_body_com.upload(zeros.data(), n_bodies * 3 * 4);
        b_body_invI.upload(zeros.data(), n_bodies * 9 * 4);
        b_body_invm.upload(zeros.data(), n_bodies * 4);
        b_body_flags.upload(zeros.data(), n_bodies * 4);
        b_body_delta.create(&m_ctx, n_bodies * 8 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    } else {
        b_body_q.create(&m_ctx, 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_qd.create(&m_ctx, 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_com.create(&m_ctx, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_invI.create(&m_ctx, 36, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_invm.create(&m_ctx, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_flags.create(&m_ctx, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        b_body_delta.create(&m_ctx, 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    if (n_cubes > 0) {
        m_bvhs.resize(n_cubes);
        std::vector<float> all_verts;
        std::vector<uint32_t> all_tris;
        size_t total_nodes = 0, total_prims = 0;
        std::vector<float> cube_avg(n_cubes);

        // Track per-mesh offsets (vertex and node) for update_mesh()
        m_vert_offsets.resize(n_cubes);
        m_node_offsets.resize(n_cubes);

        for (size_t c = 0; c < n_cubes; c++) {
            m_vert_offsets[c] = all_verts.size() / 3; // vertex index start
            uint32_t v_offset = uint32_t(all_verts.size() / 3);
            all_verts.insert(all_verts.end(), m_meshes[c].vertices.begin(), m_meshes[c].vertices.end());
            std::vector<uint32_t> ctris = m_meshes[c].indices;
            for (size_t t = 0; t < ctris.size(); t++) ctris[t] += v_offset;
            uint32_t t_offset = uint32_t(all_tris.size() / 3);
            all_tris.insert(all_tris.end(), ctris.begin(), ctris.end());

            m_node_offsets[c] = total_nodes; // node index start
            m_bvhs[c].build(all_verts, ctris);
            total_nodes += m_bvhs[c].lowers.size();
            total_prims += m_bvhs[c].primitive_indices.size();

            double sum_len = 0.0;
            for (size_t t = 0; t < ctris.size() / 3; t++)
                for (int e = 0; e < 3; e++) {
                    uint32_t i0 = ctris[t*3+e], i1 = ctris[t*3+(e+1)%3];
                    float dx = all_verts[i1*3]-all_verts[i0*3], dy = all_verts[i1*3+1]-all_verts[i0*3+1], dz = all_verts[i1*3+2]-all_verts[i0*3+2];
                    sum_len += std::sqrt(dx*dx+dy*dy+dz*dz);
                }
            cube_avg[c] = float(sum_len / double(ctris.size()));
        }

        // Keep CPU-side copy for update_mesh()
        m_all_verts = all_verts;

        std::vector<float> nl(total_nodes * 4), nu(total_nodes * 4);
        std::vector<uint32_t> prim(total_prims), roots(n_cubes), meta(3 + n_cubes, 0);
        size_t node_base = 0, prim_base = 0;
        for (size_t c = 0; c < n_cubes; c++) {
            roots[c] = uint32_t(node_base + m_bvhs[c].root);
            for (size_t i = 0; i < m_bvhs[c].lowers.size(); i++) {
                size_t g = node_base + i;
                nl[g*4+0] = m_bvhs[c].lowers[i].x; nl[g*4+1] = m_bvhs[c].lowers[i].y; nl[g*4+2] = m_bvhs[c].lowers[i].z;
                nu[g*4+0] = m_bvhs[c].uppers[i].x; nu[g*4+1] = m_bvhs[c].uppers[i].y; nu[g*4+2] = m_bvhs[c].uppers[i].z;

                uint32_t l_packed = m_bvhs[c].lowers[i].packed;
                uint32_t u_packed = m_bvhs[c].uppers[i].packed;
                uint32_t is_leaf = (l_packed >> 30u) & 0x3u;

                if (is_leaf != 0u) {
                    uint32_t leaf_lo = (l_packed & 0x3FFFFFFFu) + uint32_t(prim_base);
                    uint32_t leaf_hi = (u_packed & 0x3FFFFFFFu) + uint32_t(prim_base);
                    uint32_t new_l = leaf_lo | (1u << 30);
                    uint32_t new_u = leaf_hi;
                    nl[g*4+3] = *reinterpret_cast<float*>(&new_l);
                    nu[g*4+3] = *reinterpret_cast<float*>(&new_u);
                } else {
                    uint32_t left_child = (l_packed & 0x3FFFFFFFu) + uint32_t(node_base);
                    uint32_t right_child = (u_packed & 0x3FFFFFFFu) + uint32_t(node_base);
                    nl[g*4+3] = *reinterpret_cast<float*>(&left_child);
                    nu[g*4+3] = *reinterpret_cast<float*>(&right_child);
                }
            }
            uint32_t t_offset = 0;
            for (size_t prev = 0; prev < c; prev++) t_offset += uint32_t(m_meshes[prev].indices.size() / 3);
            for (size_t i = 0; i < m_bvhs[c].primitive_indices.size(); i++)
                prim[prim_base + i] = m_bvhs[c].primitive_indices[i] + t_offset;
            prim_base += m_bvhs[c].primitive_indices.size();
            node_base += m_bvhs[c].lowers.size();
        }

        meta[0] = uint32_t(total_nodes);
        meta[1] = MeshBvh::LEAF_SIZE;
        meta[2] = *reinterpret_cast<uint32_t*>(&cube_avg[0]);
        for (size_t c = 0; c < n_cubes; c++) meta[3 + c] = roots[c];
        std::printf("[VKXPBD DIAG] finalize_meshes: n_cubes=%zu, total_nodes=%zu\n", n_cubes, total_nodes);
        for (size_t c = 0; c < n_cubes; c++) {
            std::printf("  Mesh %zu: root=%u, nodes=%zu, prims=%zu\n", c, roots[c], m_bvhs[c].lowers.size(), m_bvhs[c].primitive_indices.size());
        }

        m_nl.create(&m_ctx, nl.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_nl.upload(nl.data(), nl.size()*4);
        m_nu.create(&m_ctx, nu.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_nu.upload(nu.data(), nu.size()*4);
        m_meta.create(&m_ctx, meta.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_meta.upload(meta.data(), meta.size()*4);
        m_vtx.create(&m_ctx, all_verts.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_vtx.upload(all_verts.data(), all_verts.size()*4);
        m_all_vels.assign(all_verts.size(), 0.0f);
        m_vvel.create(&m_ctx, m_all_vels.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_vvel.upload(m_all_vels.data(), m_all_vels.size()*4);
        m_idx.create(&m_ctx, all_tris.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_idx.upload(all_tris.data(), all_tris.size()*4);
        m_prim.create(&m_ctx, prim.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_prim.upload(prim.data(), prim.size()*4);

        std::vector<uint32_t> mesh_prim_offsets(n_cubes), mesh_prim_counts(n_cubes);
        size_t p_acc = 0;
        for (size_t c = 0; c < n_cubes; c++) {
            mesh_prim_offsets[c] = uint32_t(p_acc);
            mesh_prim_counts[c]  = uint32_t(m_bvhs[c].primitive_indices.size());
            p_acc += m_bvhs[c].primitive_indices.size();
        }

        std::vector<uint32_t> mesh_params{uint32_t(m_n_parts), uint32_t(m_contact_max),
                                          *reinterpret_cast<uint32_t*>(&m_config.soft_contact_margin), uint32_t(n_cubes)};
        for (size_t c = 0; c < n_cubes; c++) {
            float sm = m_meshes[c].shape_margin;
            mesh_params.push_back(*reinterpret_cast<uint32_t*>(&sm));
        }
        for (size_t c = 0; c < n_cubes; c++) {
            uint32_t wt = m_meshes[c].is_watertight ? 1u : 0u;
            mesh_params.push_back(wt);
        }
        for (size_t c = 0; c < n_cubes; c++) {
            mesh_params.push_back(mesh_prim_offsets[c]);
        }
        for (size_t c = 0; c < n_cubes; c++) {
            mesh_params.push_back(mesh_prim_counts[c]);
        }
        m_params.create(&m_ctx, mesh_params.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_params.upload(mesh_params.data(), mesh_params.size()*4);

        // Shape rotation quaternions (xyzw, 4 floats per mesh) — used by generate_mesh_contacts
        // to replicate Warp's local-space normalize + world-transform for contact normals.
        std::vector<float> mesh_rots(n_cubes * 4);
        for (size_t c = 0; c < n_cubes; c++) {
            mesh_rots[c*4+0] = m_meshes[c].shape_rot[0];
            mesh_rots[c*4+1] = m_meshes[c].shape_rot[1];
            mesh_rots[c*4+2] = m_meshes[c].shape_rot[2];
            mesh_rots[c*4+3] = m_meshes[c].shape_rot[3];
        }
        m_mesh_rot.create(&m_ctx, mesh_rots.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_mesh_rot.upload(mesh_rots.data(), mesh_rots.size()*4);
    }

    init_pipelines();
    m_meshes_finalized = true;
}

void XpbdSolver::init_pipelines() {
    std::string prefix = m_config.shader_dir.empty() ? "" : (m_config.shader_dir + "/");
    sh_gen.create(&m_ctx, prefix + "generate_plane_contacts.spv", make_bindings(9), 16);
    set_gen = alloc_set(&m_ctx, m_pool, sh_gen.set_layout);
    m_pool_guard->sets.push_back(set_gen);
    write_set(&m_ctx, set_gen, 0, b_q.buffer, b_q.size);
    write_set(&m_ctx, set_gen, 1, b_radius.buffer, b_radius.size);
    write_set(&m_ctx, set_gen, 2, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_gen, 3, b_cc.buffer, b_cc.size);
    write_set(&m_ctx, set_gen, 4, b_cpart.buffer, b_cpart.size);
    write_set(&m_ctx, set_gen, 5, b_cshape.buffer, b_cshape.size);
    write_set(&m_ctx, set_gen, 6, b_cbodypos.buffer, b_cbodypos.size);
    write_set(&m_ctx, set_gen, 7, b_cbodyvel.buffer, b_cbodyvel.size);
    write_set(&m_ctx, set_gen, 8, b_cnormal.buffer, b_cnormal.size);

    if (!m_meshes.empty()) {
        sh_mesh.create(&m_ctx, prefix + "generate_mesh_contacts.spv", make_bindings(19), 0);
        set_mesh = alloc_set(&m_ctx, m_pool, sh_mesh.set_layout);
        m_pool_guard->sets.push_back(set_mesh);
        write_set(&m_ctx, set_mesh, 0, b_q.buffer, b_q.size);
        write_set(&m_ctx, set_mesh, 1, b_radius.buffer, b_radius.size);
        write_set(&m_ctx, set_mesh, 2, b_flags.buffer, b_flags.size);
        write_set(&m_ctx, set_mesh, 3, m_vtx.buffer, m_vtx.size);
        write_set(&m_ctx, set_mesh, 4, m_idx.buffer, m_idx.size);
        write_set(&m_ctx, set_mesh, 5, m_nl.buffer, m_nl.size);
        write_set(&m_ctx, set_mesh, 6, m_nu.buffer, m_nu.size);
        write_set(&m_ctx, set_mesh, 7, m_meta.buffer, m_meta.size);
        write_set(&m_ctx, set_mesh, 8, m_prim.buffer, m_prim.size);
        write_set(&m_ctx, set_mesh, 9, b_cc.buffer, b_cc.size);
        write_set(&m_ctx, set_mesh, 10, b_cpart.buffer, b_cpart.size);
        write_set(&m_ctx, set_mesh, 11, b_cshape.buffer, b_cshape.size);
        write_set(&m_ctx, set_mesh, 12, b_cbodypos.buffer, b_cbodypos.size);
        write_set(&m_ctx, set_mesh, 13, b_cbodyvel.buffer, b_cbodyvel.size);
        write_set(&m_ctx, set_mesh, 14, b_cnormal.buffer, b_cnormal.size);
        write_set(&m_ctx, set_mesh, 15, m_params.buffer, m_params.size);
        write_set(&m_ctx, set_mesh, 16, b_mesh_diag.buffer, b_mesh_diag.size);
        write_set(&m_ctx, set_mesh, 17, m_vvel.buffer, m_vvel.size);
        write_set(&m_ctx, set_mesh, 18, m_mesh_rot.buffer, m_mesh_rot.size); // shape rotation quats
    }

    sh_int.create(&m_ctx, prefix + "integrate_particles.spv", make_bindings(9), 16);
    sh_con.create(&m_ctx, prefix + "solve_particle_shape_contacts.spv", make_bindings(22), 32);
    sh_pcon.create(&m_ctx, prefix + "solve_particle_particle_contacts.spv", make_bindings(11), 48);
    sh_app.create(&m_ctx, prefix + "apply_particle_deltas.spv", make_bindings(7), 16);

    set_int = alloc_set(&m_ctx, m_pool, sh_int.set_layout);
    set_con = alloc_set(&m_ctx, m_pool, sh_con.set_layout);
    set_pcon = alloc_set(&m_ctx, m_pool, sh_pcon.set_layout);
    set_app = alloc_set(&m_ctx, m_pool, sh_app.set_layout);
    set_con_flip = alloc_set(&m_ctx, m_pool, sh_con.set_layout);
    set_pcon_flip = alloc_set(&m_ctx, m_pool, sh_pcon.set_layout);
    set_app_flip = alloc_set(&m_ctx, m_pool, sh_app.set_layout);
    m_pool_guard->sets.push_back(set_int);
    m_pool_guard->sets.push_back(set_con);
    m_pool_guard->sets.push_back(set_pcon);
    m_pool_guard->sets.push_back(set_app);
    m_pool_guard->sets.push_back(set_con_flip);
    m_pool_guard->sets.push_back(set_pcon_flip);
    m_pool_guard->sets.push_back(set_app_flip);

    write_set(&m_ctx, set_int, 0, b_q.buffer, b_q.size);
    write_set(&m_ctx, set_int, 1, b_qd.buffer, b_qd.size);
    write_set(&m_ctx, set_int, 2, b_f.buffer, b_f.size);
    write_set(&m_ctx, set_int, 3, b_invm.buffer, b_invm.size);
    write_set(&m_ctx, set_int, 4, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_int, 5, b_world.buffer, b_world.size);
    write_set(&m_ctx, set_int, 6, b_grav.buffer, b_grav.size);
    write_set(&m_ctx, set_int, 7, b_q_init.buffer, b_q_init.size);
    write_set(&m_ctx, set_int, 8, b_qd_init.buffer, b_qd_init.size);

    write_set(&m_ctx, set_con, 0, b_q.buffer, b_q.size);
    write_set(&m_ctx, set_con, 1, b_qd.buffer, b_qd.size);
    write_set(&m_ctx, set_con, 2, b_invm.buffer, b_invm.size);
    write_set(&m_ctx, set_con, 3, b_radius.buffer, b_radius.size);
    write_set(&m_ctx, set_con, 4, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_con, 5, b_body_q.buffer, b_body_q.size);
    write_set(&m_ctx, set_con, 6, b_body_qd.buffer, b_body_qd.size);
    write_set(&m_ctx, set_con, 7, b_body_com.buffer, b_body_com.size);
    write_set(&m_ctx, set_con, 8, b_body_invI.buffer, b_body_invI.size);
    write_set(&m_ctx, set_con, 9, b_body_invm.buffer, b_body_invm.size);
    write_set(&m_ctx, set_con, 10, b_body_flags.buffer, b_body_flags.size);
    write_set(&m_ctx, set_con, 11, b_shape_body.buffer, b_shape_body.size);
    write_set(&m_ctx, set_con, 12, b_mu.buffer, b_mu.size);
    write_set(&m_ctx, set_con, 13, b_cc.buffer, b_cc.size);
    write_set(&m_ctx, set_con, 14, b_cpart.buffer, b_cpart.size);
    write_set(&m_ctx, set_con, 15, b_cshape.buffer, b_cshape.size);
    write_set(&m_ctx, set_con, 16, b_cbodypos.buffer, b_cbodypos.size);
    write_set(&m_ctx, set_con, 17, b_cbodyvel.buffer, b_cbodyvel.size);
    write_set(&m_ctx, set_con, 18, b_cnormal.buffer, b_cnormal.size);
    write_set(&m_ctx, set_con, 19, b_delta.buffer, b_delta.size);
    write_set(&m_ctx, set_con, 20, b_body_delta.buffer, b_body_delta.size);
    write_set(&m_ctx, set_con, 21, b_p11_diag.buffer, b_p11_diag.size);

    // set_con_flip: reads from b_q_out and b_qd_out
    write_set(&m_ctx, set_con_flip, 0, b_q_out.buffer, b_q_out.size);
    write_set(&m_ctx, set_con_flip, 1, b_qd_out.buffer, b_qd_out.size);
    write_set(&m_ctx, set_con_flip, 2, b_invm.buffer, b_invm.size);
    write_set(&m_ctx, set_con_flip, 3, b_radius.buffer, b_radius.size);
    write_set(&m_ctx, set_con_flip, 4, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_con_flip, 5, b_body_q.buffer, b_body_q.size);
    write_set(&m_ctx, set_con_flip, 6, b_body_qd.buffer, b_body_qd.size);
    write_set(&m_ctx, set_con_flip, 7, b_body_com.buffer, b_body_com.size);
    write_set(&m_ctx, set_con_flip, 8, b_body_invI.buffer, b_body_invI.size);
    write_set(&m_ctx, set_con_flip, 9, b_body_invm.buffer, b_body_invm.size);
    write_set(&m_ctx, set_con_flip, 10, b_body_flags.buffer, b_body_flags.size);
    write_set(&m_ctx, set_con_flip, 11, b_shape_body.buffer, b_shape_body.size);
    write_set(&m_ctx, set_con_flip, 12, b_mu.buffer, b_mu.size);
    write_set(&m_ctx, set_con_flip, 13, b_cc.buffer, b_cc.size);
    write_set(&m_ctx, set_con_flip, 14, b_cpart.buffer, b_cpart.size);
    write_set(&m_ctx, set_con_flip, 15, b_cshape.buffer, b_cshape.size);
    write_set(&m_ctx, set_con_flip, 16, b_cbodypos.buffer, b_cbodypos.size);
    write_set(&m_ctx, set_con_flip, 17, b_cbodyvel.buffer, b_cbodyvel.size);
    write_set(&m_ctx, set_con_flip, 18, b_cnormal.buffer, b_cnormal.size);
    write_set(&m_ctx, set_con_flip, 19, b_delta.buffer, b_delta.size);
    write_set(&m_ctx, set_con_flip, 20, b_body_delta.buffer, b_body_delta.size);
    write_set(&m_ctx, set_con_flip, 21, b_p11_diag.buffer, b_p11_diag.size);

    write_set(&m_ctx, set_pcon, 0, b_point_ids.buffer, b_point_ids.size);
    write_set(&m_ctx, set_pcon, 1, b_cell_starts.buffer, b_cell_starts.size);
    write_set(&m_ctx, set_pcon, 2, b_cell_ends.buffer, b_cell_ends.size);
    write_set(&m_ctx, set_pcon, 3, b_q.buffer, b_q.size);
    write_set(&m_ctx, set_pcon, 4, b_qd.buffer, b_qd.size);
    write_set(&m_ctx, set_pcon, 5, b_invm.buffer, b_invm.size);
    write_set(&m_ctx, set_pcon, 6, b_radius.buffer, b_radius.size);
    write_set(&m_ctx, set_pcon, 7, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_pcon, 8, b_delta.buffer, b_delta.size);
    write_set(&m_ctx, set_pcon, 9, b_diag_header.buffer, b_diag_header.size);
    write_set(&m_ctx, set_pcon, 10, b_diag_pairs.buffer, b_diag_pairs.size);

    // set_pcon_flip: reads from b_q_out and b_qd_out
    write_set(&m_ctx, set_pcon_flip, 0, b_point_ids.buffer, b_point_ids.size);
    write_set(&m_ctx, set_pcon_flip, 1, b_cell_starts.buffer, b_cell_starts.size);
    write_set(&m_ctx, set_pcon_flip, 2, b_cell_ends.buffer, b_cell_ends.size);
    write_set(&m_ctx, set_pcon_flip, 3, b_q_out.buffer, b_q_out.size);
    write_set(&m_ctx, set_pcon_flip, 4, b_qd_out.buffer, b_qd_out.size);
    write_set(&m_ctx, set_pcon_flip, 5, b_invm.buffer, b_invm.size);
    write_set(&m_ctx, set_pcon_flip, 6, b_radius.buffer, b_radius.size);
    write_set(&m_ctx, set_pcon_flip, 7, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_pcon_flip, 8, b_delta.buffer, b_delta.size);
    write_set(&m_ctx, set_pcon_flip, 9, b_diag_header.buffer, b_diag_header.size);
    write_set(&m_ctx, set_pcon_flip, 10, b_diag_pairs.buffer, b_diag_pairs.size);

    // set_app (even iter): reads b_q, writes b_q_out / b_qd_out
    write_set(&m_ctx, set_app, 0, b_q_init.buffer, b_q_init.size);
    write_set(&m_ctx, set_app, 1, b_q.buffer, b_q.size);
    write_set(&m_ctx, set_app, 2, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_app, 3, b_delta.buffer, b_delta.size);
    write_set(&m_ctx, set_app, 4, b_q_out.buffer, b_q_out.size);
    write_set(&m_ctx, set_app, 5, b_qd_out.buffer, b_qd_out.size);
    write_set(&m_ctx, set_app, 6, b_p11_apply_diag.buffer, b_p11_apply_diag.size);

    // set_app_flip (odd iter): reads b_q_out, writes b_q / b_qd
    write_set(&m_ctx, set_app_flip, 0, b_q_init.buffer, b_q_init.size);
    write_set(&m_ctx, set_app_flip, 1, b_q_out.buffer, b_q_out.size);
    write_set(&m_ctx, set_app_flip, 2, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_app_flip, 3, b_delta.buffer, b_delta.size);
    write_set(&m_ctx, set_app_flip, 4, b_q.buffer, b_q.size);
    write_set(&m_ctx, set_app_flip, 5, b_qd.buffer, b_qd.size);
    write_set(&m_ctx, set_app_flip, 6, b_p11_apply_diag.buffer, b_p11_apply_diag.size);

    sh_gci.create(&m_ctx, prefix + "grid_cell_indices.spv", make_bindings(3), 16);
    sh_goff.create(&m_ctx, prefix + "grid_offsets.spv", make_bindings(3), 16);
    set_gci = alloc_set(&m_ctx, m_pool, sh_gci.set_layout);
    set_goff = alloc_set(&m_ctx, m_pool, sh_goff.set_layout);
    m_pool_guard->sets.push_back(set_gci);
    m_pool_guard->sets.push_back(set_goff);

    write_set(&m_ctx, set_gci, 0, b_q.buffer, b_q.size);
    write_set(&m_ctx, set_gci, 1, g_cells.buffer, g_cells.size);
    write_set(&m_ctx, set_gci, 2, g_ids.buffer, g_ids.size);

    write_set(&m_ctx, set_goff, 0, m_sorter.keys_a.buffer, m_sorter.keys_a.size);
    write_set(&m_ctx, set_goff, 1, b_cell_starts.buffer, b_cell_starts.size);
    write_set(&m_ctx, set_goff, 2, b_cell_ends.buffer, b_cell_ends.size);
    write_set(&m_ctx, set_pcon, 0, m_sorter.vals_a.buffer, m_sorter.vals_a.size);
    write_set(&m_ctx, set_pcon_flip, 0, m_sorter.vals_a.buffer, m_sorter.vals_a.size);

    sh_rest.create(&m_ctx, prefix + "apply_particle_shape_restitution.spv", make_bindings(18), 16);
    set_rest = alloc_set(&m_ctx, m_pool, sh_rest.set_layout);
    m_pool_guard->sets.push_back(set_rest);
    write_set(&m_ctx, set_rest, 0, b_qd.buffer, b_qd.size);
    write_set(&m_ctx, set_rest, 1, b_q_init.buffer, b_q_init.size);
    write_set(&m_ctx, set_rest, 2, b_qd_init.buffer, b_qd_init.size);
    write_set(&m_ctx, set_rest, 3, b_radius.buffer, b_radius.size);
    write_set(&m_ctx, set_rest, 4, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_rest, 5, b_body_q.buffer, b_body_q.size);
    write_set(&m_ctx, set_rest, 6, b_body_q.buffer, b_body_q.size); // prev = same for static/kinematic
    write_set(&m_ctx, set_rest, 7, b_body_qd.buffer, b_body_qd.size);
    write_set(&m_ctx, set_rest, 8, b_body_qd.buffer, b_body_qd.size);
    write_set(&m_ctx, set_rest, 9, b_body_com.buffer, b_body_com.size);
    write_set(&m_ctx, set_rest, 10, b_shape_body.buffer, b_shape_body.size);
    write_set(&m_ctx, set_rest, 11, b_cc.buffer, b_cc.size);
    write_set(&m_ctx, set_rest, 12, b_cpart.buffer, b_cpart.size);
    write_set(&m_ctx, set_rest, 13, b_cshape.buffer, b_cshape.size);
    write_set(&m_ctx, set_rest, 14, b_cbodypos.buffer, b_cbodypos.size);
    write_set(&m_ctx, set_rest, 15, b_cbodyvel.buffer, b_cbodyvel.size);
    write_set(&m_ctx, set_rest, 16, b_cnormal.buffer, b_cnormal.size);
    write_set(&m_ctx, set_rest, 17, b_qd.buffer, b_qd.size);
}

void XpbdSolver::set_particles(const float* q, const float* qd, const float* inv_mass,
                               const float* radii, const int32_t* flags, size_t n) {
    if (!m_initialized) throw std::runtime_error("XpbdSolver not initialized");
    if (n != m_n_parts) throw std::runtime_error("Particle count mismatch");

    std::vector<float> q4(n * 4), qd4(n * 4);
    for (size_t i = 0; i < n; i++) {
        q4[i * 4 + 0] = q[i * 3 + 0];
        q4[i * 4 + 1] = q[i * 3 + 1];
        q4[i * 4 + 2] = q[i * 3 + 2];
        q4[i * 4 + 3] = 0.0f;
        if (qd) {
            qd4[i * 4 + 0] = qd[i * 3 + 0];
            qd4[i * 4 + 1] = qd[i * 3 + 1];
            qd4[i * 4 + 2] = qd[i * 3 + 2];
            qd4[i * 4 + 3] = 0.0f;
        }
    }
    b_q.upload(q4.data(), q4.size() * 4);
    b_qd.upload(qd4.data(), qd4.size() * 4);
    if (inv_mass) b_invm.upload(inv_mass, n * 4);
    if (radii) b_radius.upload(radii, n * 4);
    if (flags) b_flags.upload(flags, n * 4);
}

void XpbdSolver::update_mesh(size_t mesh_index, const float* verts, size_t num_verts, const float* vels) {
    if (!m_meshes_finalized)
        throw std::runtime_error("update_mesh: call finalize_meshes() first");
    if (mesh_index >= m_meshes.size())
        throw std::runtime_error("update_mesh: mesh_index out of range");

    size_t expected = m_meshes[mesh_index].vertices.size() / 3;
    if (num_verts != expected)
        throw std::runtime_error("update_mesh: vertex count mismatch");

    // 1. Patch the CPU-side packed vertex array and velocities
    size_t v_start = m_vert_offsets[mesh_index]; // first vertex index in m_all_verts
    for (size_t i = 0; i < num_verts * 3; i++)
        m_all_verts[v_start * 3 + i] = verts[i];

    if (vels != nullptr) {
        for (size_t i = 0; i < num_verts * 3; i++)
            m_all_vels[v_start * 3 + i] = vels[i];
    } else {
        std::fill(m_all_vels.data() + v_start * 3, m_all_vels.data() + (v_start + num_verts) * 3, 0.0f);
    }

    // Also update m_meshes[mesh_index] so a re-finalize would be consistent
    std::copy(verts, verts + num_verts * 3, m_meshes[mesh_index].vertices.begin());

    // 2. Update the BVH's internal point array so refit() reads new positions
    //    m_bvhs[c].points stores the full all_verts slice seen at build() time.
    //    We need to patch only the vertices that belong to this mesh.
    //    m_bvhs[c] was built with all_verts that starts at v_start, so the
    //    relevant indices in m_bvhs[c].points start at v_start * 3 as well.
    for (size_t i = 0; i < num_verts * 3; i++)
        m_bvhs[mesh_index].points[v_start * 3 + i] = verts[i];

    // 3. Refit just this BVH (updates lowers/uppers xyz, keeps packed child links)
    m_bvhs[mesh_index].refit();

    // 4. Re-upload the vertex positions and velocities to GPU
    m_vtx.upload(m_all_verts.data(), m_all_verts.size() * 4);
    m_vvel.upload(m_all_vels.data(), m_all_vels.size() * 4);

    // 5. Rebuild the packed nl/nu arrays for the affected BVH's nodes and re-upload.
    //    We rebuild the whole nl/nu from all BVHs (cheap relative to simulation cost).
    size_t total_nodes = 0;
    for (auto& bvh : m_bvhs) total_nodes += bvh.lowers.size();

    std::vector<float> nl(total_nodes * 4), nu(total_nodes * 4);
    size_t node_base = 0;
    for (size_t c = 0; c < m_bvhs.size(); c++) {
        const auto& bvh = m_bvhs[c];
        size_t prim_base = 0;
        for (size_t prev = 0; prev < c; prev++) prim_base += m_bvhs[prev].primitive_indices.size();

        for (size_t i = 0; i < bvh.lowers.size(); i++) {
            size_t g = node_base + i;
            nl[g*4+0] = bvh.lowers[i].x; nl[g*4+1] = bvh.lowers[i].y; nl[g*4+2] = bvh.lowers[i].z;
            nu[g*4+0] = bvh.uppers[i].x; nu[g*4+1] = bvh.uppers[i].y; nu[g*4+2] = bvh.uppers[i].z;

            uint32_t l_packed = bvh.lowers[i].packed;
            uint32_t u_packed = bvh.uppers[i].packed;
            uint32_t is_leaf = (l_packed >> 30u) & 0x3u;

            if (is_leaf != 0u) {
                uint32_t leaf_lo = (l_packed & 0x3FFFFFFFu) + uint32_t(prim_base);
                uint32_t leaf_hi = (u_packed & 0x3FFFFFFFu) + uint32_t(prim_base);
                uint32_t new_l = leaf_lo | (1u << 30);
                uint32_t new_u = leaf_hi;
                nl[g*4+3] = *reinterpret_cast<float*>(&new_l);
                nu[g*4+3] = *reinterpret_cast<float*>(&new_u);
            } else {
                uint32_t left_child  = (l_packed & 0x3FFFFFFFu) + uint32_t(node_base);
                uint32_t right_child = (u_packed & 0x3FFFFFFFu) + uint32_t(node_base);
                nl[g*4+3] = *reinterpret_cast<float*>(&left_child);
                nu[g*4+3] = *reinterpret_cast<float*>(&right_child);
            }
        }
        node_base += bvh.lowers.size();
    }
    m_nl.upload(nl.data(), nl.size() * 4);
    m_nu.upload(nu.data(), nu.size() * 4);
}

void XpbdSolver::step(float dt, int substeps, int iterations) {
    if (!m_meshes_finalized) finalize_meshes();

    float sdt = dt / float(substeps);
    float cell_width_inv = 1.0f / m_config.search_radius;

    struct PCGen { float soft_margin; float altitude; uint32_t contact_max; float pad0; } pc_gen{
        m_config.soft_contact_margin, m_config.ground_plane_altitude, uint32_t(m_contact_max), 0.0f
    };
    struct PCMesh { uint32_t n, contact_max, margin_bits, pad; } pc_mesh{
        (uint32_t)m_n_parts, (uint32_t)m_contact_max,
        *reinterpret_cast<uint32_t*>(&m_config.soft_contact_margin), 0
    };
    struct PCInt { float dt, v_max, pad0, pad1; } pc_int{sdt, m_config.particle_v_max, 0, 0};
    struct PCCon {
        float mu, ka, cmax, dt, relax;
        int iter_index;
        int sub_index;
        int pad1;
    } pc_con{
        m_config.soft_contact_mu, m_config.particle_adhesion, (float)m_contact_max,
        sdt, m_config.soft_contact_relaxation, 0, 0, 0
    };
    struct PCPCon {
        float mu, coh, max_r, dt, relax, n, inv;
        uint32_t iter_index;
        uint32_t enable_diag;
        uint32_t max_pairs;
        uint32_t pad1, pad2;
    } pc_pcon{
        m_config.particle_mu, m_config.particle_cohesion, m_config.particle_max_radius,
        sdt, m_config.soft_contact_relaxation, (float)m_n_parts, cell_width_inv, 0,
        m_config.enable_diagnostics ? 1u : 0u,
        m_config.max_diag_pairs ? m_config.max_diag_pairs : 100000u,
        0, 0
    };
    struct PCApp {
        float dt, v_max;
        int iter_index;
        int sub_index;
    } pc_app{sdt, m_config.particle_v_max, 0, 0};
    struct PCGOff { int num_points; int pad0, pad1, pad2; } pc_goff{int(m_n_parts), 0, 0, 0};
    struct PCGci { float cell_width_inv; float pad0, pad1, pad2; } pc_gci{cell_width_inv, 0, 0, 0};
    struct PCRest { float particle_ka; float restitution; uint32_t contact_max; uint32_t pad0; } pc_rest{
        m_config.particle_adhesion, m_config.soft_contact_restitution, uint32_t(m_contact_max), 0
    };

    auto bar = [&]() {
        VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                          VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &b;
        vkCmdPipelineBarrier2(m_ctx.cmd, &dep);
    };

    auto cmd_dispatch = [&](VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                            void* pc, size_t pc_size, uint32_t groups) {
        vkCmdBindPipeline(m_ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(m_ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        if (pc_size) vkCmdPushConstants(m_ctx.cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pc_size, pc);
        vkCmdDispatch(m_ctx.cmd, groups, 1, 1);
        bar();
    };

    // Ensure sufficient command buffers allocated for all substeps
    if (m_substep_cmds.size() < (size_t)substeps) {
        if (!m_substep_cmds.empty()) {
            vkFreeCommandBuffers(m_ctx.device, m_ctx.command_pool, (uint32_t)m_substep_cmds.size(), m_substep_cmds.data());
            m_substep_cmds.clear();
        }
        m_substep_cmds.resize(substeps, VK_NULL_HANDLE);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = m_ctx.command_pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = (uint32_t)substeps;
        VK_CHECK(vkAllocateCommandBuffers(m_ctx.device, &cai, m_substep_cmds.data()));
    }

    if (!m_frame_fence) {
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(m_ctx.device, &fci, nullptr, &m_frame_fence));
    }

    // Ensure query pool capacity for all substeps in this frame (64 queries per substep)
    uint32_t required_queries = (uint32_t)substeps * 64;
    if (m_profiling_enabled && (!m_query_pool || m_query_pool_capacity < required_queries)) {
        if (m_query_pool) {
            vkDestroyQueryPool(m_ctx.device, m_query_pool, nullptr);
            m_query_pool = VK_NULL_HANDLE;
        }
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = required_queries;
        VK_CHECK(vkCreateQueryPool(m_ctx.device, &qpci, nullptr, &m_query_pool));
        m_query_pool_capacity = required_queries;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    auto step_wall_t0 = std::chrono::high_resolution_clock::now();

    for (int sub = 0; sub < substeps; sub++) {
        pc_con.sub_index = sub;
        pc_app.sub_index = sub;

        VkCommandBuffer cmd = m_substep_cmds[sub];
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));

        auto bar_cmd = [&](VkCommandBuffer c) {
            VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                              VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &b;
            vkCmdPipelineBarrier2(c, &dep);
        };

        auto cmd_dispatch_c = [&](VkCommandBuffer c, VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                                  void* pc, size_t pc_size, uint32_t groups) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
            if (pc_size) vkCmdPushConstants(c, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pc_size, pc);
            vkCmdDispatch(c, groups, 1, 1);
            bar_cmd(c);
        };

        // Every substep command buffer begins with a full pipeline barrier
        bar_cmd(cmd);

        uint32_t q_base = (uint32_t)sub * 64;
        auto ts_mark = [&](uint32_t offset) {
            if (m_profiling_enabled && m_query_pool) {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_query_pool, q_base + offset);
            }
        };

        if (sub == 0 && m_profiling_enabled && m_query_pool) {
            vkResetQueryPool(m_ctx.device, m_query_pool, 0, required_queries);
        }

        // 1. Collide phase: zero contact counter on GPU via vkCmdFillBuffer
        ts_mark(0);
        vkCmdFillBuffer(cmd, b_cc.buffer, 0, 4, 0);
        bar_cmd(cmd);

        if (m_config.has_ground_plane) {
            cmd_dispatch_c(cmd, sh_gen.pipeline, sh_gen.layout, set_gen, &pc_gen, sizeof(PCGen),
                           (uint32_t)((m_n_parts + 63) / 64));
        }
        if (!m_meshes.empty()) {
            cmd_dispatch_c(cmd, sh_mesh.pipeline, sh_mesh.layout, set_mesh, nullptr, 0,
                           (uint32_t)((m_n_parts + 63) / 64));
        }
        ts_mark(1); // 0..1: Collide phase (ground plane + mesh contact gen)

        // 2. Step phase (integrate + sort + grid offsets)
        // 2a. Initial state capture (fused into integrate)
        ts_mark(2);
        ts_mark(3); // 2..3: initial q/qd copy (now 0 ms)

        // 2b. Integrate (in-place x, v + writes q_init, qd_init)
        ts_mark(4);
        cmd_dispatch_c(cmd, sh_int.pipeline, sh_int.layout, set_int, &pc_int, sizeof(PCInt),
                       (uint32_t)((m_n_parts + 63) / 64));
        ts_mark(5); // 4..5: integrate

        // 2c. Integrate state update (fused into integrate)
        ts_mark(6);
        ts_mark(7); // 6..7: integrate copies (now 0 ms)

        // 2d. Grid cell indices
        ts_mark(8);
        cmd_dispatch_c(cmd, sh_gci.pipeline, sh_gci.layout, set_gci, &pc_gci, sizeof(PCGci),
                       (uint32_t)((m_n_parts + 63) / 64));
        ts_mark(9); // 8..9: grid cell indices

        // 2e. Radix sort (3 passes: histogram, scan, scatter per pass)
        uint32_t qidx = 10;
        GpuSort::PC sort_pc{uint32_t(m_n_parts), 0, m_sorter.groups, 0};
        for (uint32_t pass = 0; pass < 3; pass++) {
            sort_pc.shift = pass * 8;

            // Histogram
            ts_mark(qidx++);
            VkDescriptorSet hs = (pass == 0) ? m_sorter.hist_src : (pass == 1 ? m_sorter.hist_a : m_sorter.hist_b);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sorter.histogram.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sorter.histogram.layout, 0, 1, &hs, 0, nullptr);
            vkCmdPushConstants(cmd, m_sorter.histogram.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GpuSort::PC), &sort_pc);
            vkCmdDispatch(cmd, m_sorter.groups, 1, 1);
            bar_cmd(cmd);
            ts_mark(qidx++);

            // Scan: Three-kernel coalesced scan
            ts_mark(qidx++);
            m_sorter.record_scan_pass(&m_ctx, cmd, pass, uint32_t(m_n_parts));
            ts_mark(qidx++);

            // Scatter
            ts_mark(qidx++);
            VkDescriptorSet sc = (pass == 0) ? m_sorter.sc_src : (pass == 1 ? m_sorter.sc_a : m_sorter.sc_b);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sorter.scatter.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sorter.scatter.layout, 0, 1, &sc, 0, nullptr);
            vkCmdPushConstants(cmd, m_sorter.scatter.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GpuSort::PC), &sort_pc);
            vkCmdDispatch(cmd, m_sorter.groups, 1, 1);
            bar_cmd(cmd);
            ts_mark(qidx++);
        }
        

        // 2f. Cell fills
        ts_mark(28);
        vkCmdFillBuffer(cmd, b_cell_starts.buffer, 0, NUM_CELLS * 4, 0);
        bar_cmd(cmd);
        vkCmdFillBuffer(cmd, b_cell_ends.buffer, 0, NUM_CELLS * 4, 0);
        bar_cmd(cmd);
        ts_mark(29); // 28..29: cell fills

        // 2g. Grid offsets
        ts_mark(30);
        cmd_dispatch_c(cmd, sh_goff.pipeline, sh_goff.layout, set_goff, &pc_goff, sizeof(PCGOff),
                       (uint32_t)((m_n_parts + 63) / 64));
        ts_mark(31); // 30..31: grid offsets

        // 3. Iteration loop
        // 3. Iteration loop (ping-pong descriptor sets between b_q/b_qd and b_q_out/b_qd_out)
        qidx = 32;
        for (int iter = 0; iter < iterations; iter++) {
            bool flip = (iter % 2 != 0);
            VkDescriptorSet cur_con = flip ? set_con_flip : set_con;
            VkDescriptorSet cur_pcon = flip ? set_pcon_flip : set_pcon;
            VkDescriptorSet cur_app = flip ? set_app_flip : set_app;

            pc_con.iter_index = iter;
            pc_pcon.iter_index = uint32_t(iter);
            vkCmdFillBuffer(cmd, b_delta.buffer, 0, m_n_parts * 3 * 4, 0);
            bar_cmd(cmd);

            ts_mark(qidx);
            cmd_dispatch_c(cmd, sh_con.pipeline, sh_con.layout, cur_con, &pc_con, sizeof(PCCon),
                           (uint32_t)((m_contact_max + 63) / 64));
            ts_mark(qidx + 1);

            ts_mark(qidx + 2);
            cmd_dispatch_c(cmd, sh_pcon.pipeline, sh_pcon.layout, cur_pcon, &pc_pcon, sizeof(PCPCon),
                           (uint32_t)((m_n_parts + 63) / 64));
            ts_mark(qidx + 3);

            ts_mark(qidx + 4);
            // Pre-apply copies eliminated via descriptor ping-pong
            ts_mark(qidx + 5);

            pc_app.iter_index = iter;
            pc_app.sub_index = sub;
            ts_mark(qidx + 6);
            cmd_dispatch_c(cmd, sh_app.pipeline, sh_app.layout, cur_app, &pc_app, sizeof(PCApp),
                           (uint32_t)((m_n_parts + 63) / 64));
            ts_mark(qidx + 7);

            ts_mark(qidx + 8);
            // Post-apply copies eliminated via descriptor ping-pong
            ts_mark(qidx + 9);

            qidx += 10;
        }

        // If total iterations is odd, final result is in b_q_out/b_qd_out, so copy back to b_q/b_qd
        if (iterations % 2 != 0) {
            VkBufferCopy c1{0, 0, m_n_parts * 4 * 4};
            vkCmdCopyBuffer(cmd, b_q_out.buffer, b_q.buffer, 1, &c1);
            VkBufferCopy c2{0, 0, m_n_parts * 4 * 4};
            vkCmdCopyBuffer(cmd, b_qd_out.buffer, b_qd.buffer, 1, &c2);
            bar_cmd(cmd);
        }

        // 4. Restitution phase
        if (m_config.enable_restitution) {
            ts_mark(52);
            cmd_dispatch_c(cmd, sh_rest.pipeline, sh_rest.layout, set_rest, &pc_rest, sizeof(PCRest),
                           (uint32_t)((m_contact_max + 63) / 64));
            ts_mark(53);
            qidx = 54;
        }

        VK_CHECK(vkEndCommandBuffer(cmd));
    }

    // Submit substep command buffers in chunks to stay well under the Windows TDR limit
    int batch_size = m_config.substep_batch_size > 0 ? m_config.substep_batch_size : 15;
    for (int start = 0; start < substeps; start += batch_size) {
        int count = std::min(batch_size, substeps - start);
        VK_CHECK(vkResetFences(m_ctx.device, 1, &m_frame_fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = (uint32_t)count;
        si.pCommandBuffers = &m_substep_cmds[start];
        VK_CHECK(vkQueueSubmit(m_ctx.compute_queue, 1, &si, m_frame_fence));
        VK_CHECK(vkWaitForFences(m_ctx.device, 1, &m_frame_fence, VK_TRUE, UINT64_MAX));
    }

    // Read query pool results if profiling enabled (across all substeps)
    if (m_profiling_enabled && m_query_pool) {
        uint32_t queries_per_sub = m_config.enable_restitution ? 54 : 52;
        double period_ns = double(m_ctx.props.limits.timestampPeriod);
        std::vector<uint64_t> qres(queries_per_sub, 0);

        for (int sub = 0; sub < substeps; sub++) {
            uint32_t q_base = (uint32_t)sub * 64;
            VkResult res = vkGetQueryPoolResults(m_ctx.device, m_query_pool, q_base, queries_per_sub,
                                                 queries_per_sub * sizeof(uint64_t), qres.data(), sizeof(uint64_t),
                                                 VK_QUERY_RESULT_64_BIT);
            if (res == VK_SUCCESS) {
                auto get_ms = [&](uint32_t s, uint32_t e) -> double {
                    if (e < queries_per_sub && s < queries_per_sub && qres[e] >= qres[s] && qres[s] > 0) {
                        return double(qres[e] - qres[s]) * period_ns * 1e-6;
                    }
                    return 0.0;
                };

                m_prof_time_contact_gen += get_ms(0, 1);
                m_prof_time_init_copies += get_ms(2, 3);
                m_prof_time_integrate += get_ms(4, 5);
                m_prof_time_int_copies += get_ms(6, 7);
                m_prof_time_grid_indices += get_ms(8, 9);

                // Sort passes (3 passes: 0, 1, 2)
                for (uint32_t p = 0; p < 3; p++) {
                    uint32_t base = 10 + p * 6;
                    m_prof_time_sort_hist[p] += get_ms(base, base + 1);
                    m_prof_time_sort_scan[p] += get_ms(base + 2, base + 3);
                    m_prof_time_sort_scatter[p] += get_ms(base + 4, base + 5);
                }

                m_prof_time_cell_fills += get_ms(28, 29);
                m_prof_time_grid_offsets += get_ms(30, 31);

                for (int iter = 0; iter < iterations; iter++) {
                    uint32_t ibase = 32 + iter * 10;
                    m_prof_time_solve_shape += get_ms(ibase, ibase + 1);
                    m_prof_time_solve_pp += get_ms(ibase + 2, ibase + 3);
                    m_prof_time_iter_copies += get_ms(ibase + 4, ibase + 5);
                    m_prof_time_apply_deltas += get_ms(ibase + 6, ibase + 7);
                    m_prof_time_iter_copies += get_ms(ibase + 8, ibase + 9);
                }

                if (m_config.enable_restitution) {
                    m_prof_time_restitution += get_ms(52, 53);
                }
                m_prof_substep_count++;
            }
        }
    }

    auto step_wall_t1 = std::chrono::high_resolution_clock::now();
    if (m_profiling_enabled) {
        m_prof_wall_time_ms += std::chrono::duration<double, std::milli>(step_wall_t1 - step_wall_t0).count();
    }

    // Optional debug verification of sort ordering (zero-overhead if unset)
    static int check_sort_order_env = -1;
    if (check_sort_order_env == -1) {
        const char* e = std::getenv("VKXPBD_CHECK_SORT_ORDER");
        check_sort_order_env = (e && (std::strcmp(e, "1") == 0 || std::strcmp(e, "true") == 0)) ? 1 : 0;
    }
    if (check_sort_order_env == 1) {
        auto keys = get_sorted_cells();
        bool ok = true;
        for (size_t i = 1; i < keys.size(); i++) {
            if (keys[i] < keys[i - 1]) {
                fprintf(stderr, "[VKXPBD SORT CHECK ERROR] keys[%zu] = %u < keys[%zu] = %u!\n",
                        i, keys[i], i - 1, keys[i - 1]);
                ok = false;
                break;
            }
        }
        if (ok) {
            printf("[VKXPBD DEBUG] Sort check passed: %zu keys strictly non-decreasing.\n", keys.size());
        }
    }

    // Keep ONE contact download per frame for stats/reporting
    {
        auto cc = b_cc.download(4);
        m_contact_count = *reinterpret_cast<int32_t*>(cc.data());
        if (m_contact_count > (int)m_contact_max) m_contact_count = (int)m_contact_max;
    }
}

void XpbdSolver::step_solve_only(float dt, int substeps, int iterations) {
    if (!m_meshes_finalized) finalize_meshes();

    float sdt = dt / float(substeps);
    float cell_width_inv = 1.0f / m_config.search_radius;

    struct PCInt { float dt, v_max, pad0, pad1; } pc_int{sdt, m_config.particle_v_max, 0, 0};
    struct PCCon {
        float mu, ka, cmax, dt, relax;
        int iter_index;
        int sub_index;
        int pad1;
    } pc_con{
        m_config.soft_contact_mu, m_config.particle_adhesion, (float)m_contact_max,
        sdt, m_config.soft_contact_relaxation, 0, 0, 0
    };
    struct PCPCon {
        float mu, coh, max_r, dt, relax, n, inv;
        uint32_t iter_index;
        uint32_t enable_diag;
        uint32_t max_pairs;
        uint32_t pad1, pad2;
    } pc_pcon{
        m_config.particle_mu, m_config.particle_cohesion, m_config.particle_max_radius,
        sdt, m_config.soft_contact_relaxation, (float)m_n_parts, cell_width_inv, 0,
        m_config.enable_diagnostics ? 1u : 0u,
        m_config.max_diag_pairs ? m_config.max_diag_pairs : 100000u,
        0, 0
    };
    struct PCApp {
        float dt, v_max;
        int iter_index;
        int sub_index;
    } pc_app{sdt, m_config.particle_v_max, 0, 0};
    struct PCGOff { int num_points; int pad0, pad1, pad2; } pc_goff{int(m_n_parts), 0, 0, 0};
    struct PCGci { float cell_width_inv; float pad0, pad1, pad2; } pc_gci{cell_width_inv, 0, 0, 0};
    struct PCRest { float particle_ka; float restitution; uint32_t contact_max; uint32_t pad0; } pc_rest{
        m_config.particle_adhesion, m_config.soft_contact_restitution, uint32_t(m_contact_max), 0
    };

    auto bar = [&]() {
        VkMemoryBarrier2 b{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                          VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &b;
        vkCmdPipelineBarrier2(m_ctx.cmd, &dep);
    };

    auto cmd_dispatch = [&](VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                             const void* pc, size_t pc_size, uint32_t groups) {
        if (pc && pc_size > 0)
            vkCmdPushConstants(m_ctx.cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pc_size, pc);
        vkCmdBindPipeline(m_ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(m_ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(m_ctx.cmd, groups, 1, 1);
    };
    auto record_dispatch = cmd_dispatch;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    for (int sub = 0; sub < substeps; sub++) {
        pc_con.sub_index = sub;
        pc_app.sub_index = sub;

        VK_CHECK(vkBeginCommandBuffer(m_ctx.cmd, &bi));

        // Step phase: in-place integrate
        cmd_dispatch(sh_int.pipeline, sh_int.layout, set_int, &pc_int, sizeof(PCInt),
                     (uint32_t)((m_n_parts + 63) / 64));
        bar();
        cmd_dispatch(sh_gci.pipeline, sh_gci.layout, set_gci, &pc_gci, sizeof(PCGci),
                     (uint32_t)((m_n_parts + 63) / 64));
        vkCmdFillBuffer(m_ctx.cmd, m_sorter.global_hist.buffer, 0, m_sorter.global_hist.size, 0);
        bar();
        m_sorter.record(&m_ctx, m_ctx.cmd, g_cells.buffer, g_ids.buffer, (uint32_t)m_n_parts);
        vkCmdFillBuffer(m_ctx.cmd, b_cell_starts.buffer, 0, NUM_CELLS * 4, 0);
        bar();
        vkCmdFillBuffer(m_ctx.cmd, b_cell_ends.buffer, 0, NUM_CELLS * 4, 0);
        bar();
        cmd_dispatch(sh_goff.pipeline, sh_goff.layout, set_goff, &pc_goff, sizeof(PCGOff),
                     (uint32_t)((m_n_parts + 63) / 64));

        // Iteration loop (using pre-injected contacts, sized to contact_max with GPU early-out)
        for (int iter = 0; iter < iterations; iter++) {
            pc_con.iter_index = iter;
            pc_pcon.iter_index = uint32_t(iter);
            vkCmdFillBuffer(m_ctx.cmd, b_delta.buffer, 0, m_n_parts * 3 * 4, 0);
            bar();

            cmd_dispatch(sh_con.pipeline, sh_con.layout, set_con, &pc_con, sizeof(PCCon),
                         (uint32_t)((m_contact_max + 63) / 64));

            cmd_dispatch(sh_pcon.pipeline, sh_pcon.layout, set_pcon, &pc_pcon, sizeof(PCPCon),
                         (uint32_t)((m_n_parts + 63) / 64));
            {
                VkBufferCopy c1{0, 0, m_n_parts * 4 * 4};
                vkCmdCopyBuffer(m_ctx.cmd, b_q.buffer, b_q_out.buffer, 1, &c1);
                VkBufferCopy c2{0, 0, m_n_parts * 4 * 4};
                vkCmdCopyBuffer(m_ctx.cmd, b_qd.buffer, b_qd_out.buffer, 1, &c2);
                bar();
            }
            pc_app.iter_index = iter;
            pc_app.sub_index = sub;
            cmd_dispatch(sh_app.pipeline, sh_app.layout, set_app, &pc_app, sizeof(PCApp),
                         (uint32_t)((m_n_parts + 63) / 64));
            {
                VkBufferCopy c1{0, 0, m_n_parts * 4 * 4};
                vkCmdCopyBuffer(m_ctx.cmd, b_q_out.buffer, b_q.buffer, 1, &c1);
                VkBufferCopy c2{0, 0, m_n_parts * 4 * 4};
                vkCmdCopyBuffer(m_ctx.cmd, b_qd_out.buffer, b_qd.buffer, 1, &c2);
                bar();
            }
        }
        if (m_config.enable_restitution) {
            cmd_dispatch(sh_rest.pipeline, sh_rest.layout, set_rest, &pc_rest, sizeof(PCRest),
                         (uint32_t)((m_contact_max + 63) / 64));
        }
        VK_CHECK(vkEndCommandBuffer(m_ctx.cmd));
        {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &m_ctx.cmd;
            VK_CHECK(vkQueueSubmit(m_ctx.compute_queue, 1, &si, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(m_ctx.compute_queue));
        }
        vkResetCommandBuffer(m_ctx.cmd, 0);
    }
}

void XpbdSolver::collide_only() {
    b_cc.fill_zero(4);
    struct PCGen { float soft_margin; float altitude; uint32_t contact_max; float pad0; } pc_gen{
        m_config.soft_contact_margin, m_config.ground_plane_altitude, uint32_t(m_contact_max), 0.0f
    };
    if (m_config.has_ground_plane) {
        record_dispatch(sh_gen.pipeline, sh_gen.layout, set_gen, &pc_gen, sizeof(PCGen),
                        (uint32_t)((m_n_parts + 63) / 64));
    }
    if (!m_meshes.empty()) {
        record_dispatch(sh_mesh.pipeline, sh_mesh.layout, set_mesh, nullptr, 0,
                        (uint32_t)((m_n_parts + 63) / 64));
    }
    auto cc = b_cc.download(4);
    m_contact_count = *reinterpret_cast<int32_t*>(cc.data());
    if (m_contact_count > (int)m_contact_max) m_contact_count = (int)m_contact_max;
}

void XpbdSolver::get_positions(float* out_q, size_t n) {
    if (n > m_n_parts) n = m_n_parts;
    auto raw = b_q.download(n * 4 * 4);
    const float* s = reinterpret_cast<const float*>(raw.data());
    for (size_t i = 0; i < n; i++) {
        out_q[i * 3 + 0] = s[i * 4 + 0];
        out_q[i * 3 + 1] = s[i * 4 + 1];
        out_q[i * 3 + 2] = s[i * 4 + 2];
    }
}

void XpbdSolver::get_velocities(float* out_qd, size_t n) {
    if (n > m_n_parts) n = m_n_parts;
    auto raw = b_qd.download(n * 4 * 4);
    const float* s = reinterpret_cast<const float*>(raw.data());
    for (size_t i = 0; i < n; i++) {
        out_qd[i * 3 + 0] = s[i * 4 + 0];
        out_qd[i * 3 + 1] = s[i * 4 + 1];
        out_qd[i * 3 + 2] = s[i * 4 + 2];
    }
}

std::vector<int32_t> XpbdSolver::get_contact_particles() {
    if (m_contact_count <= 0) return {};
    auto raw = b_cpart.download(m_contact_count * 4);
    const int32_t* p = reinterpret_cast<const int32_t*>(raw.data());
    return std::vector<int32_t>(p, p + m_contact_count);
}

std::vector<int32_t> XpbdSolver::get_contact_shapes() {
    if (m_contact_count <= 0) return {};
    auto raw = b_cshape.download(m_contact_count * 4);
    const int32_t* p = reinterpret_cast<const int32_t*>(raw.data());
    return std::vector<int32_t>(p, p + m_contact_count);
}

std::vector<float> XpbdSolver::get_contact_normals() {
    if (m_contact_count <= 0) return {};
    auto raw = b_cnormal.download(m_contact_count * 3 * 4);
    const float* p = reinterpret_cast<const float*>(raw.data());
    return std::vector<float>(p, p + m_contact_count * 3);
}

std::vector<float> XpbdSolver::get_contact_body_positions() {
    if (m_contact_count <= 0) return {};
    auto raw = b_cbodypos.download(m_contact_count * 3 * 4);
    const float* p = reinterpret_cast<const float*>(raw.data());
    return std::vector<float>(p, p + m_contact_count * 3);
}

std::vector<float> XpbdSolver::get_contact_body_velocities() {
    if (m_contact_count <= 0) return {};
    auto raw = b_cbodyvel.download(m_contact_count * 3 * 4);
    const float* p = reinterpret_cast<const float*>(raw.data());
    return std::vector<float>(p, p + m_contact_count * 3);
}

void XpbdSolver::set_contact_data(const std::vector<int32_t>& particles,
                                   const std::vector<int32_t>& shapes,
                                   const std::vector<float>& body_positions,
                                   const std::vector<float>& normals,
                                   int contact_count) {
    m_contact_count = contact_count;
    if (contact_count <= 0) return;
    b_cpart.upload(reinterpret_cast<const uint8_t*>(particles.data()),   contact_count * 4);
    b_cshape.upload(reinterpret_cast<const uint8_t*>(shapes.data()),      contact_count * 4);
    b_cbodypos.upload(reinterpret_cast<const uint8_t*>(body_positions.data()), contact_count * 3 * 4);
    b_cnormal.upload(reinterpret_cast<const uint8_t*>(normals.data()),    contact_count * 3 * 4);
}

std::vector<uint32_t> XpbdSolver::get_sorted_cells() {
    if (m_n_parts == 0) return {};
    auto raw = m_sorter.keys_a.download(m_n_parts * 4);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(raw.data());
    return std::vector<uint32_t>(p, p + m_n_parts);
}

std::vector<uint32_t> XpbdSolver::get_sorted_point_ids() {
    if (m_n_parts == 0) return {};
    auto raw = m_sorter.vals_a.download(m_n_parts * 4);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(raw.data());
    return std::vector<uint32_t>(p, p + m_n_parts);
}

std::vector<int32_t> XpbdSolver::get_cell_starts() {
    auto raw = b_cell_starts.download(NUM_CELLS * 4);
    const int32_t* p = reinterpret_cast<const int32_t*>(raw.data());
    return std::vector<int32_t>(p, p + NUM_CELLS);
}

std::vector<int32_t> XpbdSolver::get_cell_ends() {
    auto raw = b_cell_ends.download(NUM_CELLS * 4);
    const int32_t* p = reinterpret_cast<const int32_t*>(raw.data());
    return std::vector<int32_t>(p, p + NUM_CELLS);
}

std::vector<float> XpbdSolver::get_deltas() {
    if (m_n_parts == 0) return {};
    auto raw = b_delta.download(m_n_parts * 3 * 4);
    const float* p = reinterpret_cast<const float*>(raw.data());
    return std::vector<float>(p, p + m_n_parts * 3);
}

uint32_t XpbdSolver::get_diag_pair_count() {
    if (!m_initialized || b_diag_header.size < 8) return 0;
    auto raw = b_diag_header.download(8);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(raw.data());
    return p[0];
}

uint32_t XpbdSolver::get_diag_overflow_count() {
    if (!m_initialized || b_diag_header.size < 8) return 0;
    auto raw = b_diag_header.download(8);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(raw.data());
    return p[1];
}

std::vector<GPUPairRecord> XpbdSolver::get_diag_pairs() {
    uint32_t count = get_diag_pair_count();
    uint32_t max_pairs = m_config.max_diag_pairs ? m_config.max_diag_pairs : 100000;
    uint32_t read_count = std::min(count, max_pairs);
    if (read_count == 0) return {};

    auto raw = b_diag_pairs.download(read_count * sizeof(GPUPairRecord));
    const GPUPairRecord* p = reinterpret_cast<const GPUPairRecord*>(raw.data());
    return std::vector<GPUPairRecord>(p, p + read_count);
}

void XpbdSolver::reset_diag_buffers() {
    if (m_initialized && b_diag_header.size >= 8) {
        std::vector<uint32_t> zeros(2, 0);
        b_diag_header.upload(zeros.data(), 8);
    }
    if (m_initialized && b_p11_diag.size > 0) {
        b_p11_diag.fill_zero(b_p11_diag.size);
    }
    if (m_initialized && b_mesh_diag.size > 0) {
        b_mesh_diag.fill_zero(b_mesh_diag.size);
    }
}

std::vector<uint32_t> XpbdSolver::get_p11_diag() {
    if (!m_initialized || b_p11_diag.size < 32 * 4) return {};
    auto raw = b_p11_diag.download(32 * 4);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(raw.data());
    return std::vector<uint32_t>(p, p + 32);
}

std::vector<float> XpbdSolver::get_shape_contact_diag() {
    if (!m_initialized || b_p11_diag.size == 0) return {};
    auto raw = b_p11_diag.download(b_p11_diag.size);
    const float* p = reinterpret_cast<const float*>(raw.data());
    return std::vector<float>(p, p + (b_p11_diag.size / sizeof(float)));
}

std::vector<uint32_t> XpbdSolver::get_p11_apply_diag() {
    if (!m_initialized || b_p11_apply_diag.size < 32 * 4) return {};
    auto raw = b_p11_apply_diag.download(32 * 4);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(raw.data());
    return std::vector<uint32_t>(p, p + 32);
}

std::vector<uint32_t> XpbdSolver::get_mesh_candidate_diag() {
    if (!m_initialized || b_mesh_diag.size < 32 * 4) return {};
    auto raw = b_mesh_diag.download(32 * 4);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(raw.data());
    return std::vector<uint32_t>(p, p + 32);
}

std::vector<float> XpbdSolver::get_mesh_bvh_lowers(size_t mesh_index) const {
    if (mesh_index >= m_bvhs.size()) return {};
    const auto& b = m_bvhs[mesh_index];
    std::vector<float> l(b.lowers.size() * 4);
    for (size_t i = 0; i < b.lowers.size(); i++) {
        l[i*4+0] = b.lowers[i].x; l[i*4+1] = b.lowers[i].y; l[i*4+2] = b.lowers[i].z;
        uint32_t lp = b.lowers[i].packed; l[i*4+3] = *reinterpret_cast<float*>(&lp);
    }
    return l;
}

std::vector<float> XpbdSolver::get_mesh_bvh_uppers(size_t mesh_index) const {
    if (mesh_index >= m_bvhs.size()) return {};
    const auto& b = m_bvhs[mesh_index];
    std::vector<float> u(b.uppers.size() * 4);
    for (size_t i = 0; i < b.uppers.size(); i++) {
        u[i*4+0] = b.uppers[i].x; u[i*4+1] = b.uppers[i].y; u[i*4+2] = b.uppers[i].z;
        uint32_t up = b.uppers[i].packed; u[i*4+3] = *reinterpret_cast<float*>(&up);
    }
    return u;
}

void XpbdSolver::print_profiler_summary() {
    if (!m_profiling_enabled || m_prof_substep_count == 0) return;

    double total_sort_ms = 0.0;
    for (int p = 0; p < 4; p++) {
        total_sort_ms += m_prof_time_sort_hist[p] + m_prof_time_sort_scan[p] + m_prof_time_sort_scatter[p];
    }
    double total_copies_ms = m_prof_time_init_copies + m_prof_time_int_copies + m_prof_time_iter_copies;

    double total_gpu_ms = m_prof_time_contact_gen +
                          m_prof_time_init_copies +
                          m_prof_time_integrate +
                          m_prof_time_int_copies +
                          m_prof_time_grid_indices +
                          total_sort_ms +
                          m_prof_time_cell_fills +
                          m_prof_time_grid_offsets +
                          m_prof_time_solve_shape +
                          m_prof_time_solve_pp +
                          m_prof_time_iter_copies +
                          m_prof_time_apply_deltas +
                          m_prof_time_restitution;

    double idle_gap_ms = m_prof_wall_time_ms > total_gpu_ms ? (m_prof_wall_time_ms - total_gpu_ms) : 0.0;
    double n_subs = double(m_prof_substep_count);

    printf("\n========================================================================================================\n");
    printf("                               VKXPBD GPU TIMESTAMP PROFILER SUMMARY                                    \n");
    printf("========================================================================================================\n");
    printf("Total Substeps Recorded: %llu\n", (unsigned long long)m_prof_substep_count);
    printf("Wall-Clock Step Time:    %10.2f ms (%8.4f ms/substep)\n", m_prof_wall_time_ms, m_prof_wall_time_ms / n_subs);
    printf("Summed GPU Active Time:  %10.2f ms (%8.4f ms/substep)\n", total_gpu_ms, total_gpu_ms / n_subs);
    printf("Host / Sync Idle Gap:    %10.2f ms (%8.4f ms/substep, %5.1f%% of wall)\n",
           idle_gap_ms, idle_gap_ms / n_subs, (idle_gap_ms / (m_prof_wall_time_ms > 0.0 ? m_prof_wall_time_ms : 1.0)) * 100.0);
    printf("--------------------------------------------------------------------------------------------------------\n");
    printf("%-35s | %10s | %14s | %10s\n", "Kernel / Stage Group", "Total (ms)", "Per-Substep (ms)", "% GPU Time");
    printf("--------------------------------------------------------------------------------------------------------\n");

    auto print_line = [&](const char* name, double ms) {
        double pct = (ms / (total_gpu_ms > 0.0 ? total_gpu_ms : 1.0)) * 100.0;
        printf("%-35s | %10.2f | %14.4f | %9.2f%%\n", name, ms, ms / n_subs, pct);
    };

    print_line("1. Contact Gen (plane + mesh)", m_prof_time_contact_gen);
    print_line("2a. Copies: Init (q, qd)", m_prof_time_init_copies);
    print_line("2b. Integrate (sh_int)", m_prof_time_integrate);
    print_line("2c. Copies: Integrate (q_out, qd_out)", m_prof_time_int_copies);
    print_line("2d. Grid Cell Indices (sh_gci)", m_prof_time_grid_indices);

    for (int p = 0; p < 3; p++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "2e. Sort Pass %d: Histogram", p);
        print_line(buf, m_prof_time_sort_hist[p]);
        snprintf(buf, sizeof(buf), "2e. Sort Pass %d: Scan (256 WGs)", p);
        print_line(buf, m_prof_time_sort_scan[p]);
        snprintf(buf, sizeof(buf), "2e. Sort Pass %d: Scatter", p);
        print_line(buf, m_prof_time_sort_scatter[p]);
    }

    print_line("2f. Cell Fills (starts + ends)", m_prof_time_cell_fills);
    print_line("2g. Grid Offsets (sh_goff)", m_prof_time_grid_offsets);
    print_line("3a. Solve Shape Contacts (sh_con)", m_prof_time_solve_shape);
    print_line("3b. Solve PP Contacts (sh_pcon)", m_prof_time_solve_pp);
    print_line("3c. Copies: Iterations (q, qd)", m_prof_time_iter_copies);
    print_line("3d. Apply Deltas (sh_app)", m_prof_time_apply_deltas);
    if (m_config.enable_restitution) {
        print_line("4.  Restitution (sh_rest)", m_prof_time_restitution);
    }

    printf("--------------------------------------------------------------------------------------------------------\n");
    printf("AGGREGATE ROLLUPS:\n");
    print_line(" -> Total Radix Sort (all passes)", total_sort_ms);
    print_line(" -> Total Buffer Copies (all groups)", total_copies_ms);
    print_line(" -> Total Solve Particle-Particle", m_prof_time_solve_pp);
    print_line(" -> Total Solve Shape Contacts", m_prof_time_solve_shape);
    printf("========================================================================================================\n\n");
    fflush(stdout);
}

} // namespace vkx

