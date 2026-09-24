#include "scene.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstring>

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

    m_ctx.init(m_config.enable_validation, m_config.device_index);

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

    // Full shape-contact diagnostic buffer: up to 4 iterations * m_contact_max * 32 floats
    size_t shape_diag_floats = 4 * (m_contact_max > 0 ? m_contact_max : 100000) * 32;
    b_p11_diag.create(&m_ctx, shape_diag_floats * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    b_p11_diag.fill_zero(shape_diag_floats * sizeof(float));


    b_p11_apply_diag.create(&m_ctx, 32 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::vector<uint32_t> p11_zero(32, 0u);
    b_p11_apply_diag.upload(p11_zero.data(), 32 * 4);

    b_mesh_diag.create(&m_ctx, 32 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    b_mesh_diag.upload(p11_zero.data(), 32 * 4);



    m_pool = make_pool(&m_ctx, 32, 128);
    m_pool_guard = std::make_unique<PoolGuard>(&m_ctx, m_pool);

    m_sorter.create(&m_ctx, m_n_parts, m_pool, *m_pool_guard, m_config.shader_dir);
    m_sorter.bind_static(&m_ctx, g_cells.buffer, g_ids.buffer);

    m_initialized = true;
}

void XpbdSolver::add_mesh(const float* verts, size_t num_verts, const uint32_t* tris, size_t num_tris, float shape_margin) {
    MeshData md;
    md.vertices.assign(verts, verts + num_verts * 3);
    md.indices.assign(tris, tris + num_tris * 3);
    md.shape_margin = shape_margin;
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

        std::vector<uint32_t> mesh_params{uint32_t(m_n_parts), uint32_t(m_contact_max),
                                          *reinterpret_cast<uint32_t*>(&m_config.soft_contact_margin), uint32_t(n_cubes)};
        for (size_t c = 0; c < n_cubes; c++) {
            float sm = m_meshes[c].shape_margin;
            mesh_params.push_back(*reinterpret_cast<uint32_t*>(&sm));
        }
        m_params.create(&m_ctx, mesh_params.size()*4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        m_params.upload(mesh_params.data(), mesh_params.size()*4);
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
        sh_mesh.create(&m_ctx, prefix + "generate_mesh_contacts.spv", make_bindings(18), 0);
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
    }

    sh_int.create(&m_ctx, prefix + "integrate_particles.spv", make_bindings(9), 16);
    sh_con.create(&m_ctx, prefix + "solve_particle_shape_contacts.spv", make_bindings(22), 32);
    sh_pcon.create(&m_ctx, prefix + "solve_particle_particle_contacts.spv", make_bindings(11), 48);
    sh_app.create(&m_ctx, prefix + "apply_particle_deltas.spv", make_bindings(7), 16);

    set_int = alloc_set(&m_ctx, m_pool, sh_int.set_layout);
    set_con = alloc_set(&m_ctx, m_pool, sh_con.set_layout);
    set_pcon = alloc_set(&m_ctx, m_pool, sh_pcon.set_layout);
    set_app = alloc_set(&m_ctx, m_pool, sh_app.set_layout);
    m_pool_guard->sets.push_back(set_int);
    m_pool_guard->sets.push_back(set_con);
    m_pool_guard->sets.push_back(set_pcon);
    m_pool_guard->sets.push_back(set_app);

    write_set(&m_ctx, set_int, 0, b_q.buffer, b_q.size);
    write_set(&m_ctx, set_int, 1, b_qd.buffer, b_qd.size);
    write_set(&m_ctx, set_int, 2, b_f.buffer, b_f.size);
    write_set(&m_ctx, set_int, 3, b_invm.buffer, b_invm.size);
    write_set(&m_ctx, set_int, 4, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_int, 5, b_world.buffer, b_world.size);
    write_set(&m_ctx, set_int, 6, b_grav.buffer, b_grav.size);
    write_set(&m_ctx, set_int, 7, b_q_out.buffer, b_q_out.size);
    write_set(&m_ctx, set_int, 8, b_qd_out.buffer, b_qd_out.size);

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

    write_set(&m_ctx, set_app, 0, b_q_init.buffer, b_q_init.size);
    write_set(&m_ctx, set_app, 1, b_q.buffer, b_q.size);
    write_set(&m_ctx, set_app, 2, b_flags.buffer, b_flags.size);
    write_set(&m_ctx, set_app, 3, b_delta.buffer, b_delta.size);
    write_set(&m_ctx, set_app, 4, b_q_out.buffer, b_q_out.size);
    write_set(&m_ctx, set_app, 5, b_qd_out.buffer, b_qd_out.size);
    write_set(&m_ctx, set_app, 6, b_p11_apply_diag.buffer, b_p11_apply_diag.size);

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

    for (int sub = 0; sub < substeps; sub++) {
        pc_con.sub_index = sub;
        // Collide phase: zero contact counter on host/GPU buffer before dispatches
        b_cc.fill_zero(4);
        if (m_config.has_ground_plane) {
            record_dispatch(sh_gen.pipeline, sh_gen.layout, set_gen, &pc_gen, sizeof(PCGen),
                            (uint32_t)((m_n_parts + 63) / 64));
        }
        if (!m_meshes.empty()) {
            record_dispatch(sh_mesh.pipeline, sh_mesh.layout, set_mesh, nullptr, 0,
                            (uint32_t)((m_n_parts + 63) / 64));
        }
        {
            auto cc = b_cc.download(4);
            m_contact_count = *reinterpret_cast<int32_t*>(cc.data());
            if (m_contact_count > (int)m_contact_max) m_contact_count = (int)m_contact_max;
        }

        // Step phase
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(m_ctx.cmd, &bi));

        {
            VkBufferCopy c{0, 0, m_n_parts * 4 * 4};
            vkCmdCopyBuffer(m_ctx.cmd, b_q.buffer, b_q_init.buffer, 1, &c);
            vkCmdCopyBuffer(m_ctx.cmd, b_qd.buffer, b_qd_init.buffer, 1, &c);
            bar();
        }

        cmd_dispatch(sh_int.pipeline, sh_int.layout, set_int, &pc_int, sizeof(PCInt),
                     (uint32_t)((m_n_parts + 63) / 64));
        {
            VkBufferCopy c1{0, 0, m_n_parts * 4 * 4};
            vkCmdCopyBuffer(m_ctx.cmd, b_q_out.buffer, b_q.buffer, 1, &c1);
            VkBufferCopy c2{0, 0, m_n_parts * 4 * 4};
            vkCmdCopyBuffer(m_ctx.cmd, b_qd_out.buffer, b_qd.buffer, 1, &c2);
            bar();
        }

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

        VK_CHECK(vkEndCommandBuffer(m_ctx.cmd));
        {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &m_ctx.cmd;
            VK_CHECK(vkQueueSubmit(m_ctx.compute_queue, 1, &si, VK_NULL_HANDLE));
            VK_CHECK(vkQueueWaitIdle(m_ctx.compute_queue));
        }
        vkResetCommandBuffer(m_ctx.cmd, 0);

        // Iteration loop
        VK_CHECK(vkBeginCommandBuffer(m_ctx.cmd, &bi));
        for (int iter = 0; iter < iterations; iter++) {
            pc_con.iter_index = iter;
            pc_pcon.iter_index = uint32_t(iter);
            vkCmdFillBuffer(m_ctx.cmd, b_delta.buffer, 0, m_n_parts * 3 * 4, 0);
            bar();
            if (m_contact_count > 0) {
                cmd_dispatch(sh_con.pipeline, sh_con.layout, set_con, &pc_con, sizeof(PCCon),
                             (uint32_t)((m_contact_count + 63) / 64));
            }
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
            }
        }

        // Restitution phase: applies particle-shape velocity restitution
        if (m_config.enable_restitution && m_contact_count > 0) {
            cmd_dispatch(sh_rest.pipeline, sh_rest.layout, set_rest, &pc_rest, sizeof(PCRest),
                         (uint32_t)((m_contact_count + 63) / 64));
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

} // namespace vkx

