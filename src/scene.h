#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include "mesh_bvh.h"
#include "context.h"
#include "buffer.h"
#include "pipeline.h"
#include "gpu_sort.h"

namespace vkx {

struct SolverConfig {
    float dt = 1.0f / 60.0f;
    int substeps = 1;
    int iterations = 2;
    float particle_v_max = 1e5f;
    float soft_contact_mu = 0.5f;
    float shape_material_mu = 1.0f;
    float soft_contact_relaxation = 0.9f;
    float particle_adhesion = 0.0f;
    float particle_mu = 0.5f;
    float particle_cohesion = 0.0f;
    float particle_max_radius = 0.02f;
    float search_radius = 0.04f;
    float soft_contact_margin = 0.01f;
    bool enable_restitution = false;
    float soft_contact_restitution = 0.0f;
    bool has_ground_plane = true;
    float ground_plane_altitude = 0.0f;
    float gravity[3] = {0.0f, 0.0f, -9.81f};
    std::string shader_dir = "shaders";
    int device_index = -1;
    bool enable_validation = false;
    bool enable_diagnostics = false;
    uint32_t max_diag_pairs = 100000;
};

struct GPUPairRecord {
    uint32_t i;
    uint32_t j;
    float dist;
    float err;
    float normal_x;
    float normal_y;
    float normal_z;
    float vrel_x;
    float vrel_y;
    float vrel_z;
    float vn;
    float vt_len;
    float denom;
    float lambda_n;
    float lambda_f;
    float delta_ix;
    float delta_iy;
    float delta_iz;
    uint32_t iter_index;
    uint32_t pad0;
};

struct MeshData {
    std::vector<float> vertices; // x,y,z world coords
    std::vector<uint32_t> indices; // 3 per triangle
    float shape_margin = 0.0f;
};

class XpbdSolver {
public:
    XpbdSolver();
    ~XpbdSolver();

    void init(const SolverConfig& config, size_t num_particles, size_t max_contacts = 0);
    void add_mesh(const float* verts, size_t num_verts, const uint32_t* tris, size_t num_tris, float shape_margin = 0.0f);
    void finalize_meshes(); // builds and packs all BVHs

    void set_particles(const float* q, const float* qd, const float* inv_mass,
                       const float* radii, const int32_t* flags, size_t n);

    // Update vertex positions (and optional velocities) for mesh[mesh_index] for animated colliders.
    // verts: world-space flattened [n_verts * 3] float array (same count as add_mesh).
    // vels:  optional world-space flattened [n_verts * 3] float array.
    // Must be called BEFORE solver.step() for the frame where the new positions apply.
    void update_mesh(size_t mesh_index, const float* verts, size_t num_verts, const float* vels = nullptr);

    void step(float dt, int substeps, int iterations);
    void collide_only();

    void get_positions(float* out_q, size_t n);
    void get_velocities(float* out_qd, size_t n);

    size_t particle_count() const { return m_n_parts; }
    size_t mesh_count() const { return m_meshes.size(); }
    int last_contact_count() const { return m_contact_count; }

    std::vector<int32_t> get_contact_particles();
    std::vector<int32_t> get_contact_shapes();
    std::vector<float> get_contact_normals();
    std::vector<float> get_contact_body_positions();
    std::vector<float> get_contact_body_velocities();

    std::vector<float> get_mesh_bvh_lowers(size_t mesh_index) const;
    std::vector<float> get_mesh_bvh_uppers(size_t mesh_index) const;

    // GPU Grid & Particle-Particle solver diagnostics (read-only)
    std::vector<uint32_t> get_sorted_cells();
    std::vector<uint32_t> get_sorted_point_ids();
    std::vector<int32_t> get_cell_starts();
    std::vector<int32_t> get_cell_ends();
    std::vector<float> get_deltas();

    // Deep GPU Particle-Particle diagnostics
    uint32_t get_diag_pair_count();
    uint32_t get_diag_overflow_count();
    std::vector<GPUPairRecord> get_diag_pairs();
    void reset_diag_buffers();

    // Shape-contact diagnostic readback (32 uints)
    std::vector<uint32_t> get_p11_diag();
    std::vector<uint32_t> get_p11_apply_diag();
    std::vector<float> get_shape_contact_diag();
    std::vector<uint32_t> get_mesh_candidate_diag();

private:
    void init_pipelines();
    void record_dispatch(VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                         void* pc, size_t pc_size, uint32_t groups);

    // Context must be declared FIRST so it is initialized before resources,
    // and destructed LAST (after all Buffers, Pipelines, and Descriptors are destroyed).
    Context m_ctx;

    SolverConfig m_config;
    size_t m_n_parts = 0;
    size_t m_contact_max = 0;
    int m_contact_count = 0;

    std::vector<MeshData> m_meshes;
    std::vector<MeshBvh> m_bvhs;
    // Packed global arrays (mirrors what's uploaded to m_vtx / m_nl / m_nu).
    // Kept CPU-side so update_mesh() can patch a single mesh's slice & refit.
    std::vector<float>    m_all_verts;   // all vertex positions, packed (3 floats each)
    std::vector<float>    m_all_vels;    // all vertex velocities, packed (3 floats each)
    std::vector<size_t>   m_vert_offsets;// m_vert_offsets[c] = first vertex index of mesh c in m_all_verts
    std::vector<size_t>   m_node_offsets;// m_node_offsets[c] = first node index of mesh c in m_nl/m_nu

    // Buffers
    Buffer b_q, b_qd, b_f, b_invm, b_radius, b_flags, b_world, b_grav;
    Buffer b_shape_body, b_mu;
    Buffer b_body_q, b_body_qd, b_body_com, b_body_invI, b_body_invm, b_body_flags;
    Buffer b_body_delta;
    Buffer b_point_ids, b_cell_starts, b_cell_ends;
    Buffer b_cc, b_cpart, b_cshape, b_cbodypos, b_cbodyvel, b_cnormal;
    Buffer b_delta, b_q_out, b_qd_out, b_q_init, b_qd_init;
    Buffer m_vtx, m_idx, m_nl, m_nu, m_meta, m_prim, m_params, m_vvel;
    Buffer g_cells, g_ids;
    Buffer b_diag_header, b_diag_pairs;
    Buffer b_p11_diag;        // shape-contact diagnostic buffer
    Buffer b_p11_apply_diag;  // apply_particle_deltas diagnostic buffer
    Buffer b_mesh_diag;       // mesh candidate diagnostic buffer

    // Sorter
    GpuSort m_sorter;

    // Pipelines
    ComputePipeline sh_gen, sh_mesh;
    ComputePipeline sh_int, sh_con, sh_pcon, sh_app;
    ComputePipeline sh_gci, sh_goff;
    ComputePipeline sh_rest;

    // Descriptor pool & sets
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    std::unique_ptr<PoolGuard> m_pool_guard;
    VkDescriptorSet set_gen = VK_NULL_HANDLE;
    VkDescriptorSet set_mesh = VK_NULL_HANDLE;
    VkDescriptorSet set_int = VK_NULL_HANDLE;
    VkDescriptorSet set_con = VK_NULL_HANDLE;
    VkDescriptorSet set_pcon = VK_NULL_HANDLE;
    VkDescriptorSet set_app = VK_NULL_HANDLE;
    VkDescriptorSet set_gci = VK_NULL_HANDLE;
    VkDescriptorSet set_goff = VK_NULL_HANDLE;
    VkDescriptorSet set_rest = VK_NULL_HANDLE;

    bool m_initialized = false;
    bool m_meshes_finalized = false;
};

} // namespace vkx
