#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "scene.h"

namespace py = pybind11;

PYBIND11_MODULE(vkxpbd, m) {
    m.doc() = "Vulkan XPBD physics solver extension module";

    py::class_<vkx::SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("dt", &vkx::SolverConfig::dt)
        .def_readwrite("substeps", &vkx::SolverConfig::substeps)
        .def_readwrite("iterations", &vkx::SolverConfig::iterations)
        .def_readwrite("particle_v_max", &vkx::SolverConfig::particle_v_max)
        .def_readwrite("soft_contact_mu", &vkx::SolverConfig::soft_contact_mu)
        .def_readwrite("shape_material_mu", &vkx::SolverConfig::shape_material_mu)
        .def_readwrite("soft_contact_relaxation", &vkx::SolverConfig::soft_contact_relaxation)
        .def_readwrite("particle_adhesion", &vkx::SolverConfig::particle_adhesion)
        .def_readwrite("particle_mu", &vkx::SolverConfig::particle_mu)
        .def_readwrite("particle_cohesion", &vkx::SolverConfig::particle_cohesion)
        .def_readwrite("particle_max_radius", &vkx::SolverConfig::particle_max_radius)
        .def_readwrite("search_radius", &vkx::SolverConfig::search_radius)
        .def_readwrite("soft_contact_margin", &vkx::SolverConfig::soft_contact_margin)
        .def_readwrite("enable_restitution", &vkx::SolverConfig::enable_restitution)
        .def_readwrite("soft_contact_restitution", &vkx::SolverConfig::soft_contact_restitution)
        .def_readwrite("has_ground_plane", &vkx::SolverConfig::has_ground_plane)
        .def_readwrite("ground_plane_altitude", &vkx::SolverConfig::ground_plane_altitude)
        .def_readwrite("shader_dir", &vkx::SolverConfig::shader_dir)
        .def_readwrite("device_index", &vkx::SolverConfig::device_index)
        .def_readwrite("enable_validation", &vkx::SolverConfig::enable_validation)
        .def_readwrite("enable_diagnostics", &vkx::SolverConfig::enable_diagnostics)
        .def_readwrite("max_diag_pairs", &vkx::SolverConfig::max_diag_pairs)
        .def_property("gravity",
            [](const vkx::SolverConfig& c) {
                return std::vector<float>{c.gravity[0], c.gravity[1], c.gravity[2]};
            },
            [](vkx::SolverConfig& c, const std::vector<float>& g) {
                if (g.size() >= 3) {
                    c.gravity[0] = g[0]; c.gravity[1] = g[1]; c.gravity[2] = g[2];
                }
            });

    m.def("enumerate_devices", &vkx::Context::enumerate_devices, "Return list of available Vulkan device names");

    py::class_<vkx::XpbdSolver>(m, "XpbdSolver")
        .def(py::init<>())
        .def("init", &vkx::XpbdSolver::init,
             py::arg("config"), py::arg("num_particles"), py::arg("max_contacts") = 0)
        .def("add_mesh", [](vkx::XpbdSolver& self,
                            py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
                            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> indices,
                            float shape_margin) {
            py::buffer_info v_info = vertices.request();
            py::buffer_info i_info = indices.request();
            size_t num_verts = v_info.size / 3;
            size_t num_tris = i_info.size / 3;
            self.add_mesh(static_cast<const float*>(v_info.ptr), num_verts,
                          static_cast<const uint32_t*>(i_info.ptr), num_tris, shape_margin);
        }, py::arg("vertices"), py::arg("indices"), py::arg("shape_margin") = 0.0f)
        .def("finalize_meshes", &vkx::XpbdSolver::finalize_meshes)
        .def("set_particles", [](vkx::XpbdSolver& self,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> q,
                                 py::object qd_obj,
                                 py::object inv_m_obj,
                                 py::object radii_obj,
                                 py::object flags_obj) {
            py::buffer_info q_info = q.request();
            size_t n = q_info.size / 3;
            const float* q_ptr = static_cast<const float*>(q_info.ptr);

            const float* qd_ptr = nullptr;
            py::array_t<float, py::array::c_style | py::array::forcecast> qd_arr;
            if (!qd_obj.is_none()) {
                qd_arr = qd_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                qd_ptr = static_cast<const float*>(qd_arr.request().ptr);
            }

            const float* inv_m_ptr = nullptr;
            py::array_t<float, py::array::c_style | py::array::forcecast> inv_m_arr;
            if (!inv_m_obj.is_none()) {
                inv_m_arr = inv_m_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                inv_m_ptr = static_cast<const float*>(inv_m_arr.request().ptr);
            }

            const float* radii_ptr = nullptr;
            py::array_t<float, py::array::c_style | py::array::forcecast> radii_arr;
            if (!radii_obj.is_none()) {
                radii_arr = radii_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                radii_ptr = static_cast<const float*>(radii_arr.request().ptr);
            }

            const int32_t* flags_ptr = nullptr;
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> flags_arr;
            if (!flags_obj.is_none()) {
                flags_arr = flags_obj.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                flags_ptr = static_cast<const int32_t*>(flags_arr.request().ptr);
            }

            self.set_particles(q_ptr, qd_ptr, inv_m_ptr, radii_ptr, flags_ptr, n);
        }, py::arg("q"), py::arg("qd") = py::none(), py::arg("inv_mass") = py::none(),
           py::arg("radii") = py::none(), py::arg("flags") = py::none())
        .def("step", &vkx::XpbdSolver::step,
             py::arg("dt") = 1.0f / 60.0f, py::arg("substeps") = 1, py::arg("iterations") = 2)
        .def("get_positions", [](vkx::XpbdSolver& self) {
            size_t n = self.particle_count();
            py::array_t<float> result({n, (size_t)3});
            py::buffer_info buf = result.request();
            self.get_positions(static_cast<float*>(buf.ptr), n);
            return result;
        })
        .def("get_velocities", [](vkx::XpbdSolver& self) {
            size_t n = self.particle_count();
            py::array_t<float> result({n, (size_t)3});
            py::buffer_info buf = result.request();
            self.get_velocities(static_cast<float*>(buf.ptr), n);
            return result;
        })
        .def("collide_only", &vkx::XpbdSolver::collide_only)
        .def("update_mesh", [](vkx::XpbdSolver& self, size_t mesh_index,
                               py::array_t<float, py::array::c_style | py::array::forcecast> verts,
                               std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> velocities) {
            py::buffer_info vinfo = verts.request();
            size_t num_verts = vinfo.size / 3;
            const float* vel_ptr = nullptr;
            if (velocities.has_value() && velocities->size() > 0) {
                py::buffer_info vel_info = velocities->request();
                if (vel_info.size / 3 != num_verts) {
                    throw std::runtime_error("update_mesh: velocities size mismatch with vertices");
                }
                vel_ptr = static_cast<const float*>(vel_info.ptr);
            }
            self.update_mesh(mesh_index, static_cast<const float*>(vinfo.ptr), num_verts, vel_ptr);
        }, py::arg("mesh_index"), py::arg("verts"), py::arg("velocities") = py::none(),
           "Update vertex positions and optional velocities of an already-added mesh (for animated colliders). "
           "verts must be a flat float32 array of shape [n_verts*3] or [n_verts, 3]. "
           "Call this before step() each frame where the mesh has moved.")
        .def_property_readonly("particle_count", &vkx::XpbdSolver::particle_count)
        .def_property_readonly("mesh_count", &vkx::XpbdSolver::mesh_count)
        .def_property_readonly("last_contact_count", &vkx::XpbdSolver::last_contact_count)
        .def("get_contact_particles", [](vkx::XpbdSolver& self) {
            auto v = self.get_contact_particles();
            py::array_t<int32_t> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(int32_t));
            return result;
        })
        .def("get_contact_shapes", [](vkx::XpbdSolver& self) {
            auto v = self.get_contact_shapes();
            py::array_t<int32_t> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(int32_t));
            return result;
        })
        .def("get_contact_normals", [](vkx::XpbdSolver& self) {
            auto v = self.get_contact_normals();
            size_t n = v.size() / 3;
            py::array_t<float> result({n, (size_t)3});
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(float));
            return result;
        })
        .def("get_contact_body_positions", [](vkx::XpbdSolver& self) {
            auto v = self.get_contact_body_positions();
            size_t n = v.size() / 3;
            py::array_t<float> result({n, (size_t)3});
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(float));
            return result;
        })
        .def("get_contact_body_velocities", [](vkx::XpbdSolver& self) {
            auto v = self.get_contact_body_velocities();
            size_t n = v.size() / 3;
            py::array_t<float> result({n, (size_t)3});
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(float));
            return result;
        })
        .def("get_sorted_cells", [](vkx::XpbdSolver& self) {
            auto v = self.get_sorted_cells();
            py::array_t<uint32_t> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(uint32_t));
            return result;
        })
        .def("get_sorted_point_ids", [](vkx::XpbdSolver& self) {
            auto v = self.get_sorted_point_ids();
            py::array_t<uint32_t> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(uint32_t));
            return result;
        })
        .def("get_cell_starts", [](vkx::XpbdSolver& self) {
            auto v = self.get_cell_starts();
            py::array_t<int32_t> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(int32_t));
            return result;
        })
        .def("get_cell_ends", [](vkx::XpbdSolver& self) {
            auto v = self.get_cell_ends();
            py::array_t<int32_t> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(int32_t));
            return result;
        })
        .def("get_deltas", [](vkx::XpbdSolver& self) {
            auto v = self.get_deltas();
            size_t n = v.size() / 3;
            py::array_t<float> result({n, (size_t)3});
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(float));
            return result;
        })
        .def("get_diag_pair_count", &vkx::XpbdSolver::get_diag_pair_count)
        .def("get_diag_overflow_count", &vkx::XpbdSolver::get_diag_overflow_count)
        .def("reset_diag_buffers", &vkx::XpbdSolver::reset_diag_buffers)
        .def("get_diag_pairs", [](vkx::XpbdSolver& self) {
            auto records = self.get_diag_pairs();
            py::list out_list;
            for (const auto& r : records) {
                py::dict d;
                d["i"] = r.i;
                d["j"] = r.j;
                d["dist"] = r.dist;
                d["err"] = r.err;
                d["normal"] = py::make_tuple(r.normal_x, r.normal_y, r.normal_z);
                d["vrel"] = py::make_tuple(r.vrel_x, r.vrel_y, r.vrel_z);
                d["vn"] = r.vn;
                d["vt_len"] = r.vt_len;
                d["denom"] = r.denom;
                d["lambda_n"] = r.lambda_n;
                d["lambda_f"] = r.lambda_f;
                d["delta_i"] = py::make_tuple(r.delta_ix, r.delta_iy, r.delta_iz);
                d["iter_index"] = r.iter_index;
                out_list.append(d);
            }
            return out_list;
        })
        .def("get_p11_diag", [](vkx::XpbdSolver& self) {
            auto v = self.get_p11_diag();
            py::array_t<uint32_t> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(uint32_t));
            return result;
        })
        .def("get_p11_apply_diag", [](vkx::XpbdSolver& self) {
            auto v = self.get_p11_apply_diag();
            py::array_t<uint32_t> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(uint32_t));
            return result;
        })
        .def("get_shape_contact_diag", [](vkx::XpbdSolver& self) {
            auto v = self.get_shape_contact_diag();
            py::array_t<float> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(float));
            return result;
        })
        .def("get_mesh_candidate_diag", [](vkx::XpbdSolver& self) {
            auto v = self.get_mesh_candidate_diag();
            py::array_t<uint32_t> result(v.size());
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(uint32_t));
            return result;
        })
        .def("get_mesh_bvh_lowers", [](vkx::XpbdSolver& self, size_t mesh_index) {
            auto v = self.get_mesh_bvh_lowers(mesh_index);
            py::array_t<float> result({v.size() / 4, (size_t)4});
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(float));
            return result;
        })
        .def("get_mesh_bvh_uppers", [](vkx::XpbdSolver& self, size_t mesh_index) {
            auto v = self.get_mesh_bvh_uppers(mesh_index);
            py::array_t<float> result({v.size() / 4, (size_t)4});
            std::memcpy(result.request().ptr, v.data(), v.size() * sizeof(float));
            return result;
        });
}

