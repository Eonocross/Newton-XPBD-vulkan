bl_info = {
    "name": "Newton Vulkan Direct Physics",
    "author": "Eonocross",
    "version": (1, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Newton",
    "description": "Bit-exact Vulkan compute backend for Newton XPBD physics simulation in Blender",
    "category": "Physics",
}

import bpy
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from bpy.app.handlers import persistent

def get_project_name():
    filepath = bpy.data.filepath
    if filepath:
        name = os.path.splitext(os.path.basename(filepath))[0]
        return name if name else "untitled"
    return "untitled"

def get_cache_root():
    proj_name = get_project_name()
    return os.path.abspath(bpy.path.abspath(f"//newton_vk_cache/{proj_name}"))

def get_cache_dir(subfolder):
    return os.path.join(get_cache_root(), subfolder)

# Ensure modules directory is in sys.path
addon_dir = os.path.dirname(os.path.abspath(__file__))
local_modules = os.path.join(addon_dir, "modules")
if os.path.exists(local_modules) and local_modules not in sys.path:
    sys.path.insert(0, local_modules)

try:
    import vkxpbd
    HAS_VKXPBD = True
except ImportError as e:
    HAS_VKXPBD = False
    print(f"[Newton Vulkan] Warning: vkxpbd module could not be imported: {e}")

@persistent
def load_npy_frame_vk(scene):
    if not hasattr(scene, "newton_vk_settings") or not scene.newton_vk_settings.disk_cache:
        return
    
    frame = scene.frame_current
    cache_dir = get_cache_dir("npy")
    npy_path = os.path.join(cache_dir, f"frame_{frame:04d}.npy")
    
    if os.path.exists(npy_path):
        obj = scene.objects.get("NewtonVulkanParticles")
        if obj and obj.type == 'MESH':
            try:
                data = np.load(npy_path)
                if len(obj.data.vertices) == data.shape[0]:
                    obj.data.vertices.foreach_set("co", data.ravel())
                    obj.data.update()
            except Exception as e:
                print(f"[Newton Vulkan] Error loading frame {frame}: {e}")

class NEWTONVK_PG_colliderItem(bpy.types.PropertyGroup):
    obj: bpy.props.PointerProperty(type=bpy.types.Object, name="", description="Mesh to use for collision")

class NEWTONVK_PG_settings(bpy.types.PropertyGroup):
    particle_count: bpy.props.IntProperty(name="Particles", default=30000, min=100)
    frames: bpy.props.IntProperty(name="Frames", default=100, min=1)
    grid_size: bpy.props.FloatVectorProperty(name="Grid Size", default=(2.0, 2.0, 2.0))
    grid_pos: bpy.props.FloatVectorProperty(name="Grid Pos", default=(0.0, 0.0, 3.0))
    disk_cache: bpy.props.BoolProperty(name="Use Disk Cache", description="Stream particles to lightweight .npy files", default=True)
    colliders: bpy.props.CollectionProperty(type=NEWTONVK_PG_colliderItem)
    
    # Particle Settings
    part_mass: bpy.props.FloatProperty(name="Mass", default=0.1, min=0.001)
    part_radius_std: bpy.props.FloatProperty(name="Radius Variance", default=0.0, min=0.0)
    part_jitter: bpy.props.FloatProperty(name="Jitter", default=0.01, min=0.0)
    part_vel: bpy.props.FloatVectorProperty(name="Initial Velocity", default=(0.0, 0.0, -1.0))
    part_vel_random: bpy.props.BoolProperty(name="Randomize Initial Velocity", description="Add deterministic uniform random perturbation to initial velocity", default=False)
    part_vel_seed: bpy.props.IntProperty(name="Velocity Seed", description="RNG seed for deterministic velocity generation", default=42)
    part_vel_random_range: bpy.props.FloatProperty(name="Velocity Random Range", description="Uniform range [-range, +range] added per XYZ component", default=1.0, min=0.0)
    
    # Collider Settings (ShapeConfig)
    col_mu: bpy.props.FloatProperty(name="Friction", default=1.0, min=0.0)
    col_restitution: bpy.props.FloatProperty(name="Bounciness", default=0.0, min=0.0, max=1.0)
    col_margin: bpy.props.FloatProperty(name="Collision Margin", default=0.0, min=0.0)
    col_mu_rolling: bpy.props.FloatProperty(name="Rolling Friction", default=0.0001, min=0.0)
    col_mu_torsional: bpy.props.FloatProperty(name="Torsional Friction", default=0.005, min=0.0)
    
    # Solver Settings (SolverXPBD)
    solver_iterations: bpy.props.IntProperty(name="Iterations", default=2, min=1)
    solver_rigid_contact_relax: bpy.props.FloatProperty(name="Contact Relaxation", default=0.8, min=0.0, max=1.0)
    solver_angular_damping: bpy.props.FloatProperty(name="Angular Damping", default=0.0, min=0.0)
    solver_enable_restitution: bpy.props.BoolProperty(name="Enable Bounciness", default=False)
    
    # Substep / Collider interpolation settings
    substeps: bpy.props.IntProperty(name="Substeps / Frame", description="Main physics substeps per rendered frame", default=10, min=1, max=100)
    collider_follows_main_substeps: bpy.props.BoolProperty(name="Collider Follows Main Substeps", description="If enabled, collider updates every main substep; if disabled, use custom collider substeps below", default=True)
    collider_substeps: bpy.props.IntProperty(name="Collider Substeps / Frame", description="Custom collider interpolation substeps when not following main substeps", default=1, min=1, max=100)

    # Ground plane settings
    has_ground_plane: bpy.props.BoolProperty(name="Enable Ground Plane", description="Infinite ground plane at Z=ground_plane_altitude", default=True)
    ground_plane_altitude: bpy.props.FloatProperty(name="Ground Altitude", description="Z coordinate of the infinite ground plane", default=0.0)

    # GPU Device Selection
    def get_vulkan_devices(self, context):
        items = [('-1', "Auto (Prefer Discrete NVIDIA)", "Automatically select discrete GPU")]
        if HAS_VKXPBD and hasattr(vkxpbd, "enumerate_devices"):
            try:
                devs = vkxpbd.enumerate_devices()
                for idx, name in enumerate(devs):
                    items.append((str(idx), f"[{idx}] {name}", name))
            except Exception:
                pass
        return items

    device_choice: bpy.props.EnumProperty(
        name="GPU Device",
        description="Select which Vulkan physical device to use for simulation",
        items=get_vulkan_devices
    )

    # Logging settings
    log_level: bpy.props.EnumProperty(
        name="Log Level",
        description="Verbosity level for console output and simulation logging",
        items=[
            ('MINIMAL', "Minimal", "Print elapsed time every 10 frames only. Validation & diagnostics disabled (fastest)"),
            ('STANDARD', "Standard", "Print concise per-frame stats and log to simulation.log"),
            ('VERBOSE', "Verbose / Max", "Full diagnostics, per-frame JSONL metrics, contact counts, and pair stats")
        ],
        default='MINIMAL'
    )

class NEWTONVK_PT_panel(bpy.types.Panel):
    bl_label = "Newton Vulkan Direct"
    bl_idname = "NEWTONVK_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Newton Vulkan'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        vk_props = scene.newton_vk_settings
        
        if not HAS_VKXPBD:
            layout.label(text="Error: vkxpbd module not found!", icon='ERROR')
            return

        box_dev = layout.box()
        box_dev.label(text="Hardware & Output:", icon='PREFERENCES')
        box_dev.prop(vk_props, "device_choice", text="GPU")
        box_dev.prop(vk_props, "disk_cache")
        box_dev.prop(vk_props, "log_level")
        
        box_grid = layout.box()
        box_grid.label(text="Grid Layout:")
        box_grid.prop(vk_props, "particle_count")
        box_grid.prop(vk_props, "frames")
        box_grid.prop(vk_props, "grid_size")
        box_grid.prop(vk_props, "grid_pos")
        
        box_part = layout.box()
        box_part.label(text="Particle Properties:")
        box_part.prop(vk_props, "part_mass")
        box_part.prop(vk_props, "part_radius_std")
        box_part.prop(vk_props, "part_jitter")
        box_part.prop(vk_props, "part_vel")
        box_part.prop(vk_props, "part_vel_random")
        if vk_props.part_vel_random:
            box_part.prop(vk_props, "part_vel_seed")
            box_part.prop(vk_props, "part_vel_random_range")
        
        box_col = layout.box()
        box_col.label(text="Collider Properties:")
        box_col.prop(vk_props, "col_mu")
        box_col.prop(vk_props, "col_restitution")
        box_col.prop(vk_props, "col_margin")
        box_col.prop(vk_props, "col_mu_rolling")
        box_col.prop(vk_props, "col_mu_torsional")
        
        box_col_list = box_col.box()
        box_col_list.label(text="Collider Objects (Empty = All):")
        for i, item in enumerate(vk_props.colliders):
            row = layout.row(align=True)
            row.prop(item, "obj", text="")
            op = row.operator("newton_vk.remove_collider", text="", icon='X')
            op.index = i
        box_col_list.operator("newton_vk.add_collider", text="Add Collider", icon='ADD')
        box_col_list.separator()
        box_col_list.operator("newton_vk.capture_deformation", icon='MOD_MESHDEFORM')
        
        box_solver = layout.box()
        box_solver.label(text="Solver Settings:")
        box_solver.prop(vk_props, "solver_iterations")
        box_solver.prop(vk_props, "solver_rigid_contact_relax")
        box_solver.prop(vk_props, "solver_angular_damping")
        box_solver.prop(vk_props, "solver_enable_restitution")
        box_solver.separator()
        box_solver.prop(vk_props, "has_ground_plane")
        if vk_props.has_ground_plane:
            box_solver.prop(vk_props, "ground_plane_altitude")
        
        box_sub = layout.box()
        box_sub.label(text="Substeps / Collider Interpolation:")
        box_sub.prop(vk_props, "substeps")
        box_sub.prop(vk_props, "collider_follows_main_substeps")
        if not vk_props.collider_follows_main_substeps:
            box_sub.prop(vk_props, "collider_substeps")
        
        layout.separator()
        layout.operator("newton_vk.simulate", icon='PLAY')

class NEWTONVK_OT_capture_deformation(bpy.types.Operator):
    bl_idname = "newton_vk.capture_deformation"
    bl_label = "Capture Deformation (Active)"
    bl_description = "Bake the deformed vertices of the active mesh to disk"
    
    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'
        
    def execute(self, context):
        obj = context.active_object
        scene = context.scene
        frames = scene.newton_vk_settings.frames
        
        cache_dir = get_cache_dir("deform")
        os.makedirs(cache_dir, exist_ok=True)
        
        saved_frame = scene.frame_current
        scene.frame_set(0)
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        num_verts = len(eval_mesh.vertices)
        
        data = np.zeros((frames, num_verts, 3), dtype=np.float32)
        
        for f in range(frames):
            scene.frame_set(f)
            depsgraph = context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            eval_mesh = eval_obj.to_mesh()
            
            if len(eval_mesh.vertices) != num_verts:
                self.report({'ERROR'}, f"Topology changed at frame {f}!")
                eval_obj.to_mesh_clear()
                scene.frame_set(saved_frame)
                return {'CANCELLED'}
                
            verts = np.empty(num_verts * 3, dtype=np.float32)
            eval_mesh.vertices.foreach_get("co", verts)
            verts = verts.reshape(-1, 3)
            
            mat = np.array(eval_obj.matrix_world, dtype=np.float32)
            transformed_verts = np.dot(verts, mat[:3, :3].T) + mat[:3, 3]
            data[f] = transformed_verts
            
            eval_obj.to_mesh_clear()
            
        scene.frame_set(saved_frame)
        
        out_path = os.path.join(cache_dir, f"{obj.name}.npy")
        np.save(out_path, data)
        self.report({'INFO'}, f"Saved deformation cache to {out_path}")
        return {'FINISHED'}

class NEWTONVK_OT_add_collider(bpy.types.Operator):
    bl_idname = "newton_vk.add_collider"
    bl_label = "Add Collider"
    
    def execute(self, context):
        context.scene.newton_vk_settings.colliders.add()
        return {'FINISHED'}

class NEWTONVK_OT_remove_collider(bpy.types.Operator):
    bl_idname = "newton_vk.remove_collider"
    bl_label = "Remove Collider"
    index: bpy.props.IntProperty()
    
    def execute(self, context):
        context.scene.newton_vk_settings.colliders.remove(self.index)
        return {'FINISHED'}

class NEWTONVK_OT_simulate(bpy.types.Operator):
    bl_idname = "newton_vk.simulate"
    bl_label = "Simulate & Cache (Vulkan)"
    bl_description = "Run Vulkan XPBD physics engine and generate cache"

    def execute(self, context):
        if not HAS_VKXPBD:
            self.report({'ERROR'}, "vkxpbd module is not available!")
            return {'CANCELLED'}
            
        scene = context.scene
        props = scene.newton_vk_settings
        
        # 1. Gather Collider Meshes
        depsgraph = context.evaluated_depsgraph_get()
        if len(props.colliders) > 0:
            objects_to_sync = [item.obj for item in props.colliders if item.obj is not None]
        else:
            objects_to_sync = scene.objects

        collected_meshes = []
        for obj in objects_to_sync:
            if obj.type == 'MESH' and not obj.name.startswith("cache_") and obj.name != "NewtonVulkanParticles" and obj.name != "NewtonParticles":
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.to_mesh()
                mesh.calc_loop_triangles()
                if len(mesh.vertices) == 0 or len(mesh.loop_triangles) == 0:
                    eval_obj.to_mesh_clear()
                    continue

                verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
                mesh.vertices.foreach_get("co", verts)
                verts = verts.reshape(-1, 3)

                indices = np.empty(len(mesh.loop_triangles) * 3, dtype=np.uint32)
                mesh.loop_triangles.foreach_get("vertices", indices)

                mat = np.array(obj.matrix_world, dtype=np.float32)
                R = mat[:3, :3]
                T = mat[:3, 3]
                world_verts = (np.dot(verts, R.T) + T).astype(np.float32)

                collected_meshes.append({
                    "name": obj.name,
                    "verts": world_verts.flatten(),
                    "indices": indices
                })
                eval_obj.to_mesh_clear()
                print(f"[Newton Vulkan] Synced collider mesh: {obj.name} ({len(world_verts)} verts, {len(indices)//3} tris)")

        # 2. Build Particle Grid
        vol = props.grid_size[0] * props.grid_size[1] * props.grid_size[2]
        spacing = (vol / props.particle_count) ** (1.0 / 3.0)
        dim_x = int(props.grid_size[0] / spacing)
        dim_y = int(props.grid_size[1] / spacing)
        dim_z = int(props.grid_size[2] / spacing)
        n_particles = dim_x * dim_y * dim_z

        print(f"[Newton Vulkan] Spawning particle grid: {dim_x}x{dim_y}x{dim_z} = {n_particles} particles")

        # Compute 3D lattice points
        px = np.arange(dim_x) * spacing
        py = np.arange(dim_y) * spacing
        pz = np.arange(dim_z) * spacing
        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T
        points += np.array(props.grid_pos)

        rng = np.random.default_rng(42)
        if props.part_jitter > 0.0:
            points += (rng.random(points.shape) - 0.5) * props.part_jitter

        radius_mean = spacing * 0.45
        radii = np.full(points.shape[0], fill_value=radius_mean)
        if props.part_radius_std > 0.0:
            radii += rng.standard_normal(radii.shape) * props.part_radius_std

        # Convert to float32 arrays for Vulkan GPU solver
        points = points.astype(np.float32)
        radii = radii.astype(np.float32)
        inv_mass = np.full(n_particles, fill_value=(1.0 / props.part_mass if props.part_mass > 0 else 0.0), dtype=np.float32)
        
        # Base velocity & deterministic randomization
        base_vel = np.array(props.part_vel, dtype=np.float32)
        if props.part_vel_random:
            rng_vel = np.random.default_rng(props.part_vel_seed)
            r_range = float(props.part_vel_random_range)
            perturbation = rng_vel.uniform(-r_range, r_range, size=(n_particles, 3)).astype(np.float32)
            velocities = base_vel.reshape(1, 3) + perturbation
        else:
            velocities = np.tile(base_vel.reshape(1, 3), (n_particles, 1)).astype(np.float32)

        v_speeds = np.linalg.norm(velocities, axis=1)
        print(f"[Newton Vulkan] Initial Velocity Stats:")
        print(f"  Randomized: {props.part_vel_random} (seed={props.part_vel_seed if props.part_vel_random else 'N/A'}, range={props.part_vel_random_range if props.part_vel_random else 'N/A'})")
        print(f"  Vx: min={np.min(velocities[:,0]):.4f}, max={np.max(velocities[:,0]):.4f}, mean={np.mean(velocities[:,0]):.4f}")
        print(f"  Vy: min={np.min(velocities[:,1]):.4f}, max={np.max(velocities[:,1]):.4f}, mean={np.mean(velocities[:,1]):.4f}")
        print(f"  Vz: min={np.min(velocities[:,2]):.4f}, max={np.max(velocities[:,2]):.4f}, mean={np.mean(velocities[:,2]):.4f}")
        print(f"  Speed: mean={np.mean(v_speeds):.4f} m/s, max={np.max(v_speeds):.4f} m/s")

        # ParticleFlags.ACTIVE (bit 0 = 1)
        flags = np.ones(n_particles, dtype=np.int32)

        # 3. Configure Vulkan Solver
        cfg = vkxpbd.SolverConfig()
        fps = 60.0
        dt = 1.0 / fps
        cfg.dt = dt
        cfg.substeps = props.substeps
        cfg.iterations = props.solver_iterations
        cfg.soft_contact_mu = 0.5
        cfg.shape_material_mu = props.col_mu
        cfg.soft_contact_relaxation = 0.9
        cfg.soft_contact_margin = 0.01 + props.col_margin
        cfg.enable_restitution = props.solver_enable_restitution
        cfg.soft_contact_restitution = props.col_restitution
        cfg.particle_max_radius = float(np.max(radii))
        cfg.search_radius = 2.0 * cfg.particle_max_radius
        cfg.has_ground_plane = props.has_ground_plane
        cfg.ground_plane_altitude = props.ground_plane_altitude

        log_lvl = getattr(props, "log_level", "MINIMAL")

        # Set shader directory path located in addons/modules/vkxpbd_shaders
        shader_dir = os.path.join(addon_dir, "modules", "vkxpbd_shaders")
        if not os.path.exists(shader_dir):
            shader_dir = os.path.join(addon_dir, "shaders")
        cfg.shader_dir = shader_dir
        # Device selection
        try:
            cfg.device_index = int(props.device_choice)
        except Exception:
            cfg.device_index = -1

        # Only enable validation layers and diagnostic readback buffers in VERBOSE mode
        cfg.enable_validation = (log_lvl == 'VERBOSE')
        cfg.enable_diagnostics = (log_lvl == 'VERBOSE')

        # Maximum potential contacts: at most 1 contact per shape per particle
        num_shapes = len(collected_meshes) + (1 if props.has_ground_plane else 0)
        num_shapes = max(num_shapes, 1)
        max_contacts = max(n_particles * num_shapes, 300000)
        solver = vkxpbd.XpbdSolver()
        solver.init(cfg, n_particles, max_contacts)

        for m in collected_meshes:
            solver.add_mesh(m["verts"], m["indices"])
        solver.finalize_meshes()

        solver.set_particles(
            points.flatten(),
            velocities.flatten(),
            inv_mass,
            radii,
            flags
        )

        # Load deform caches for animated colliders (matched by mesh name)
        deform_dir = get_cache_dir("deform")
        proj_name = get_project_name()
        warp_deform_dir = os.path.abspath(bpy.path.abspath(f"//newton_cache/{proj_name}/deform"))
        deform_caches = {}  # name -> {"data": ndarray[frames, nverts, 3], "mesh_idx": int}
        for mesh_idx, m in enumerate(collected_meshes):
            cache_path = os.path.join(deform_dir, f"{m['name']}.npy")
            if not os.path.exists(cache_path) and os.path.exists(os.path.join(warp_deform_dir, f"{m['name']}.npy")):
                cache_path = os.path.join(warp_deform_dir, f"{m['name']}.npy")
            if os.path.exists(cache_path):
                data = np.load(cache_path)
                expected_nverts = len(m["verts"]) // 3
                if data.ndim == 3 and data.shape[1] == expected_nverts:
                    deform_caches[m["name"]] = {"data": data, "mesh_idx": mesh_idx}
                    print(f"[Newton Vulkan] Loaded deform cache for '{m['name']}': {data.shape[0]} frames, {data.shape[1]} verts")
                else:
                    print(f"[Newton Vulkan] Skipping deform cache '{m['name']}': shape {data.shape} vs expected nverts={expected_nverts}")

        # 4. Simulate & Save Disk Cache
        cache_dir = get_cache_dir("npy")
        cache_root = get_cache_root()
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(cache_root, exist_ok=True)

        # Setup Loggers
        log_path = os.path.join(cache_root, "simulation.log")
        jsonl_path = os.path.join(cache_root, "simulation_metrics.jsonl")

        sim_init_meta = {
            "timestamp": datetime.now().isoformat(),
            "blender_version": list(bpy.app.version),
            "engine": "Newton Vulkan Direct XPBD",
            "particle_count": n_particles,
            "radius_stats": {"min": float(np.min(radii)), "max": float(np.max(radii)), "mean": float(np.mean(radii))},
            "mass_stats": {"mass": float(props.part_mass), "inv_mass": float(1.0 / props.part_mass if props.part_mass > 0 else 0.0)},
            "initial_velocity_stats": {
                "randomized": bool(props.part_vel_random),
                "seed": int(props.part_vel_seed) if props.part_vel_random else None,
                "range": float(props.part_vel_random_range) if props.part_vel_random else None,
                "base": list(props.part_vel),
                "mean_speed": float(np.mean(v_speeds)),
                "max_speed": float(np.max(v_speeds)),
            },
            "solver_config": {
                "dt": dt,
                "fps": fps,
                "substeps": props.substeps,
                "iterations": props.solver_iterations,
                "friction_mu": props.col_mu,
                "contact_margin": props.col_margin,
                "contact_relaxation": props.solver_rigid_contact_relax,
                "soft_contact_relaxation": 0.9,
                "soft_contact_margin": 0.01 + props.col_margin,
                "particle_soft_mu": 0.5,
                "shape_material_mu": props.col_mu,
                "search_radius": float(2.0 * np.max(radii)),
                "enable_restitution": props.solver_enable_restitution,
                "restitution": props.col_restitution if props.solver_enable_restitution else 0.0,
                "angular_damping": props.solver_angular_damping,
                "has_ground_plane": props.has_ground_plane,
                "ground_plane_altitude": props.ground_plane_altitude,
                "collider_count": len(collected_meshes),
            }
        }

        if log_lvl in ('STANDARD', 'VERBOSE'):
            with open(log_path, "w", encoding="utf-8") as f_log:
                f_log.write("="*80 + "\n")
                f_log.write("NEWTON VULKAN DIRECT XPBD SIMULATION LOG\n")
                f_log.write("="*80 + "\n")
                f_log.write(f"Timestamp:        {sim_init_meta['timestamp']}\n")
                f_log.write(f"Engine:           {sim_init_meta['engine']}\n")
                f_log.write(f"Particles:        {n_particles}\n")
                f_log.write(f"Initial Velocity: Base={list(props.part_vel)}, Random={props.part_vel_random} (seed={props.part_vel_seed}, range={props.part_vel_random_range})\n")
                f_log.write(f"                  Speed: mean={np.mean(v_speeds):.4f} m/s, max={np.max(v_speeds):.4f} m/s\n")
                f_log.write(f"Solver Config:    dt={dt:.4f}s, substeps={props.substeps}, iterations={props.solver_iterations}\n")
                f_log.write(f"Soft Contact:     particle_soft_mu=0.5, shape_mu={props.col_mu}, soft_relax=0.9, rigid_relax={props.solver_rigid_contact_relax}\n")
                f_log.write(f"                  soft_margin={0.01 + props.col_margin:.4f}, search_radius={2.0 * np.max(radii):.4f}\n")
                f_log.write(f"Restitution:      enabled={props.solver_enable_restitution}, e={props.col_restitution if props.solver_enable_restitution else 0.0}\n")
                f_log.write(f"Ground Plane:     enabled={props.has_ground_plane}, alt={props.ground_plane_altitude}\n")
                f_log.write("="*80 + "\n\n")
                f_log.write(f"{'Frame':<6} | {'Physics (ms)':<13} | {'Disk IO (ms)':<13} | {'Total (ms)':<11} | {'Contacts':<9} | {'Min Z (m)':<10}\n")
                f_log.write("-" * 80 + "\n")

        if log_lvl == 'VERBOSE':
            with open(jsonl_path, "w", encoding="utf-8") as f_j:
                f_j.write(json.dumps({"type": "init", "data": sim_init_meta}) + "\n")

        # Save initial frame 0 (state before any simulation steps)
        t_io0 = time.perf_counter()
        pos_init = solver.get_positions()
        np.save(os.path.join(cache_dir, "frame_0000.npy"), pos_init)
        init_io_ms = (time.perf_counter() - t_io0) * 1000.0

        if log_lvl == 'VERBOSE':
            frame_0_rec = {
                "type": "frame",
                "frame": 0,
                "elapsed_s": 0.0,
                "physics_ms": 0.0,
                "io_ms": round(init_io_ms, 2),
                "frame_ms": round(init_io_ms, 2),
                "contacts": 0,
                "min_z": round(float(np.min(pos_init[:, 2])), 4),
                "max_z": round(float(np.max(pos_init[:, 2])), 4)
            }
            with open(jsonl_path, "a", encoding="utf-8") as f_j:
                f_j.write(json.dumps(frame_0_rec) + "\n")

        if log_lvl in ('STANDARD', 'VERBOSE'):
            with open(log_path, "a", encoding="utf-8") as f_log:
                f_log.write(f"{0:<6d} | {0.0:<13.2f} | {init_io_ms:<13.2f} | {init_io_ms:<11.2f} | {0:<9d} | {float(np.min(pos_init[:, 2])):<10.4f}\n")

        if log_lvl == 'MINIMAL':
            print(f"[Newton Vulkan] Starting simulation ({props.frames} frames, {props.substeps} substeps/frame, Log Level: MINIMAL)...", flush=True)

        physics_times = []
        io_times = []
        frame_times = []
        contact_counts = []
        sim_start_time = time.perf_counter()

        for frame in range(1, props.frames + 1):
            frame_t0 = time.perf_counter()

            # Update animated mesh positions BEFORE stepping so collisions use
            # the correct geometry for this frame.
            for name, info in deform_caches.items():
                data = info["data"]
                curr_f = min(frame, data.shape[0] - 1)
                prev_f = max(0, curr_f - 1)
                curr_verts = data[curr_f].astype(np.float32)
                prev_verts = data[prev_f].astype(np.float32)
                vel_mesh = (curr_verts - prev_verts) / dt if dt > 0.0 else np.zeros_like(curr_verts)
                solver.update_mesh(info["mesh_idx"], curr_verts.flatten(), vel_mesh.flatten())

            # Physics step: solver.step internally executes the compute queues and waits on fence/idle
            t_phys0 = time.perf_counter()
            solver.step(dt, props.substeps, props.solver_iterations)
            t_phys = (time.perf_counter() - t_phys0) * 1000.0
            physics_times.append(t_phys)

            # Save resulting post-step state
            t_io0 = time.perf_counter()
            pos = solver.get_positions()
            np.save(os.path.join(cache_dir, f"frame_{frame:04d}.npy"), pos)
            t_io = (time.perf_counter() - t_io0) * 1000.0
            io_times.append(t_io)

            total_frame_t = (time.perf_counter() - frame_t0) * 1000.0
            frame_times.append(total_frame_t)
            elapsed_total = time.perf_counter() - sim_start_time

            # Metrics / Logging depending on log level
            if log_lvl == 'MINIMAL':
                if frame % 10 == 0 or frame == 1 or frame == props.frames:
                    print(f"[Vulkan] Frame {frame:3d}/{props.frames} | Step: {t_phys:5.2f} ms (IO: {t_io:4.2f} ms) | Elapsed: {elapsed_total:5.2f} s", flush=True)

            elif log_lvl == 'STANDARD':
                cnt = solver.last_contact_count
                contact_counts.append(cnt)
                min_z = float(np.min(pos[:, 2]))
                with open(log_path, "a", encoding="utf-8") as f_log:
                    f_log.write(f"{frame:<6d} | {t_phys:<13.2f} | {t_io:<13.2f} | {total_frame_t:<11.2f} | {cnt:<9d} | {min_z:<10.4f}\n")
                print(f"[Vulkan] Frame {frame:3d}/{props.frames} | step={t_phys:6.2f} ms | io={t_io:5.2f} ms | total={elapsed_total:5.2f} s | mesh_contacts={cnt}", flush=True)

            elif log_lvl == 'VERBOSE':
                cnt = solver.last_contact_count
                pp_cnt = solver.get_diag_pair_count()
                solver.reset_diag_buffers()

                contact_counts.append(cnt)
                min_z = float(np.min(pos[:, 2]))
                max_z = float(np.max(pos[:, 2]))

                frame_rec = {
                    "type": "frame",
                    "frame": frame,
                    "elapsed_s": round(elapsed_total, 4),
                    "physics_ms": round(t_phys, 2),
                    "io_ms": round(t_io, 2),
                    "frame_ms": round(total_frame_t, 2),
                    "contacts": cnt,
                    "pp_interactions": pp_cnt,
                    "min_z": round(min_z, 4),
                    "max_z": round(max_z, 4)
                }
                with open(jsonl_path, "a", encoding="utf-8") as f_j:
                    f_j.write(json.dumps(frame_rec) + "\n")

                with open(log_path, "a", encoding="utf-8") as f_log:
                    f_log.write(f"{frame:<6d} | {t_phys:<13.2f} | {t_io:<13.2f} | {total_frame_t:<11.2f} | {cnt:<9d} | {min_z:<10.4f}\n")

                print(f"[Vulkan] Frame {frame:3d}/{props.frames} | step={t_phys:6.2f} ms | io={t_io:5.2f} ms | frame={total_frame_t:6.2f} ms | total={elapsed_total:5.2f} s | mesh_contacts={cnt} | pp_interactions={pp_cnt}", flush=True)

        total_wall_s = time.perf_counter() - sim_start_time
        avg_phys = float(np.mean(physics_times))
        avg_frame = float(np.mean(frame_times))
        peak_cnt = max(contact_counts) if contact_counts else 0
        total_substeps = props.frames * props.substeps
        total_particle_steps = n_particles * total_substeps

        summary_text = (
            f"\n{'='*80}\n"
            f"SIMULATION SUMMARY: Newton Vulkan Direct XPBD\n"
            f"{'='*80}\n"
            f"Total Wall Time:       {total_wall_s:.3f} s\n"
            f"Average Physics Time:  {avg_phys:.2f} ms (min={np.min(physics_times):.2f}, max={np.max(physics_times):.2f})\n"
            f"Average Frame Time:    {avg_frame:.2f} ms (min={np.min(frame_times):.2f}, max={np.max(frame_times):.2f})\n"
            f"Total Substeps:        {total_substeps} ({props.frames} frames * {props.substeps} substeps)\n"
            f"Total Particle-Steps:  {total_particle_steps} ({n_particles} particles * {total_substeps} substeps)\n"
            f"Peak Contact Count:    {peak_cnt}\n"
            f"{'='*80}\n"
        )
        print(summary_text, flush=True)

        if log_lvl in ('STANDARD', 'VERBOSE'):
            with open(log_path, "a", encoding="utf-8") as f_log:
                f_log.write(summary_text)

        if log_lvl == 'VERBOSE':
            with open(jsonl_path, "a", encoding="utf-8") as f_j:
                f_j.write(json.dumps({
                    "type": "summary",
                    "total_wall_s": round(total_wall_s, 4),
                    "avg_physics_ms": round(avg_phys, 2),
                    "min_physics_ms": round(float(np.min(physics_times)), 2),
                    "max_physics_ms": round(float(np.max(physics_times)), 2),
                    "avg_frame_ms": round(avg_frame, 2),
                    "peak_contacts": peak_cnt,
                    "total_substeps": total_substeps,
                    "total_particle_steps": total_particle_steps
                }) + "\n")

        # 5. Create or Update Blender Mesh Object
        mesh_name = "NewtonVulkanParticles"
        if mesh_name in bpy.data.objects:
            obj = bpy.data.objects[mesh_name]
        else:
            obj = bpy.data.objects.new(mesh_name, bpy.data.meshes.new(mesh_name))
            scene.collection.objects.link(obj)

        old_mesh = obj.data
        new_mesh = bpy.data.meshes.new(mesh_name)
        first_frame_data = np.load(os.path.join(cache_dir, "frame_0000.npy"))
        
        new_mesh.vertices.add(len(first_frame_data))
        new_mesh.vertices.foreach_set("co", first_frame_data.ravel())
        new_mesh.update()
        obj.data = new_mesh
        if old_mesh:
            bpy.data.meshes.remove(old_mesh)

        mod = obj.modifiers.get("MeshToPoints")
        if not mod:
            mod = obj.modifiers.new("MeshToPoints", type='NODES')
            node_group = bpy.data.node_groups.new(name="MeshToPointsGroupVK", type="GeometryNodeTree")
            mod.node_group = node_group

            node_group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            node_group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

            input_node = node_group.nodes.new("NodeGroupInput")
            output_node = node_group.nodes.new("NodeGroupOutput")
            mtop_node = node_group.nodes.new("GeometryNodeMeshToPoints")
            mtop_node.inputs['Radius'].default_value = radius_mean

            node_group.links.new(input_node.outputs[0], mtop_node.inputs[0])
            node_group.links.new(mtop_node.outputs[0], output_node.inputs[0])

        self.report({'INFO'}, f"Vulkan Simulation Complete! Cached to Disk ({cache_dir})")
        return {'FINISHED'}

classes = (
    NEWTONVK_PG_colliderItem,
    NEWTONVK_PG_settings,
    NEWTONVK_PT_panel,
    NEWTONVK_OT_add_collider,
    NEWTONVK_OT_capture_deformation,
    NEWTONVK_OT_remove_collider,
    NEWTONVK_OT_simulate
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.newton_vk_settings = bpy.props.PointerProperty(type=NEWTONVK_PG_settings)
    
    if load_npy_frame_vk not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(load_npy_frame_vk)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.newton_vk_settings
    
    if load_npy_frame_vk in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(load_npy_frame_vk)

if __name__ == "__main__":
    register()
