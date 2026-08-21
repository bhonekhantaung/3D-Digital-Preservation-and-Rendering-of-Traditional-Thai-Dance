"""Export selected Blender mesh objects as a dense surface point cloud.

Run this inside the original Blender file:
1. In Object Mode, select every dancer mesh (body, costume, headdress, ornaments).
2. Do not select the floor, backdrop, lights, camera, or armature.
3. Open Blender's Scripting workspace, load this file, and press Run Script.

The output is a binary PLY containing positions, normals, and UV-texture colors.
Geometry is sampled from the evaluated meshes, so armature poses and modifiers are applied.
"""

import os
import struct

import bpy
import numpy as np


POINT_COUNT = 300_000
RANDOM_SEED = 20260821
EXCLUDED_NAMES = {"CheckerFloor", "AutoPivot", "AutoCamera", "SunLight"}

# "//" means the folder containing the .blend file. Change this to an absolute
# path if the Blender file has not been saved yet.
OUTPUT_PATH = bpy.path.abspath("//dancer_exact_pointcloud_textured.ply")


def linear_to_srgb(rgb):
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


def material_rgb(obj, material_index):
    if 0 <= material_index < len(obj.material_slots):
        material = obj.material_slots[material_index].material
        if material is not None:
            return linear_to_srgb(material.diffuse_color[:3])
    return np.array([0.72, 0.72, 0.72], dtype=np.float64)


def material_for_index(obj, material_index):
    if 0 <= material_index < len(obj.material_slots):
        return obj.material_slots[material_index].material
    return None


def find_base_color_image(material):
    """Find the first image texture upstream of Principled Base Color."""
    if material is None or not material.use_nodes or material.node_tree is None:
        return None

    principled_nodes = [
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    ]
    for principled in principled_nodes:
        base_color = principled.inputs.get("Base Color")
        if base_color is None or not base_color.is_linked:
            continue

        pending = [link.from_node for link in base_color.links]
        visited = set()
        while pending:
            node = pending.pop(0)
            if node in visited:
                continue
            visited.add(node)
            if node.type == "TEX_IMAGE" and node.image is not None:
                return node.image
            for input_socket in node.inputs:
                if input_socket.is_linked:
                    pending.extend(link.from_node for link in input_socket.links)
    return None


def load_texture(image):
    width, height = int(image.size[0]), int(image.size[1])
    if width <= 0 or height <= 0:
        return None
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    # Blender exposes color texture pixels in scene-linear space; PLY viewers expect sRGB.
    pixels = pixels.copy()
    pixels[:, :, :3] = linear_to_srgb(pixels[:, :, :3]).astype(np.float32)
    return pixels


def collect_world_triangles(objects):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    normals = []
    colors = []
    uvs = []
    texture_ids = []
    areas = []
    textures = []
    image_to_texture_id = {}

    for source_obj in objects:
        obj = source_obj.evaluated_get(depsgraph)
        mesh = obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            mesh.calc_loop_triangles()
            world = obj.matrix_world
            normal_matrix = world.to_3x3().inverted().transposed()

            world_vertices = np.asarray(
                [tuple(world @ vertex.co) for vertex in mesh.vertices],
                dtype=np.float64,
            )
            world_normals = np.asarray(
                [tuple((normal_matrix @ vertex.normal).normalized()) for vertex in mesh.vertices],
                dtype=np.float64,
            )
            uv_data = mesh.uv_layers.active.data if mesh.uv_layers.active is not None else None

            for tri in mesh.loop_triangles:
                ids = np.asarray(tri.vertices, dtype=np.int64)
                triangle = world_vertices[ids]
                area = 0.5 * np.linalg.norm(
                    np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                )
                if not np.isfinite(area) or area <= 1e-14:
                    continue

                material_index = mesh.polygons[tri.polygon_index].material_index
                material = material_for_index(source_obj, material_index)
                image = find_base_color_image(material)
                texture_id = -1
                if image is not None and uv_data is not None:
                    image_key = image.as_pointer()
                    if image_key not in image_to_texture_id:
                        texture = load_texture(image)
                        if texture is not None:
                            image_to_texture_id[image_key] = len(textures)
                            textures.append(texture)
                    texture_id = image_to_texture_id.get(image_key, -1)

                vertices.append(triangle)
                normals.append(world_normals[ids])
                colors.append(material_rgb(source_obj, material_index))
                if uv_data is not None:
                    uvs.append(np.asarray([tuple(uv_data[loop].uv) for loop in tri.loops]))
                else:
                    uvs.append(np.zeros((3, 2), dtype=np.float64))
                texture_ids.append(texture_id)
                areas.append(area)
        finally:
            obj.to_mesh_clear()

    if not areas:
        raise RuntimeError("The selected objects did not contain any usable mesh triangles.")

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
        np.asarray(colors, dtype=np.float64),
        np.asarray(uvs, dtype=np.float64),
        np.asarray(texture_ids, dtype=np.int32),
        np.asarray(areas, dtype=np.float64),
        textures,
    )


def sample_surface(
    triangles, vertex_normals, triangle_colors, triangle_uvs,
    triangle_texture_ids, areas, textures, count, seed
):
    rng = np.random.default_rng(seed)
    triangle_ids = rng.choice(len(areas), size=count, p=areas / areas.sum())

    # Uniform barycentric sampling over triangle area.
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    weights = np.column_stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2])

    chosen_triangles = triangles[triangle_ids]
    points = np.sum(chosen_triangles * weights[:, :, None], axis=1)

    chosen_normals = vertex_normals[triangle_ids]
    sampled_normals = np.sum(chosen_normals * weights[:, :, None], axis=1)
    lengths = np.linalg.norm(sampled_normals, axis=1, keepdims=True)
    sampled_normals /= np.maximum(lengths, 1e-12)

    sampled_rgb = triangle_colors[triangle_ids].copy()
    sampled_texture_ids = triangle_texture_ids[triangle_ids]
    sampled_uvs = np.sum(triangle_uvs[triangle_ids] * weights[:, :, None], axis=1)

    for texture_id, texture in enumerate(textures):
        point_ids = np.flatnonzero(sampled_texture_ids == texture_id)
        if len(point_ids) == 0:
            continue
        height, width = texture.shape[:2]
        uv = np.mod(sampled_uvs[point_ids], 1.0)
        pixel_x = np.floor(uv[:, 0] * width).astype(np.int64).clip(0, width - 1)
        pixel_y = np.floor(uv[:, 1] * height).astype(np.int64).clip(0, height - 1)
        sampled_rgb[point_ids] = texture[pixel_y, pixel_x, :3]

    rgb = np.rint(sampled_rgb * 255.0).clip(0, 255).astype(np.uint8)
    return points.astype(np.float32), sampled_normals.astype(np.float32), rgb


def write_binary_ply(path, points, normals, colors):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    record = struct.Struct("<ffffffBBB")
    with open(path, "wb") as handle:
        handle.write(header)
        for point, normal, color in zip(points, normals, colors):
            handle.write(
                record.pack(
                    float(point[0]), float(point[1]), float(point[2]),
                    float(normal[0]), float(normal[1]), float(normal[2]),
                    int(color[0]), int(color[1]), int(color[2]),
                )
            )


selected_meshes = [
    obj
    for obj in bpy.context.selected_objects
    if obj.type == "MESH" and obj.name not in EXCLUDED_NAMES
]
if not selected_meshes:
    raise RuntimeError(
        "Select the dancer's mesh objects first. In your scene, select 'mesh_node', "
        "not 'CheckerFloor'."
    )

print("Selected dancer meshes:")
for selected in selected_meshes:
    print("  ", selected.name)

(
    triangles,
    triangle_normals,
    triangle_colors,
    triangle_uvs,
    triangle_texture_ids,
    triangle_areas,
    textures,
) = collect_world_triangles(selected_meshes)
print(f"Collected {len(triangles):,} evaluated triangles")
print(f"Loaded {len(textures)} UV color texture(s)")

points, normals, colors = sample_surface(
    triangles,
    triangle_normals,
    triangle_colors,
    triangle_uvs,
    triangle_texture_ids,
    triangle_areas,
    textures,
    POINT_COUNT,
    RANDOM_SEED,
)
write_binary_ply(OUTPUT_PATH, points, normals, colors)

print(f"Exported {len(points):,} surface points to: {OUTPUT_PATH}")


def draw_export_result(self, context):
    self.layout.label(text=f"Exported {len(points):,} points")
    self.layout.label(text=f"Loaded {len(textures)} UV color texture(s)")
    self.layout.label(text=os.path.basename(OUTPUT_PATH))


bpy.context.window_manager.popup_menu(
    draw_export_result,
    title="Point-cloud export complete",
    icon="CHECKMARK",
)
