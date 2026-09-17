import cubvh
import torch
import numpy as np
import trimesh
from diso import DiffDMC
import argparse
from tqdm import tqdm
import os
import math
from os.path import join as pjoin 
from to_watertight_mesh import generate_dense_grid_points
from sharp_sample import save_vertices_as_ply_open3d
import bpy
import bmesh
import gc
import open3d as o3d
import time
import numpy as np
import fpsample
from pysdf import SDF
import io 

def build_visual_from_mesh(base_mesh):
    """
    Returns a `TextureVisuals` or `ColorVisuals` object from base_mesh
    if possible. Returns None if no material information is available.
    """
    if not hasattr(base_mesh, 'visual'):
        return None

    visual = base_mesh.visual

    # Case 1: Textured mesh
    if isinstance(visual, trimesh.visual.texture.TextureVisuals):
        if visual.uv is not None and hasattr(visual, 'material'):
            material = visual.material
            if getattr(material, 'image', None) is not None:
                return trimesh.visual.texture.TextureVisuals(
                    uv=visual.uv,
                    image=material.image,
                    material=material  # Retain original material
                )
    return None  # No usable visual

def remesh(grid_xyz, grid_size, base_mesh, resolution):
    eps = 2 / resolution
    # normalize mesh to [-1,1]
    vertices = base_mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) / 2
    scale = 2.0 / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    f = cubvh.cuBVH(torch.as_tensor(vertices, dtype=torch.float32, device='cuda'), torch.as_tensor(base_mesh.faces, dtype=torch.float32, device='cuda')) # build with numpy.ndarray/torch.Tensor
    #f = cubvh.cuBVH(vertices.astype(np.float32), base_mesh.faces.astype(np.float32)) # build with numpy.ndarray/torch.Tensor
    grid_udf, _ ,_= f.unsigned_distance(grid_xyz, return_uvw=False) 
    grid_udf = grid_udf.view((grid_size[0], grid_size[1], grid_size[2]))

    grid_udf = grid_udf.cude().contiguous()
    # grid_udf = grid_udf.to(device=grid_udf.device, dtype=torch.float32).contiguous()
    diffdmc = DiffDMC(dtype=torch.float32).to(device=grid_udf.device)
    vertices, faces = diffdmc(grid_udf, isovalue=eps, normalize=False)
    bbox_min = np.array((-1.05, -1.05, -1.05))
    bbox_max= np.array((1.05, 1.05, 1.05))
    bbox_size = bbox_max - bbox_min
    vertices = (vertices + 1) / grid_size[0] * bbox_size[0] + bbox_min[0]
    # if len(vertices) == 0: #problematic mesh
    #     return None
    visual=build_visual_from_mesh(base_mesh)

    mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy(), visual=visual)
    # keep the max component of the extracted mesh
    components = mesh.split(only_watertight=False)
    bbox = []
    for c in components:
        bbmin = c.vertices.min(0)
        bbmax = c.vertices.max(0)
        bbox.append((bbmax - bbmin).max())
    max_component = np.argmax(bbox)
    mesh = components[max_component]
    return mesh
    
def import_trimesh_to_blender(mesh: trimesh.Trimesh, name="ImportedMesh"):
    # Create a new mesh and object
    new_mesh = bpy.data.meshes.new(name)
    new_object = bpy.data.objects.new(name, new_mesh)
    
    # Link the object to the scene
    bpy.context.collection.objects.link(new_object)

    # Set the object as active
    bpy.context.view_layer.objects.active = new_object
    bpy.context.view_layer.objects.active.select_set(True)

    # Create mesh data
    new_mesh.from_pydata(mesh.vertices.tolist(), [], mesh.faces.tolist())
    new_mesh.update()

    return new_object

def process_mesh(mesh, point_number, ply_output_path, npz_output_path, sharpness_threshold):
    # 导入mesh
    obj = import_trimesh_to_blender(mesh)
    bpy.ops.object.mode_set(mode='EDIT')

    # 确保在边模式下选择
    bpy.ops.mesh.select_mode(type="EDGE")

    # 选择Sharp Edge
    bpy.ops.mesh.edges_select_sharp(sharpness=sharpness_threshold)

    # 打印Sharp Edge
    bpy.ops.object.mode_set(mode='OBJECT')  # 临时切换回Object模式以访问选择状态
    # mesh = obj.data

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    sharp_edges = [edge for edge in bm.edges if edge.select]


# 收集 sharp edges 的顶点对
    sharp_edges_vertices = []
    link_normal1 =[]
    link_normal2 = []
    sharp_edges_angle = []
    #不重复点集
    vertices_set = set()
    for edge in sharp_edges:
        vertices_set.update(edge.verts[:]) #不重复点集
        

        sharp_edges_vertices.append([edge.verts[0].index, edge.verts[1].index])# 收集 sharp edges 的顶点对 index

        normal1 = edge.link_faces[0].normal
        normal2 = edge.link_faces[1].normal

        link_normal1.append(normal1)
        link_normal2.append(normal2)

        if normal1.length==0.0 or normal2.length==0.0:
            sharp_edges_angle.append(0.0)
        # Compute the angle between the two normals
        else:
            sharp_edges_angle.append(math.degrees(normal1.angle(normal2)))

    vertices=[]
    vertices_index=[]
    vertices_normal=[]

    for vertice in vertices_set:
        vertices.append(vertice.co)
        vertices_index.append(vertice.index)
        vertices_normal.append(vertice.normal)


    vertices = np.array(vertices)
    vertices_index = np.array(vertices_index)
    vertices_normal = np.array(vertices_normal)

    sharp_edges_count = np.array(len(sharp_edges))
    sharp_edges_angle_array = np.array(sharp_edges_angle)
    if sharp_edges_count>0:
        sharp_edge_link_normal = np.array(np.concatenate([link_normal1,link_normal2], axis=1))
        nan_mask = np.isnan(sharp_edge_link_normal)
        # 使用布尔索引将检测到的NaN值替换为0
        sharp_edge_link_normal = np.where(nan_mask, 0, sharp_edge_link_normal)
        
        nan_mask = np.isnan(vertices_normal)
        # 使用布尔索引将检测到的NaN值替换为0
        vertices_normal = np.where(nan_mask, 0, vertices_normal)

    # 转换为 numpy 数组
    sharp_edges_vertices_array = np.array(sharp_edges_vertices)


    if sharp_edges_count>0:
        num_target_sharp_vertices = point_number // 2
        sharp_edge_length = sharp_edges_count
        sharp_edges_vertices_pair = sharp_edges_vertices_array
        sharp_vertices_pair = mesh.vertices[sharp_edges_vertices_pair] # 顶点对坐标 1225，2，3
        # sharp_edge_link_normal = data['sharp_edge_link_normal']
        epsilon = 1e-4  # 一个小的数值
        edge_normal =  0.5*sharp_edge_link_normal[:,:3] + 0.5 * sharp_edge_link_normal[:,3:]
        norms = np.linalg.norm(edge_normal, axis=1, keepdims=True)
        norms = np.where(norms > epsilon, norms, epsilon)
        edge_normal = edge_normal / norms # 
        known_vertices = vertices # 不重复的sharp vertices
        known_vertices_normal = vertices_normal
        known_vertices = np.concatenate([known_vertices,known_vertices_normal], axis=1)

        num_known_vertices = known_vertices.shape[0] # 不重复的sharp vertices的个数
        if  num_known_vertices<num_target_sharp_vertices: # 如果已知的顶点数量小于目标的顶点数量
            num_new_vertices = num_target_sharp_vertices - num_known_vertices 
            if num_new_vertices >= sharp_edge_length: # 如果需要补充的顶点数量大于sharp edge的个数,说明每一条sharp edge之间至少需要插值一个新的顶点
                num_new_vertices_per_pair = num_new_vertices // sharp_edge_length # 计算每个顶点对对（边）均匀分配的顶点数
                new_vertices = np.zeros((sharp_edge_length, num_new_vertices_per_pair, 6)) # 初始化新顶点数组 在每一个shapr edge中插值

                start_vertex = sharp_vertices_pair[:, 0]
                end_vertex = sharp_vertices_pair[:, 1]
                for j in range(1, num_new_vertices_per_pair+1):
                    t = j / float(num_new_vertices_per_pair+1)
                    new_vertices[:, j - 1 , :3] = (1 - t) * start_vertex + t * end_vertex
                    
                    new_vertices[:, j - 1 , 3:] = edge_normal # 边内的normal是一样的
                new_vertices= new_vertices.reshape(-1,6)

                remaining_vertices = num_new_vertices % sharp_edge_length # 计算需要额外分配的顶点数。
                if remaining_vertices>0:
                    rng = np.random.default_rng()
                    ind = rng.choice(sharp_edge_length, remaining_vertices, replace=False)
                    new_vertices_remain = np.zeros((remaining_vertices, 6))# 初始化新顶点数组
                    start_vertex = sharp_vertices_pair[ind, 0]
                    end_vertex = sharp_vertices_pair[ind, 1]
                    t = np.random.rand(remaining_vertices).reshape(-1,1)
                    new_vertices_remain[:,:3] = (1 - t) * start_vertex + t * end_vertex

                    edge_normal =  0.5*sharp_edge_link_normal[ind,:3] + 0.5 * sharp_edge_link_normal[ind,3:]
                    edge_normal = edge_normal / np.linalg.norm(edge_normal,axis=1,keepdims=True)
                    new_vertices_remain[:,3:] = edge_normal

                    new_vertices = np.concatenate([new_vertices,new_vertices_remain], axis=0)
            else:
                remaining_vertices = num_new_vertices % sharp_edge_length # 计算需要额外分配的顶点数 
                if remaining_vertices>0:
                    rng = np.random.default_rng() 
                    ind = rng.choice(sharp_edge_length, remaining_vertices, replace=False)
                    new_vertices_remain = np.zeros((remaining_vertices, 6))# 初始化新顶点数组
                    start_vertex = sharp_vertices_pair[ind, 0]
                    end_vertex = sharp_vertices_pair[ind, 1]
                    t = np.random.rand(remaining_vertices).reshape(-1,1)
                    new_vertices_remain[:,:3] = (1 - t) * start_vertex + t * end_vertex

                    edge_normal =  0.5*sharp_edge_link_normal[ind,:3] + 0.5 * sharp_edge_link_normal[ind,3:]
                    edge_normal = edge_normal / np.linalg.norm(edge_normal,axis=1,keepdims=True)
                    new_vertices_remain[:,3:] = edge_normal

                    new_vertices = new_vertices_remain


            target_vertices = np.concatenate([new_vertices,known_vertices], axis=0)
        else:
            target_vertices = known_vertices

        sharp_surface = target_vertices # sharp表面点的位置及其normal

        sharp_surface_points = sharp_surface[:,:3]

        sharp_near_surface_points= [
                        sharp_surface_points + np.random.normal(scale=0.001, size=(len(sharp_surface_points), 3)),
                        sharp_surface_points + np.random.normal(scale=0.005, size=(len(sharp_surface_points),3)),
                        sharp_surface_points + np.random.normal(scale=0.007, size=(len(sharp_surface_points),3)),
                        sharp_surface_points + np.random.normal(scale=0.01, size=(len(sharp_surface_points),3))
            ]
        sharp_near_surface_points = np.concatenate(sharp_near_surface_points)
        
        f = SDF(mesh.vertices, mesh.faces); # (num_vertices, 3) and (num_faces, 3)
        sharp_sdf = f(sharp_near_surface_points).reshape(-1,1)

        sharp_near_surface = np.concatenate([sharp_near_surface_points, sharp_sdf], axis=1) # sharp 近表面点的位置及其sdf
        # sample points near the surface and in the space within bounds
        
        coarse_surface_points, faces = mesh.sample(200000, return_index=True)
        normals = mesh.face_normals[faces]
        coarse_surface = np.concatenate([coarse_surface_points, normals], axis=1)

        coarse_near_surface_points= [
                    coarse_surface_points + np.random.normal(scale=0.001, size=(len(coarse_surface_points), 3)),
                    coarse_surface_points + np.random.normal(scale=0.005, size=(len(coarse_surface_points),3)),
        ]

        coarse_near_surface_points = np.concatenate(coarse_near_surface_points)
        space_points = np.random.uniform(-1.05, 1.05, (200000, 3))
        rand_points = np.concatenate([coarse_near_surface_points, space_points], axis=0)
        coarse_sdf = f(rand_points).reshape(-1,1)

        rand_points = np.concatenate([rand_points, coarse_sdf], axis=1)  # rand points包括coarse 近表面的和空间中均匀分布的点的位置及其sdf
        
        coarse_surface_points, faces = mesh.sample(200000, return_index=True)
        normals = mesh.face_normals[faces]
        coarse_surface = np.concatenate([coarse_surface_points, normals], axis=1)

        fps_coarse_surface_list=[]
        for _ in range(1):
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(coarse_surface_points, num_target_sharp_vertices, h=5)
            fps_coarse_surface = coarse_surface[kdline_fps_samples_idx].reshape(-1,1,6)
            fps_coarse_surface_list.append(fps_coarse_surface) 
        fps_coarse_surface = np.concatenate(fps_coarse_surface_list, axis=1)
        
        fps_sharp_surface_list=[]
        if sharp_surface.shape[0]>num_target_sharp_vertices:
            for _ in range(1):
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(sharp_surface_points, num_target_sharp_vertices, h=5)
                fps_sharp_surface = sharp_surface[kdline_fps_samples_idx].reshape(-1,1,6)
                fps_sharp_surface_list.append(fps_sharp_surface) 

            fps_sharp_surface = np.concatenate(fps_sharp_surface_list, axis=1)
        else:
            fps_sharp_surface = sharp_surface[:,None]

        sharp_surface[np.isinf(sharp_surface)] = 1
        sharp_surface[np.isnan(sharp_surface)] = 1
        fps_coarse_surface[np.isinf(fps_coarse_surface)] = 1
        fps_coarse_surface[np.isnan(fps_coarse_surface)] = 1
        np.savez(
            npz_output_path,
            fps_sharp_surface = fps_sharp_surface.astype(np.float32),
            sharp_near_surface = sharp_near_surface.astype(np.float32),
            fps_coarse_surface = fps_coarse_surface.astype(np.float32),
            rand_points = rand_points.astype(np.float32),
        )
    else:
        print(f'{ply_output_path}'+ ' no sharp edges!')



    # if sharp_edges_count>0:
    # # 保存顶点为PLY文件
    #     save_vertices_as_ply_open3d(sharp_surface[:,:3], ply_output_path)

    # 删除导入的对象
    bm.free()
    del sharp_edges_angle_array,vertices,sharp_edges_count,sharp_edges_vertices_array,sharp_edges_vertices,sharp_edges_angle,sharp_edges
    bpy.data.objects.remove(obj, do_unlink=True)
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        default="512",
        type= int,
        help="resolution",
    )
    
    parser.add_argument(
        "--data_dir",
        type= str,
        default="/home/dcor/inbargat1/datasets/partnet/03001627",
        help="shapenet mesh data directory",
    )
    
    parser.add_argument(
        "--obj_out_dir",
        type= str,
        default="sharp_edge_sampling/remesh",
        help="watertight obj files out directory",
    )
    
    parser.add_argument(
        "--angle_threshold",
        type=int,
        default=15,
        help="angle threhold for sharp edge",
    )
    parser.add_argument(
        "--point_number",
        type=int,
        default=65536,
        help="number of points",
    )
    
    parser.add_argument(
        "--sharp_point_path",
        type=str,
        default="sharp_edge_sampling/sharp_point_ply",
        help="ply path",
    )

    parser.add_argument(
        "--sample_path",
        type= str,
        default="sharp_edge_sampling/sample",
        help="npz path",
    )
    args, extras = parser.parse_known_args()
    mesh_const_str = "models/model_normalized.obj"
    sharp_point_path = pjoin(args.sharp_point_path, "shapenet", "chairs")
    sample_path = pjoin(args.sample_path, "shapenet", "chairs")
    obj_path = pjoin(args.obj_out_dir, "shapenet", "chairs")
    os.makedirs(sharp_point_path, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)
    model_names = os.listdir(args.data_dir) 
    print(f'mesh count: {len(model_names)}')
    sharpness_threshold = math.radians(args.angle_threshold)
    for name in tqdm(model_names):
        mesh_path = pjoin(args.data_dir, name, mesh_const_str)
        ply_output_path = pjoin(sharp_point_path, f'{name}.ply')
        npz_output_path = pjoin(sample_path,  f'{name}.npz')
        obj_output_path = pjoin(obj_path, name, f'{name}.obj')
        if os.path.isfile(obj_output_path):
            continue
        print(f'processing file: {mesh_path}')
        mesh = trimesh.load(mesh_path, process=True, force='mesh')
        # print(mesh)
        # print("Vertices:", len(mesh.vertices))
        # print("Faces:", len(mesh.faces))
        # If the mesh is in a scene for some reason
        # if isinstance(mesh, trimesh.Scene):
        #     print("here")
        #     mesh = mesh.
        grid_xyz, grid_size = generate_dense_grid_points(resolution = args.resolution)
        grid_xyz = torch.FloatTensor(grid_xyz).cuda()
        remeshed = remesh(grid_xyz, grid_size, mesh, args.resolution)
        # if remeshed == None:
        #     continue
        #else: 
        os.makedirs(pjoin(obj_path, name), exist_ok=True)
        remeshed.export(obj_output_path)

        # try:
        #     process_mesh(remeshed, args.point_number, ply_output_path,npz_output_path, sharpness_threshold)
        #     gc.collect()
        # except Exception as e:
        #     print(f"ERROR: in processing path: {npz_output_path}. Error: {e}")
        #     gc.collect()
        #     raise e
        
        
        
        
    
    
    
    
