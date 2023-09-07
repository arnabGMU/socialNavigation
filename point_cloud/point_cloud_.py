import open3d as o3d
import numpy as np
import os

from scipy.spatial.transform import Rotation as R_scipy
from utils.utils import transformation_matrix
from matplotlib import pyplot as plt
from point_cloud.point_cloud import get_point_clouds

def get_point_clouds(env, state):
    cam1 = o3d.camera.PinholeCameraIntrinsic()
    cam1.intrinsic_matrix =  env.simulator.renderer.get_intrinsics()
    pcds = []
    
    rgb = state['rgb']
    depth = state['depth']
    plt.figure()
    plt.imshow(rgb)
    plt.show()
    plt.imshow(depth)
    plt.show()
    return
    
    color_images = sorted(os.listdir(rgb_path), key=lambda name: int(name.split('.')[0]))
    depth_images = sorted(os.listdir(depth_path), key=lambda name: int(name.split('.')[0]))
    
    for i in range(len(color_images)):
        #theta = np.radians(i* -30)
        #c, s = np.cos(theta), np.sin(theta)
        #R = np.array(((c, 0, s), (0, 1, 0), (-s,0,c)))
        r = R_scipy.from_euler('zxy', [0,0,i*-30], degrees=True)
        R = r.as_dcm()

        if look_down == True:
            #theta = np.radians(-75)
            #c_look_down, s_look_down = np.cos(theta), np.sin(theta)
            #R_look_down = np.array(((c_look_down, -s_look_down, 0), (s_look_down, c_look_down, 0), (0,0,1)))
            #R = np.dot(R_look_down, R)

            r = R_scipy.from_euler('zxy', [0,-45,i*-30], degrees=True)
            R = r.as_dcm()

        
        #t = np.zeros((3,1))
        t = np.array([current_point[0]-starting_point[0], current_point[1]- starting_point[1], current_point[2] - starting_point[2]])
        T = transformation_matrix(R,t)
        
        color_image_path = os.path.join(rgb_path, color_images[i])
        depth_image_path = os.path.join(depth_path, depth_images[i])

        color_image = o3d.io.read_image(color_image_path)
        depth_image = o3d.io.read_image(depth_image_path)

        #color_image = o3d.geometry.Image((color_images[i]).astype(np.uint8))
        #depth_image = o3d.geometry.Image((depth_images[i]).astype(np.uint8))
        
        #print(np.asarray(color_image))
        #print(np.asarray(depth_image))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image
        )    
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            cam1)
        
        pcd_np = (np.asarray(pcd.points)*10000)
        #print(pcd_np.shape)
        pcd_np_xz = (np.asarray(pcd.points)*10000)[:,[0,2]]
        pcd_distance = np.linalg.norm(pcd_np_xz,axis=1)
        pcd_np = pcd_np[pcd_distance < 7] # 7
        #print(pcd_np.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        #pcd = pcd.voxel_down_sample(voxel_size=0.000001)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd.transform(T)
        #o3d.visualization.draw_geometries([pcd])
        pcds.append(pcd)
    return pcds

def get_min_max_pcd(pcds, point, scale=1, index=None):
    max_x = -np.inf
    max_y = -np.inf
    max_z = -np.inf

    min_x = np.inf
    min_y = np.inf
    min_z = np.inf
    
    for pcd in pcds:
        pc_points = (np.asarray(pcd.points)*scale) + point

        try:
            max_x_pc = np.max(pc_points[:,0])
            max_y_pc = np.max(pc_points[:,1])
            max_z_pc = np.max(pc_points[:,2])
        except:
            continue

        min_x_pc = np.min(pc_points[:,0])
        min_y_pc = np.min(pc_points[:,1])
        min_z_pc = np.min(pc_points[:,2])

        if max_x_pc > max_x:
            max_x = max_x_pc
        if max_y_pc > max_y:
            max_y = max_y_pc
        if max_z_pc > max_z:
            max_z = max_z_pc

        if min_x_pc < min_x:
            min_x = min_x_pc
        if min_y_pc < min_y:
            min_y = min_y_pc
        if min_z_pc < min_z:
            min_z = min_z_pc 
    return min_x,max_x,min_y,max_y,min_z,max_z
