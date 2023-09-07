import open3d as o3d
import numpy as np

from gibson2.utils.mesh_util import xyzw2wxyz, quat2rotmat
from transforms3d.euler import euler2quat 

def visualize_np_point_cloud(pc):
    pc[:, [0,1,2]] = pc[:, [1,2,0]]
    pc[:,[0,1]] *= -1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd])

def get_point_clouds(env, visualize=False, mode="world_coordinate"):
    eye_pos, eye_orn = env.robots[0].eyes.get_position(), env.robots[0].eyes.get_orientation()
    camera_in_wf = quat2rotmat(xyzw2wxyz(eye_orn))
    camera_in_wf[:3,3] = eye_pos

    # Transforming coordinates of points from opengl frame to camera frame
    camera_in_openglf = quat2rotmat(euler2quat(np.pi / 2.0, 0, -np.pi / 2.0))

    # Pose of the simulated robot in world frame
    robot_pos, robot_orn = env.robots[0].get_position(), env.robots[0].get_orientation()
    robot_in_wf = quat2rotmat(xyzw2wxyz(robot_orn))
    robot_in_wf[:3, 3] = robot_pos

    # Pose of the camera in robot frame
    cam_in_robot_frame = np.dot(np.linalg.inv(robot_in_wf), camera_in_wf)

    [td_image] = env.simulator.renderer.render(modes=('3d'))
    point_in_cf = np.dot(td_image.reshape(-1,4), camera_in_openglf.T)
    point_in_rf = np.dot(point_in_cf,cam_in_robot_frame.T)
    pcd_distance = np.linalg.norm(point_in_rf, axis=1)
    point_in_rf = point_in_rf[pcd_distance <= 10]
    point_in_wf = np.dot(point_in_rf, robot_in_wf.T)
    
    
    #pcd = pcd.voxel_down_sample(voxel_size=0.05)

    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if visualize == True:
        visualize_np_point_cloud(point_in_wf[:,[0,1,2]])
    #np.save('test_pc_3', point_in_wf[:,[0,1,2]])
    if mode == "world_coordinate":
        return point_in_wf[:,[0,1,2]]
    if mode == "robot_coordinate":
        return point_in_rf[:, [0,1,2]]

def get_min_max_pcd(pcd, min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None):
    '''
    max_x = np.max(pcd[:,0])
    max_y = np.max(pcd[:,1])
    max_z = np.max(pcd[:,2])

    min_x = np.min(pcd[:,0])
    min_y = np.min(pcd[:,1])
    min_z = np.min(pcd[:,2])

    return min_x,max_x,min_y,max_y,min_z,max_z
    '''
    max_x_prev = -np.inf if max_x == None else max_x
    max_y_prev = -np.inf if max_y == None else max_y
    max_z_prev = -np.inf if max_z == None else max_z

    min_x_prev = np.inf if min_x == None else min_x
    min_y_prev = np.inf if min_y == None else min_y
    min_z_prev = np.inf if min_z == None else min_z

    max_x = np.max(pcd[:,0])
    max_y = np.max(pcd[:,1])
    max_z = np.max(pcd[:,2])

    min_x = np.min(pcd[:,0])
    min_y = np.min(pcd[:,1])
    min_z = np.min(pcd[:,2])

    max_x = max(max_x_prev, max_x)
    max_y = max(max_y_prev, max_y)
    max_z = max(max_z_prev, max_z)

    min_x = min(min_x_prev, min_x)
    min_y = min(min_y_prev, min_y)
    min_z = min(min_z_prev, min_z)

    return min_x,max_x,min_y,max_y,min_z,max_z
    
    for pcd in pcds:
        max_x = np.max(pcd[:,0])
        max_y = np.max(pcd[:,1])
        max_z = np.max(pcd[:,2])


        min_x = np.min(pcd[:,0])
        min_y = np.min(pcd[:,1])
        min_z = np.min(pcd[:,2])

        '''
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
        '''
    return min_x,max_x,min_y,max_y,min_z,max_z
    
    