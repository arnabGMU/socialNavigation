import numpy as np
import networkx as nx
import scipy
import math

from matplotlib import pyplot as plt
from frontier.frontier import show_frontiers, Frontier

from collections import defaultdict


def visualize_occupancy_grid(occupancy_grid, robot_position, goal, store=False, show=False, output_path=None):
    plt.plot(robot_position[0], robot_position[1], marker="o", markersize=10, alpha=0.8)
    plt.plot(goal[0], goal[1], marker="o", markersize=10, alpha=0.8)
    plt.imshow(occupancy_grid, cmap='gray') #plt.show()
    if show == True:
        plt.show()
    if store == True:
        plt.savefig(output_path)
    plt.close()

def fill_invisible_cells_close_to_the_robot(occupancy_grid, robot_x, robot_y):
    top = robot_y - 21
    for i in range(21):
        row = robot_y - i
        if row == 0:
            top = row
            break
        if occupancy_grid[row][robot_x] != 0.5:
            top = row
            break
    if top > robot_y-7:
        top = max(0, robot_y-7)
    
    bottom = robot_y + 21
    for i in range(21):
        row = robot_y + i
        if row == occupancy_grid.shape[0]-1:
            bottom = row
            break
        if occupancy_grid[row][robot_x] != 0.5:
            bottom = row
            break
    if bottom < robot_y+7:
        bottom = min(occupancy_grid.shape[0]-1, robot_y+7)
    
    left = robot_x - 21
    for i in range(21):
        col = robot_x - i
        if col == 0:
            left= col
            break
        if occupancy_grid[robot_y][col] != 0.5:
            left = col
            break
    if left > robot_x-7:
        left = max(0, robot_x-7)
 
    right = robot_x + 21
    for i in range(21):
        col = robot_x +i
        if col == occupancy_grid.shape[1]-1:
            right = col
            break
        if occupancy_grid[robot_y][col] != 0.5:
            right= col
            break
 
    if right < robot_x+7:
        right = min(occupancy_grid.shape[1]-1, robot_x+7)
    
    for i in range(top, bottom+1):
        for j in range(left, right+1):
            if occupancy_grid[i][j] == 0.5:
                occupancy_grid[i][j] = 1
    
    return occupancy_grid
    '''
    pos_x = robot_x - 21
    pos_y = robot_y - 21
    
    for i in range(41):
        pos_x += 1
        pos_y = robot_y - 21
        for j in range(41):               
            pos_y += 1
            if pos_x<0 or pos_y<0 or pos_x >= occupancy_grid.shape[1] or pos_y >= occupancy_grid.shape[0]:
                continue
            if occupancy_grid[pos_y][pos_x] == 0.5:
                occupancy_grid[pos_y][pos_x] = 1
    return occupancy_grid
    '''
def get_robot_pos_on_grid(robot_position_wc, min_x, min_y, RESOLUTION=0.05):
    if robot_position_wc is None:
        return None
    
    robot_x = ((robot_position_wc[0] - min_x) / RESOLUTION).astype(int)
    robot_y = ((robot_position_wc[1] - min_y) / RESOLUTION).astype(int)    
    
    return (robot_y, robot_x)

def get_turn_angle(angle, yaw):
    yaw *= -1
    angle = angle - yaw
    if -math.pi < angle <= math.pi:
        return -angle
    elif angle > math.pi:
        return math.pi*2 - angle
    elif angle <= -math.pi:
        return -angle-(math.pi*2)

     
def update_map(occupancy_grid, occupancy_grid_prev, robot_pos_prev, prev_robot_pos_current_map):
    if occupancy_grid_prev is None:
        return occupancy_grid
    # Calculate the robot position from the previous occupancy grid to the current occupancy map
    i_current = prev_robot_pos_current_map[0] - robot_pos_prev[0]
    j_current = prev_robot_pos_current_map[1] - robot_pos_prev[1]

    i_current_end = min(occupancy_grid.shape[0], occupancy_grid_prev.shape[0] + i_current)
    j_current_end = min(occupancy_grid.shape[1], occupancy_grid_prev.shape[1] + j_current)

    # store the already occupied and free space indices from the current observation
    free_space = occupancy_grid == 1
    occupied = occupancy_grid == 0

    # Using the previous robot position index in the current map, overlap the previous map to the current.
    occupancy_grid[i_current:i_current_end, j_current:j_current_end] = \
        occupancy_grid_prev[:i_current_end-i_current, :j_current_end-j_current]
    
    # Reapply the current observation values
    occupancy_grid[free_space] = 1
    occupancy_grid[occupied] = 0

    return occupancy_grid

    unknown_current_map = occupancy_grid == 0.5
    mask_map = np.zeros(occupancy_grid.shape)
    mask_map[i_current:i_current_end, j_current:j_current_end] = 1
    relevent_map = np.logical_and(unknown_current_map, mask_map)
    index = np.where(relevent_map==1)
    occupancy_grid[index] = occupancy_grid_prev[index[0]-i_current, index[1]-j_current]

    index = np.where(occupancy_grid_prev==0)
    #a = np.logical_and(index[0]+i_current < occupancy_grid.shape[0], index[1]+j_current < occupancy_grid.shape[1])
    #occupancy_grid[index[0][a]+i_current, index[1][a]+j_current] = occupancy_grid[index[0][a], index[1][a]]    
    
    for i in range(index[0].size):
        try:
            occupancy_grid[index[0][i]+i_current, index[1][i]+j_current] = occupancy_grid_prev[index[0][i], index[1][i]]
        except:
            pass
    
    return occupancy_grid
    i_prev = 0

    while i_current < i_current_end and i_current < occupancy_grid.shape[0]:
        j_curr = j_current
        j_prev = 0
        while j_curr < j_current_end and j_curr < occupancy_grid.shape[1]:
            # If current cell is unknown or prev cell is occupied
            # replace it with whatever prev map has in it
            if occupancy_grid[i_current][j_curr] == 0.5 or occupancy_grid_prev[i_prev][j_prev] == 0:
                occupancy_grid[i_current][j_curr] = occupancy_grid_prev[i_prev][j_prev]
                #print(f'i curr j curr {i_current, j_curr} ijprev {i_prev, j_prev}')
            
            j_curr += 1
            j_prev += 1
        
        i_current += 1
        i_prev += 1
    
    return occupancy_grid
        
    
    

def create_occupancy_grid(current_pcds,min_x,max_x,min_y,max_y,min_z,max_z, \
                          robot_position, goal_pos, RESOLUTION = 0.05, visualize=True, \
                            index=None, output_dir=None, mode="world_coordinate"):
    x_range = int((max_x - min_x) / RESOLUTION)
    y_range = int((max_y - min_y) / RESOLUTION)
    z_range = int((max_z - min_z) / RESOLUTION)
    
    if mode == "world_coordinate":
        THRESHOLD_LOW = 0.1
    elif mode == "robot_coordinate":
        THRESHOLD_LOW = 0.2
    THRESHOLD_HIGH = 1
    #THRESHOLD_LOW = min_z + 0.1
    #THRESHOLD_HIGH = max_y - (max_y-min_y)*0.35
    #THRESHOLD_HIGH = min_y + (max_y-min_y)*0.65
    #THRESHOLD_HIGH = min_z + 0.5
    
    occupancy_grid = np.ones((y_range+1, x_range+1)) * 0.5 #unexlpored
    for i,pcd in enumerate(current_pcds):
        #pc_points = (np.asarray(pcd.points) * scale) + current_point
        #o3d.visualization.draw_geometries([pcd])
        pc_points = pcd

        #l1.append(pcd)

        x = ((pc_points[:,0] - min_x) / RESOLUTION).astype(int)
        y = ((pc_points[:,1] - min_y) / RESOLUTION).astype(int)
        obj = np.logical_and(THRESHOLD_LOW < pc_points[:,2], pc_points[:,2] < THRESHOLD_HIGH)
        free_space = pc_points[:,2] <= THRESHOLD_LOW

        occupancy_grid[y[free_space], x[free_space]] = 1 #freespace = 0
        occupancy_grid[y[obj],x[obj]] = 0 #occupied = 1

        goal_x, goal_y = ((goal_pos[0] - min_x) / RESOLUTION).astype(int), ((goal_pos[1] - min_y) / RESOLUTION).astype(int)
        if mode == "robot_coordinate":
            robot_x, robot_y = ((0 - min_x) / RESOLUTION).astype(int), ((0 - min_y) / RESOLUTION).astype(int)
        if mode == "world_coordinate":
            robot_x, robot_y = ((robot_position[0] - min_x) / RESOLUTION).astype(int), ((robot_position[1] - min_y) / RESOLUTION).astype(int)
            if index == 0:
                occupancy_grid[goal_y][goal_x] = 0.5
        for i in range(-3,4):
            for j in range(-3,4):
                try:
                    occupancy_grid[robot_y+i][robot_x+j] = 0.5
                except:
                    pass
        #occupancy_grid = fill_invisible_cells_close_to_the_robot(occupancy_grid, robot_x, robot_y)
        

        if visualize == True:            
            visualize_occupancy_grid(occupancy_grid, (robot_x, robot_y), (goal_x, goal_y), store=True, show=False, output_path=f'{output_dir}/og/{index}')
        
        #np.save(f'pc/pc_obj_{index}', pc_points[obj])
        #np.save(f'pc/pc_free_{index}', pc_points[free_space])
        #o3d.visualization.draw_geometries([get_pcd_from_numpy(pc_points[free_space])])
        #l.extend(pc_points)
        #l.extend(pc_points[free_space])
        #l2.extend(pc_points[obj])
        
        '''
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(current_pcds[0])
        o3d.visualization.draw_geometries([pcd1])

        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(pc_points[free_space])
        o3d.visualization.draw_geometries([pcd3])

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc_points[obj])
        o3d.visualization.draw_geometries([pcd2])
        '''
    #o3d.visualization.draw_geometries(current_pcds)    
    #o3d.visualization.draw_geometries([get_pcd_from_numpy(np.array(l))])
    #o3d.visualization.draw_geometries([get_pcd_from_numpy(np.array(l2))])
    
    return occupancy_grid, (robot_y, robot_x), (goal_y, goal_x)

def get_closest_free_space_to_robot(occupancy_grid, robot_pos):
    filtered_grid = scipy.ndimage.maximum_filter((occupancy_grid == 1), size=3)
    labels, nb = scipy.ndimage.label(filtered_grid)

    frontiers = []
    for ii in range(nb):
        raw_frontier_indices = np.where(labels == (ii + 1))
        frontiers.append(Frontier(raw_frontier_indices[0], raw_frontier_indices[1]))
        
        free_space = np.vstack((raw_frontier_indices[0], raw_frontier_indices[1])).T
        free_space_nodes = tuple(map(tuple, free_space))

        if robot_pos in free_space_nodes:
            #show_frontiers(occupancy_grid, frontiers)
            return free_space

def dist(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b)

def visualize_path(occupancy_grid, path, store=False, show=False, output_path=None):
    plt.figure()
    for i in path:
        plt.plot(i[1], i[0], marker="o", markersize=5, alpha=0.8)
    plt.imshow(occupancy_grid, cmap='gray'); 
    if show == True:
        plt.show()
    if store == True:
        plt.savefig(output_path)
    plt.close()

def inflate_grid(grid,
                 inflation_radius,
                 obstacle_threshold=0,
                 collision_val=0):
    """Inflates obstacles in an occupancy grid
    Creates a mask for all grid cells exceeding the obstacle threshold and
    uses a convolution to compute how much the obstacles should inflate. The
    inflated mask is used to set the cells of a copy of the initial grid to
    occupied. All other cells are unchanged: free space and unnoccupied cells
    outside of the inflation radius are preserved (therefore, frontiers can
    still be computed from an inflated grid).
    Args:
        grid (np.Array): occupancy grid
        inflation_radius (float): radius (in grid units) to inflate obstacles.
            Note that this is a float; a fractional inflation radius can be
            used to determine whether or not corners of a box are included.
        obstacle_threshold (float): value above which a cell is an obstacle.
        collision_val (float): value obstacles are given after inflation.
    Returns:
        inflated_grid (np.Array): grid with inflated obstacles.
    """

    obstacle_grid = np.zeros(grid.shape)
    obstacle_grid[grid == obstacle_threshold] = 1

    kernel_size = int(1 + 2 * math.ceil(inflation_radius))
    cind = int(math.ceil(inflation_radius))
    y, x = np.ogrid[-cind:kernel_size - cind, -cind:kernel_size - cind]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[y * y + x * x <= inflation_radius * inflation_radius] = 1
    inflated_mask = scipy.ndimage.filters.convolve(obstacle_grid,
                                                   kernel,
                                                   mode='constant',
                                                   cval=0)
    inflated_mask = inflated_mask >= 1.0
    inflated_grid = grid.copy()
    inflated_grid[inflated_mask] = collision_val

    return inflated_grid

def get_cost(path, cost):
    total_cost = 0
    for i in range(len(path)-1):
        try:
            total_cost += cost[(path[i], path[i+1])]
        except:
            total_cost += cost[(path[i+1], path[i])]
    return total_cost

def a_star_search(occupancy_grid, frontiers, robot_pos, goal_pos, index, goal_found=False, output_dir=None):
    graph = nx.grid_graph((occupancy_grid.shape[1], occupancy_grid.shape[0]))
    nx.set_edge_attributes(graph, np.inf, "cost")
    graph_unknown = nx.grid_graph((occupancy_grid.shape[1], occupancy_grid.shape[0]))
    nx.set_edge_attributes(graph_unknown, np.inf, "cost_unknown")
    
    free_space = np.vstack((np.where(occupancy_grid==1)[0], np.where(occupancy_grid==1)[1])).T
    unknwon_space = np.vstack((np.where(occupancy_grid==0.5)[0], np.where(occupancy_grid==0.5)[1])).T
    for f in frontiers:
        free_space = np.vstack((free_space, f.frontier.T))
        free_space = np.vstack((free_space, np.array([int(f.centroid[0]), int(f.centroid[1])])))
        
        unknwon_space = np.vstack((unknwon_space, f.frontier.T))
        unknwon_space = np.vstack((unknwon_space, np.array([int(f.centroid[0]), int(f.centroid[1])])))
    free_space_nodes = tuple(map(tuple, free_space))
    unknwon_space_nodes = tuple(map(tuple, unknwon_space))

    free_space_nodes = {node:None for node in free_space_nodes}
    unknwon_space_nodes = {node: None for node in unknwon_space_nodes}

    cost = {}
    cost_unknown = {}
    for edge in graph.edges():
        if edge[0] in free_space_nodes and edge[1] in free_space_nodes:
            cost[edge] = np.linalg.norm(np.array(edge[0])-np.array(edge[1]))
        else:
            cost[edge] = np.inf
        if edge[0] in unknwon_space_nodes and edge[1] in unknwon_space_nodes:
            cost_unknown[edge] = np.linalg.norm(np.array(edge[0])-np.array(edge[1]))
        else:
            cost_unknown[edge] = np.inf
    nx.set_edge_attributes(graph, cost, "cost")
    nx.set_edge_attributes(graph_unknown, cost_unknown, "cost_unknown")
        
    if goal_found == True:
        path = nx.astar_path(graph, robot_pos, goal_pos, heuristic=dist, weight="cost")
        cost_path = get_cost(path, cost)
        #show_frontiers(occupancy_grid, [goal_pos], show=False, store=True, output_path=f'output1/closest_frontier/{index}')
        #visualize_path(occupancy_grid, path, show=False, store=True, output_path=f'output1/path/{index}')

        return path, cost_path
    
    min_cost = np.inf
    for f in frontiers:
        try:
            path = nx.astar_path(graph, robot_pos, (int(f.centroid[0]), int(f.centroid[1])), heuristic=dist, weight="cost")
            path_unknwon = nx.astar_path(graph_unknown, (int(f.centroid[0]), int(f.centroid[1])), goal_pos, heuristic=dist, weight="cost_unknown")
        except:
            print(f"target {(int(f.centroid[0]), int(f.centroid[1]))} not found in graph")
        cost_path = get_cost(path, cost)
        print("cost_path", cost_path)
        if cost_path > min_cost:
            continue
        
        cost_path_unknown = get_cost(path_unknwon, cost_unknown)
        print("cost_path_unknown", cost_path_unknown)
        #visualize_path(occupancy_grid, path_to_frontier, show=True, store=False, output_path=f'output/path/{index}')
        #visualize_path(occupancy_grid, unknwon_path, show=True, store=False, output_path=f'output/path_unknown/{index}')
        if  cost_path + cost_path_unknown < min_cost:
            min_cost = cost_path + cost_path_unknown
            closest_frontier_to_goal = f
            path_to_frontier = path
            unknwon_path = path_unknwon
        

    print("min cost", min_cost)
    if min_cost != np.inf:    
        show_frontiers(occupancy_grid, [closest_frontier_to_goal], show=False, store=True, output_path=f'{output_dir}/closest_frontier/{index}')
        visualize_path(occupancy_grid, path_to_frontier, show=False, store=True, output_path=f'{output_dir}/path/{index}')
        visualize_path(occupancy_grid, unknwon_path, show=False, store=True, output_path=f'{output_dir}/path_unknown/{index}')
        return path_to_frontier, min_cost
    else:
        return [robot_pos, robot_pos], min_cost

'''
def a_star_search(occupancy_grid, frontiers, robot_pos, goal_pos, index, goal_found=False):
    graph = nx.grid_graph((occupancy_grid.shape[1], occupancy_grid.shape[0]))
    graph_unknown = nx.grid_graph((occupancy_grid.shape[1], occupancy_grid.shape[0]))
    
    free_space = np.vstack((np.where(occupancy_grid==1)[0], np.where(occupancy_grid==1)[1])).T
    for f in frontiers:
        free_space = np.vstack((free_space, f.frontier.T))
        free_space = np.vstack((free_space, np.array([int(f.centroid[0]), int(f.centroid[1])])))
    free_space_nodes = tuple(map(tuple, free_space))

    cost = {}
    for edge in graph.edges():
        if edge[0] in free_space_nodes and edge[1] in free_space_nodes:
            cost[edge] = np.linalg.norm(np.array(edge[0])-np.array(edge[1]))
        else:
            cost[edge] = np.inf
    nx.set_edge_attributes(graph, cost, "cost")

    if goal_found == True:
        path = nx.astar_path(graph, robot_pos, goal_pos, heuristic=dist, weight="cost")
        show_frontiers(occupancy_grid, goal_pos, show=False, store=True, output_path=f'output1/closest_frontier/{index}')
        visualize_path(occupancy_grid, path, show=False, store=True, output_path=f'output1/path/{index}')

        return path

    unknwon_space = np.vstack((np.where(occupancy_grid==0.5)[0], np.where(occupancy_grid==0.5)[1])).T
    for f in frontiers:
        unknwon_space = np.vstack((unknwon_space, f.frontier.T))
        unknwon_space = np.vstack((unknwon_space, np.array([int(f.centroid[0]), int(f.centroid[1])])))
    unknwon_space_nodes = tuple(map(tuple, unknwon_space))

    cost_unknown = {}
    for edge in graph_unknown.edges():
        if edge[0] in unknwon_space_nodes and edge[1] in unknwon_space_nodes:
            cost_unknown[edge] = np.linalg.norm(np.array(edge[0])-np.array(edge[1]))
        else:
            cost_unknown[edge] = np.inf
    nx.set_edge_attributes(graph_unknown, cost_unknown, "cost_unknown")

    
    #print(graph.edges())
    
    d = np.inf
    for f in frontiers:
        #frontier_to_goal_dist = np.linalg.norm(f.centroid - goal_pos)
        #if f.frontier.shape[1] <= 5:
        #    continue
        #show_frontiers(occupancy_grid, [f])
        #print(int(f.centroid[0]), int(f.centroid[1]))
        try:
            path = nx.astar_path(graph, robot_pos, (int(f.centroid[0]), int(f.centroid[1])), heuristic=dist, weight="cost")
            path_unknwon = nx.astar_path(graph_unknown, (int(f.centroid[0]), int(f.centroid[1])), goal_pos, heuristic=dist, weight="cost_unknown")
        except:
            print(f"target {(int(f.centroid[0]), int(f.centroid[1]))} not found in graph")
        cost_path = get_cost(path, cost)
        print("cost_path", cost_path)
        if cost_path > d:
            continue
        
        cost_path_unknown = get_cost(path_unknwon, cost_unknown)
        print("cost_path_unknown", cost_path_unknown)
        #visualize_path(occupancy_grid, path_to_frontier, show=True, store=False, output_path=f'output/path/{index}')
        #visualize_path(occupancy_grid, unknwon_path, show=True, store=False, output_path=f'output/path_unknown/{index}')
        if  cost_path + cost_path_unknown < d:
            d = cost_path + cost_path_unknown
            closest_frontier_to_goal = f
            path_to_frontier = path
            unknwon_path = path_unknwon
        

    print("min cost", d)    
    show_frontiers(occupancy_grid, [closest_frontier_to_goal], show=False, store=True, output_path=f'output1/closest_frontier/{index}')

    visualize_path(occupancy_grid, path_to_frontier, show=False, store=True, output_path=f'output1/path/{index}')
    visualize_path(occupancy_grid, unknwon_path, show=False, store=True, output_path=f'output1/path_unknown/{index}')

    return path_to_frontier
'''