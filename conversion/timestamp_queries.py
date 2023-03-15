import os
from query_session import execute_many, execute_one
from utils import quaternion_to_euler

def get_all_timestamps_from_db(log_file: str) -> list:
    query = """
        SELECT  lp.timestamp
        FROM lidar_pc AS lp
        ORDER BY lp.timestamp
    """
    timestamp_ls = [row['timestamp'] for row in execute_many(query, [], log_file)]
    return timestamp_ls

def get_ego_state_by_timestamp_from_db(log_file: str, timestamp: str) -> dict:
    query = """
        SELECT  lp.timestamp,
                ep.x,
                ep.y,
                ep.qw,
                ep.qx,
                ep.qy,
                ep.qz,
                ep.vx,
                ep.vy,
                ep.acceleration_x,
                ep.acceleration_y
        FROM ego_pose AS ep
        INNER JOIN lidar_pc AS lp
            ON lp.ego_pose_token = ep.token
        WHERE lp.timestamp = ?
    """

    row = execute_one(query, [str(timestamp)], log_file)
    if row is None:
        return None
    
    _, _, heading = quaternion_to_euler(row['qx'], row['qy'], row['qz'], row['qw'])

    ego_pose = {
        'x': row['x'],
        'y': row['y'],
        'heading': heading,
        'vx': row['vx'],
        'vy': row['vy'],
        'ax': row['acceleration_x'],
        'ay': row['acceleration_y'],
    }
    return ego_pose

def get_agents_by_timestamp_from_db(log_file: str, timestamp: str) -> dict:
    query = """
        SELECT  lp.timestamp,
                lb.x,
                lb.y,
                lb.z,
                lb.vx,
                lb.vy,
                lb.vz,
                lb.yaw,
                cg.name,
                tr.token
        FROM lidar_pc AS lp
        INNER JOIN lidar_box as lb
            ON lp.token = lb.lidar_pc_token
        INNER JOIN track as tr
            ON lb.track_token = tr.token
        INNER JOIN category as cg
            ON tr.category_token = cg.token
        WHERE (cg.name='pedestrian' OR cg.name='vehicle' OR cg.name='bicycle')
            AND (lp.timestamp = ?)
        ORDER BY lp.timestamp
    """
    agents_dict = {}
    for row in execute_many(query, [str(timestamp)], log_file):
        agents_dict[str(row["token"].hex())] = {
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'vx': row['vx'],
            'vy': row['vy'],
            'vz': row['vz'],
            'heading': row['yaw'],
            'name': row['name']
        }
    return agents_dict

if __name__ == '__main__':
    data_root = '/home/gac/nuplan_explore/nuplan/dataset/nuplan-v1.1/mini'
    
    log_file = os.path.join(data_root, '2021.05.12.22.00.38_veh-35_01008_01518.db')
    timestamp = '1620857889651124'

    print(get_agents_by_timestamp_from_db(log_file, timestamp))
