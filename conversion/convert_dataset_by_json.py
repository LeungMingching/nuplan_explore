import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
from timestamp_queries import *
from utils import *

config = {
    'db_root': '/home/gac/nuplan_explore/nuplan/dataset/nuplan-v1.1/mini',
    'json_root': '/home/gac/nuplan_explore/nuplan-devkit/tutorials/DATA',
    'coverted_root': './converted_dataset_by_json',
    'past_time_horizon': 0, # past agent traj not supported yet
    'future_time_horizon': 1, # future ego traj not supported yet
    'roi_radius': 100,
    'num_objects': 10
}

def discover_labels(index_dir):
    sub_folders = [name for name in os.listdir(index_dir) if os.path.isdir(os.path.join(index_dir, name))]
    return sub_folders

def resize_num_agents(agents_state_array, num_objcts, num_features):
    resized_agents_state_array = agents_state_array.copy()
    resized_agents_state_array.resize((num_objcts,num_features))
    return resized_agents_state_array

def construct_ego_state_by_timestamp(
    log_db_file: str,
    timestamp: str
) -> np.ndarray:
    """
    [x, y, heading, v_x, v_y, a_x, a_y, is_valid(1)]
    """
    ego_state = get_ego_state_by_timestamp_from_db(log_db_file,timestamp)

    ego_state_array = np.array([
        ego_state['x'],
        ego_state['y'],
        ego_state['heading'],
        ego_state['vx'],
        ego_state['vy'],
        ego_state['ax'],
        ego_state['ay'],
        1
    ])

    return ego_state_array

def construct_agents_state_by_timestamp(
    log_db_file: str,
    timestamp: str,
    prev_timestamp: str,
    future_timestamp: str,
    ego_current_state: np.ndarray,
    roi_radius: float,
    num_interested_obj: int
) -> np.ndarray:
    """
    [
        [x, y, heading, v_x, v_y, a_x, a_y, is_valid(1 or 0)],
        [x, y, heading, v_x, v_y, a_x, a_y, is_valid(1 or 0)],
        ...
    ]
    """
    agents_dict = get_agents_by_timestamp_from_db(log_db_file, timestamp)
    ego_x = ego_current_state[0]
    ego_y = ego_current_state[1]

    agents_state_array = []
    agents_distance = []
    agents_track_token = []
    for token in agents_dict:
        agent_x = agents_dict[token]['x']
        agent_y = agents_dict[token]['y']
        dx = agent_x - ego_x
        dy = agent_y - ego_y
        distance = np.linalg.norm(np.asarray([dx, dy]))

        if distance <= roi_radius:
            agents_state_array.append([
                agents_dict[token]['x'],
                agents_dict[token]['y'],
                agents_dict[token]['heading'],
                agents_dict[token]['vx'],
                agents_dict[token]['vy'],
                0,
                0,
                1
            ])
            agents_distance.append(distance)
            agents_track_token.append(token)
    agents_state_array = np.asarray(agents_state_array)
    agents_distance = np.asarray(agents_distance)
    agents_track_token = np.asarray(agents_track_token)

    # sort by distance
    agents_state_array = agents_state_array[agents_distance.argsort()]
    agents_track_token = agents_track_token[agents_distance.argsort()]
    agents_distance = agents_distance[agents_distance.argsort()]

    # assign acceleration
    t_interval = float(future_timestamp) - float(prev_timestamp)
    prev_agents_dict = get_agents_by_timestamp_from_db(log_db_file, prev_timestamp)
    future_agents_dict = get_agents_by_timestamp_from_db(log_db_file, future_timestamp)

    for i in range(len(agents_track_token)):
        track_token = agents_track_token[i]
        try:
            prev_agent_vx = prev_agents_dict[track_token]['vx']
            prev_agent_vy = prev_agents_dict[track_token]['vy']
            future_agent_vx = future_agents_dict[track_token]['vx']
            future_agent_vy = future_agents_dict[track_token]['vy']

            acceleration_x = (future_agent_vx - prev_agent_vx) / t_interval
            acceleration_y = (future_agent_vy - prev_agent_vy) / t_interval

            agents_state_array[i][5] = acceleration_x
            agents_state_array[i][6] = acceleration_y
            
        except:
            agents_state_array[i][5] = 0
            agents_state_array[i][6] = 0
            continue

    # resize
    agents_state_array = resize_num_agents(agents_state_array, num_interested_obj, 8)

    return agents_state_array

def main(config):
    # read config
    db_root = config['db_root']
    json_root = config['json_root']
    coverted_root = config['coverted_root']
    past_time_horizon = config['past_time_horizon']
    future_time_horizon = config['future_time_horizon']
    roi_radius = config['roi_radius']
    num_objects = config['num_objects']
    
    # loop for all labels
    label_list = discover_labels(json_root)
    for label in label_list:
        label_path = os.path.join(coverted_root, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        
        # loop for all json files
        index_file_list = glob(os.path.join(json_root, label, '*'))
        for idx_file in index_file_list:
            with open(idx_file, 'rb') as f:
                index_dict = json.load(f)
            log_db_file = os.path.join(db_root, index_dict['log_db_file'])
            start_timestamp = index_dict['clip_head']['timestamp']
            end_timestamp = index_dict['clip_tail']['timestamp']

            # TODO: build_observation_between_timestamps()
            db_timestamp_array = np.array(get_all_timestamps_from_db(log_db_file))
            db_timestamp_array.sort()

            # observation
            idx_obs_start = find_nearest_idx(db_timestamp_array, start_timestamp + past_time_horizon*1e6) + 1 # +1 to calculate acceleration
            idx_obs_end = find_nearest_idx(db_timestamp_array, end_timestamp - future_time_horizon*1e6)
            observation_list = []
            # loop all frames within a clip
            for idx in range(idx_obs_start, idx_obs_end):
                timestamp = db_timestamp_array[idx]
                prev_timestamp = db_timestamp_array[idx - 1]
                future_timestamp = db_timestamp_array[idx + 1]

                ego_state_array = construct_ego_state_by_timestamp(log_db_file, timestamp)
                agents_state_array = construct_agents_state_by_timestamp(log_db_file, timestamp, prev_timestamp, future_timestamp, ego_state_array, roi_radius, num_objects)
                ego_state_array = np.expand_dims(ego_state_array, axis=0)
                observation = np.concatenate((ego_state_array, agents_state_array), axis=0)
                observation_list.append(observation)
            observation_list = np.asarray(observation_list)
            
            # TODO: build_observation_between_timestamps()
            # look ahead point
            look_ahead_pt_list = []

            
            print(f'log_db_file: \n{log_db_file}')
            print(f'start_timestamp: \n{start_timestamp}')
            print(f'end_timestamp: \n{end_timestamp}')
            print(f'db_timestamp_array: \n{db_timestamp_array}')
            print(f'observation_list len: \n{len(observation_list)}')

    pass

if __name__ == '__main__':
    main(config)