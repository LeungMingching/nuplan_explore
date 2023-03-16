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

def align_length(array1, array2):
    if len(array1) > len(array2):
        array1 = array1[:len(array2)]
    elif len(array1) < len(array2):
        array2 = array2[:len(array1)]
    return array1, array2

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

def get_observation_timestamp_range(
    log_db_file,
    clip_start_timestamp,
    clip_end_timestamp,
    past_time_horizon,
    future_time_horizon
):
    db_timestamp_array = np.array(get_all_timestamps_from_db(log_db_file))
    db_timestamp_array.sort()
    idx_obs_start = find_nearest_idx(
        db_timestamp_array, clip_start_timestamp + past_time_horizon*1e6)
    idx_obs_end = find_nearest_idx(
        db_timestamp_array, clip_end_timestamp - future_time_horizon*1e6)
    return db_timestamp_array[idx_obs_start:idx_obs_end]

def get_look_ahead_pt_timestamp_range(
    log_db_file,
    clip_start_timestamp,
    clip_end_timestamp,
    past_time_horizon,
    future_time_horizon
):
    db_timestamp_array = np.array(get_all_timestamps_from_db(log_db_file))
    db_timestamp_array.sort()
    idx_lap_start = find_nearest_idx(
        db_timestamp_array, clip_start_timestamp + past_time_horizon*1e6 + future_time_horizon*1e6)
    idx_lap_end = find_nearest_idx(
        db_timestamp_array, clip_end_timestamp)
    return db_timestamp_array[idx_lap_start:idx_lap_end]

def build_observation_between_timestamps(
    log_db_file,
    timestamp_array,
    roi_radius,
    num_objects
):
    observation_list = []
    # loop all frames within a clip
    for idx in range(1, len(timestamp_array)-1):
        timestamp = timestamp_array[idx]
        prev_timestamp = timestamp_array[idx - 1]
        future_timestamp = timestamp_array[idx + 1]

        ego_state_array = construct_ego_state_by_timestamp(log_db_file, timestamp)
        agents_state_array = construct_agents_state_by_timestamp(log_db_file, timestamp, prev_timestamp, future_timestamp, ego_state_array, roi_radius, num_objects)
        ego_state_array = np.expand_dims(ego_state_array, axis=0)
        observation = np.concatenate((ego_state_array, agents_state_array), axis=0)
        observation_list.append(observation)
    return np.asarray(observation_list)

def build_look_ahead_pt_between_timestamps(
    log_db_file,
    timestamp_array
):
    look_ahead_pt_list = []
    for idx in range(1, len(timestamp_array)-1):
        timestamp = timestamp_array[idx]
        ego_state_array = construct_ego_state_by_timestamp(log_db_file, timestamp)
        look_ahead_pt_x = ego_state_array[0]
        look_ahead_pt_y = ego_state_array[1]
        look_ahead_pt_v = np.linalg.norm(
            np.asarray([ego_state_array[3], ego_state_array[4]]))
        look_ahead_pt = np.array([look_ahead_pt_x, look_ahead_pt_y, look_ahead_pt_v])
        look_ahead_pt_list.append(look_ahead_pt)
    return np.asarray(look_ahead_pt_list)

def extract_data_by_json(
    log_db_file,
    clip_start_timestamp,
    clip_end_timestamp,
    past_time_horizon,
    future_time_horizon,
    roi_radius,
    num_objects
):
    # observation
    observation_timestamp_range = get_observation_timestamp_range(
        log_db_file, clip_start_timestamp, clip_end_timestamp, past_time_horizon, future_time_horizon)
    observation_array = build_observation_between_timestamps(
        log_db_file, observation_timestamp_range, roi_radius, num_objects)
    
    # loop-ahead point
    look_ahead_pt_timestamp_range = get_look_ahead_pt_timestamp_range(
        log_db_file, clip_start_timestamp, clip_end_timestamp, past_time_horizon, future_time_horizon)
    look_ahead_pt_array = build_look_ahead_pt_between_timestamps(
        log_db_file, look_ahead_pt_timestamp_range)
    
    # align
    observation_array, look_ahead_pt_array = align_length(observation_array, look_ahead_pt_array)
    assert len(observation_array) == len(look_ahead_pt_array), 'Not equal length'
    
    return observation_array, look_ahead_pt_array

def main(config):
    # read config
    db_root = config['db_root']
    json_root = config['json_root']
    coverted_root = config['coverted_root']
    past_time_horizon = config['past_time_horizon']
    future_time_horizon = config['future_time_horizon']
    roi_radius = config['roi_radius']
    num_objects = config['num_objects']
    
    toc = {}
    toc['total_frame'] = 0
    # loop for all labels
    label_list = discover_labels(json_root)
    for label in label_list:
        label_path = os.path.join(coverted_root, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        
        # loop for all json files
        index_file_list = glob(os.path.join(json_root, label, '*'))
        toc[label] = {}
        toc[label]['clip'] = len(index_file_list)
        toc[label]['frame'] = 0
        for idx_file in tqdm(index_file_list, desc=f'Working on {label}'):
            with open(idx_file, 'rb') as f:
                index_dict = json.load(f)
            log_db_file = os.path.join(db_root, index_dict['log_db_file'])
            clip_start_timestamp = index_dict['clip_head']['timestamp']
            clip_end_timestamp = index_dict['clip_tail']['timestamp']

            # skip if converted
            observation_file = os.path.join(label_path, f'{clip_start_timestamp}_observation.npy')
            look_ahead_pt_file = os.path.join(label_path, f'{clip_start_timestamp}_look_ahead_pt.npy')
            if os.path.exists(look_ahead_pt_file):
                continue
            
            # extract
            observation_array, look_ahead_pt_array = extract_data_by_json(
                log_db_file, clip_start_timestamp, clip_end_timestamp,
                past_time_horizon, future_time_horizon, roi_radius, num_objects)

            toc[label]['frame'] += len(look_ahead_pt_array)
            toc['total_frame'] += len(look_ahead_pt_array)
            # save
            with open(observation_file, 'wb') as f:
                np.save(f, observation_array)
            with open(look_ahead_pt_file, 'wb') as f:
                np.save(f, look_ahead_pt_array)

    toc_path = os.path.join(coverted_root, 'table_of_content.json')
    if not os.path.exists(toc_path):
        json_f = json.dumps(toc, indent=4, sort_keys=True)
        with open(toc_path, 'w') as f:        
            f.write(json_f)
    print('Done!')
    pass

if __name__ == '__main__':
    main(config)