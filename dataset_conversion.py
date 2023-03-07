import os
from tqdm import tqdm

import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs

from tutorials.utils.tutorial_utils import get_scenario_type_token_map, get_default_scenario_from_token

DATA_ROOT = '~/nuplan_explore/nuplan'
MAP_ROOT = '~/nuplan_explore/nuplan/dataset/maps'
DB_FILES = '~/nuplan_explore/nuplan/dataset/nuplan-v1.1/mini'
MAP_VERSION = '~/nuplan_explore/nuplan-maps-v1.0'

CONVERTED_ROOT = './converted_dataset/'

def read_ego_past_states(
    scenario: AbstractScenario,
    time_horizon: float
) -> np.ndarray:
    
    ego_past_traj = list(scenario.get_ego_past_trajectory(
        iteration=1,
        time_horizon=time_horizon
    ))

    ego_past_array = np.array(
        [[past_state.agent.center.x,
          past_state.agent.center.y,
          past_state.agent.center.heading,
          past_state.dynamic_car_state.center_velocity_2d.x,
          past_state.dynamic_car_state.center_velocity_2d.y,
          past_state.dynamic_car_state.center_acceleration_2d.x,
          past_state.dynamic_car_state.center_acceleration_2d.y] for past_state in ego_past_traj]
    )

    ego_past_time_array = np.array(
        [past_state.time_seconds for past_state in ego_past_traj]
    )

    return ego_past_array, ego_past_time_array

def read_objects_past_states(
    scenario: AbstractScenario,
    time_horizon: float
) -> np.ndarray:
    
    tracked_objs_within_t = scenario.get_tracked_objects_within_time_window_at_iteration(
        iteration=1,
        past_time_horizon=time_horizon,
        future_time_horizon=0,
    )
    tracked_objs_list = tracked_objs_within_t.tracked_objects.get_agents()
    # TODO: to be done

def read_ego_future_states(
    scenario: AbstractScenario,
    time_horizon: float
) -> np.ndarray:
    
    ego_future_traj = list(scenario.get_ego_future_trajectory(
        iteration=1,
        time_horizon=time_horizon
    ))

    ego_future_array = np.array(
        [[future_state.agent.center.x,
          future_state.agent.center.y,
          future_state.agent.center.heading,
          future_state.dynamic_car_state.center_velocity_2d.x,
          future_state.dynamic_car_state.center_velocity_2d.y,
          future_state.dynamic_car_state.center_acceleration_2d.x,
          future_state.dynamic_car_state.center_acceleration_2d.y] for future_state in ego_future_traj]
    )

    ego_future_time_array = np.array(
        [future_state.time_seconds for future_state in ego_future_traj]
    )

    return ego_future_array, ego_future_time_array

def get_look_ahead_point_by_time(
    scenario: AbstractScenario,
    time_horizon: float
) -> np.ndarray:

    ego_future_array, _ = read_ego_future_states(scenario, time_horizon)

    look_ahead_pt_x = ego_future_array[-1][0]
    look_ahead_pt_y = ego_future_array[-1][1]
    look_ahead_pt_v = np.linalg.norm(np.asarray([ego_future_array[-1][3], ego_future_array[-1][4]]))

    look_ahead_pt = np.array(
        [look_ahead_pt_x, look_ahead_pt_y, look_ahead_pt_v]
    )

    return look_ahead_pt

def construct_ego_current_state(
    scenario: AbstractScenario
) -> np.ndarray:
    """
    [x, y, heading, v_x, v_y, a_x, a_y, is_valid(1)]
    """
    
    ego_current_state = scenario.get_ego_state_at_iteration(iteration=1)

    ego_current_array = np.array(
        [ego_current_state.agent.center.x,
         ego_current_state.agent.center.y,
         ego_current_state.agent.center.heading,
         ego_current_state.dynamic_car_state.center_velocity_2d.x,
         ego_current_state.dynamic_car_state.center_velocity_2d.y,
         ego_current_state.dynamic_car_state.center_acceleration_2d.x,
         ego_current_state.dynamic_car_state.center_acceleration_2d.y,
         1]
    )

    return ego_current_array

def construct_objects_current_state(
    scenario: AbstractScenario,
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
    
    objects_current_state = scenario.get_tracked_objects_at_iteration(iteration=1).tracked_objects.get_agents()
    ego_x = ego_current_state[0]
    ego_y = ego_current_state[1]

    # filter
    objects_current_info = []
    objects_distance = []
    objects_track_token = []
    for obj_current_state in objects_current_state:
        obj_x = obj_current_state.center.x
        obj_y = obj_current_state.center.y
        dx = obj_x - ego_x
        dy = obj_y - ego_y
        distance = np.linalg.norm(np.asarray([dx, dy]))

        if distance <= roi_radius:
            objects_current_info.append(
                [obj_current_state.center.x,
                 obj_current_state.center.y,
                 obj_current_state.center.heading,
                 obj_current_state.velocity.x,
                 obj_current_state.velocity.y,
                 0,
                 0,
                 1]
            )
            objects_distance.append(distance)
            objects_track_token.append(obj_current_state.track_token)
    objects_current_info = np.asarray(objects_current_info)
    objects_distance = np.asarray(objects_distance)
    objects_track_token = np.asarray(objects_track_token)

    # sort by distance
    objects_current_info = objects_current_info[objects_distance.argsort()]
    objects_track_token = objects_track_token[objects_distance.argsort()]
    objects_distance = objects_distance[objects_distance.argsort()]

    # assign acceleration
    t_interval = scenario.database_interval

    objects_prev_state = scenario.get_tracked_objects_at_iteration(iteration=0).tracked_objects.get_agents()
    objects_future_state = scenario.get_tracked_objects_at_iteration(iteration=2).tracked_objects.get_agents()

    prev_token_array = np.array([obj.track_token for obj in objects_prev_state])
    future_token_array = np.array([obj.track_token for obj in objects_future_state])

    for i in range(len(objects_track_token)):
        track_token = objects_track_token[i]

        idx_prev_obj = np.where(prev_token_array==track_token)[0].item() \
            if len(np.where(prev_token_array==track_token)[0]) != 0 else None
        idx_future_obj = np.where(future_token_array==track_token)[0].item() \
            if len(np.where(future_token_array==track_token)[0]) != 0 else None

        if idx_prev_obj == None or idx_future_obj == None:
            objects_current_info[i][5] = 0
            objects_current_info[i][6] = 0
            continue
        
        prev_velocity_x = objects_prev_state[idx_prev_obj].velocity.x
        prev_velocity_y = objects_prev_state[idx_prev_obj].velocity.y
        future_velocity_x = objects_future_state[idx_future_obj].velocity.x
        future_velocity_y = objects_future_state[idx_future_obj].velocity.y

        acceleration_x = (future_velocity_x - prev_velocity_x) / 2*t_interval
        acceleration_y = (future_velocity_y - prev_velocity_y) / 2*t_interval

        objects_current_info[i][5] = acceleration_x
        objects_current_info[i][6] = acceleration_y

    # slice or pad
    num_roi_objects = len(objects_current_info)
    num_features = objects_current_info.shape[1] if len(objects_current_info.shape) > 1 else 8
    
    num_pad_objects = num_interested_obj - num_roi_objects
    if num_pad_objects == num_interested_obj:
        objects_current_array = np.zeros((num_pad_objects, num_features))
    elif num_pad_objects <= 0:
        objects_current_array = objects_current_info[0:num_interested_obj, :]
    else:
        objects_virtual_info = np.zeros((num_pad_objects, num_features))
        objects_current_array = np.concatenate((objects_current_info, objects_virtual_info), axis=0)
    
    return objects_current_array

def process(
    data_root: str,
    map_root: str,
    db_files: str,
    map_version: str,
    past_time_horizon: float,
    future_time_horizon: float
) -> None:
    
    # get all scenarios
    print('Getting all scenarios...')
    log_db_files = discover_log_dbs(db_files)
    scenario_type_token_map = get_scenario_type_token_map(log_db_files)
    scenario_type_list = sorted(scenario_type_token_map.keys())

    # loop for all scenario
    for scenario_type in tqdm(scenario_type_list, desc='Scenario progress: '):
        scenario_path = CONVERTED_ROOT + scenario_type
        if not os.path.exists(scenario_path):
            os.makedirs(scenario_path)
        
        if os.path.exists(scenario_path + '/observation_array.npy') \
            and os.path.exists(scenario_path + '/look_ahead_pt_array.npy'):
            continue

        # loop for all dbs
        observation_list = []
        look_ahead_pt_list = []
        for log_db_file, token in tqdm(scenario_type_token_map[scenario_type], desc='Converting ' + scenario_type + ': '):
            # if os.path.exists(scenario_path + '/' + token + '.npy'):
            #     continue
            
            scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)

            ego_state = construct_ego_current_state(scenario)
            objects_state = construct_objects_current_state(scenario, ego_state, 1000, 10)
            ego_state = np.expand_dims(ego_state, axis=0)
            observation = np.concatenate((ego_state, objects_state), axis=0)
            observation_list.append(observation)

            # get look-ahead point
            look_ahead_pt = get_look_ahead_point_by_time(scenario, future_time_horizon)
            look_ahead_pt_list.append(look_ahead_pt)

        observation_array = np.asarray(observation_list)
        look_ahead_pt_array = np.asarray(look_ahead_pt_list)
        
        # save input files
        with open(scenario_path + '/observation_array.npy', 'wb') as f:
            np.save(f, observation_array)

        # save ouput dict
        with open(scenario_path + '/look_ahead_pt_array.npy', 'wb') as f:
                np.save(f, look_ahead_pt_array)

if __name__ == '__main__':

    past_time_horizon = 5
    future_time_horizon = 5

    process(
        DATA_ROOT,
        MAP_ROOT,
        DB_FILES,
        MAP_VERSION,
        past_time_horizon,
        future_time_horizon
    )