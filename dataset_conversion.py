import os
import math

import itertools
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_lidarpc_token_map_name_from_db,
    get_lidarpc_token_timestamp_from_db,
    get_lidarpc_tokens_with_scenario_tag_from_db,
)
from nuplan.planning.nuboard.base.data_class import NuBoardFile, SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.base.simulation_tile import SimulationTile
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)

from tutorials.utils.tutorial_utils import get_scenario_type_token_map, get_default_scenario_from_token

def construct_ego_past_states(
    scenario: AbstractScenario,
    time_horizon: float
) -> np.ndarray:
    
    ego_past_traj = list(scenario.get_ego_past_trajectory(
        iteration=0,
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

    return ego_past_array

def construct_objects_past_states(
    scenario: AbstractScenario,
    time_horizon: float
) -> np.ndarray:
    
    tracked_objs_within_t = scenario.get_tracked_objects_within_time_window_at_iteration(
        iteration=0,
        past_time_horizon=time_horizon,
        future_time_horizon=0,
    )
    tracked_objs_list = tracked_objs_within_t.tracked_objects.get_agents()

def construct_ego_future_states(
    scenario: AbstractScenario,
    time_horizon: float
) -> np.ndarray:
    
    ego_future_traj = list(scenario.get_ego_future_trajectory(
        iteration=0,
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

    ego_future_array, _ = construct_ego_future_states(scenario, time_horizon)

    look_ahead_pt_x = ego_future_array[-1][0]
    look_ahead_pt_y = ego_future_array[-1][1]
    look_ahead_pt_v = np.linalg.norm(np.asarray([ego_future_array[-1][3], ego_future_array[-1][4]]))

    look_ahead_pt = np.ndarray(
        [look_ahead_pt_x, look_ahead_pt_y, look_ahead_pt_v]
    )

    return look_ahead_pt

def process(
    data_root: str,
    map_root: str,
    db_files: str,
    map_version: str,
    past_time_horizon: float,
    future_time_horizon: float
) -> None:
    
    # get all scenarios
    log_db_files = discover_log_dbs(db_files)
    scenario_type_token_map = get_scenario_type_token_map(log_db_files)
    scenario_type_list = sorted(scenario_type_token_map.keys())

    # loop for all scenario
    for scenario_type in scenario_type_list:
        os.makedirs('./dataset/' + scenario_type)
        scenario_path = './dataset/' + scenario_type

        # loop for all dbs
        for log_db_file, token in scenario_type_token_map[scenario_type]:
            scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)

            # construct ego
            ego_past_states_array = construct_ego_past_states(scenario, past_time_horizon)

            # TODO: construct objects

            # get look-ahead point
            look_ahead_pt = get_look_ahead_point_by_time(scenario, future_time_horizon)


if __name__ == '__main__':

    data_root = '/home/gac/nuplan_explore/nuplan'
    map_root = '/home/gac/nuplan_explore/nuplan/dataset/maps'
    db_files = '/home/gac/nuplan_explore/nuplan/dataset/nuplan-v1.1/mini'
    map_version = '/home/gac/nuplan_explore/nuplan-maps-v1.0'

    past_time_horizon = 5
    future_time_horizon = 5

    process(
        data_root,
        map_root,
        db_files,
        map_version,
        past_time_horizon,
        future_time_horizon
    )