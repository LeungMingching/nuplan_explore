import itertools
import logging
import random
import os
import json
from collections import defaultdict
from dataclasses import dataclass
from os.path import join, basename
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import numpy.typing as npt
from bokeh.document.document import Document
from bokeh.io import show
from bokeh.io.state import curstate
from bokeh.layouts import column

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
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

logger = logging.getLogger(__name__)


@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def construct_nuboard_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to nuBoard configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, 'config', 'common')
    config_name = 'default_nuboard'
    config_path = join(base_config_path, 'config/nuboard')
    experiment_dir = "file://" + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


def construct_simulation_hydra_paths(base_config_path: str) -> HydraConfigPaths:
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = join(base_config_path, 'config', 'simulation')
    experiment_dir = "file://" + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


def save_scenes_to_dir(
    scenario: AbstractScenario, save_dir: str, simulation_history: SimulationHistory
) -> SimulationScenarioKey:
    """
    Save scenes to a directory.
    :param scenario: Scenario.
    :param save_dir: Save path.
    :param simulation_history: Simulation history.
    :return: Scenario key of simulation.
    """
    planner_name = "tutorial_planner"
    scenario_type = scenario.scenario_type
    scenario_name = scenario.scenario_name
    log_name = scenario.log_name

    save_path = Path(save_dir)
    file = save_path / planner_name / scenario_type / log_name / scenario_name / (scenario_name + ".msgpack.xz")
    file.parent.mkdir(exist_ok=True, parents=True)

    # Create a dummy planner
    dummy_planner = _create_dummy_simple_planner(acceleration=[5.0, 5.0])
    simulation_log = SimulationLog(
        planner=dummy_planner, scenario=scenario, simulation_history=simulation_history, file_path=file
    )
    simulation_log.save_to_file()

    return SimulationScenarioKey(
        planner_name=planner_name,
        scenario_name=scenario_name,
        scenario_type=scenario_type,
        nuboard_file_index=0,
        log_name=log_name,
        files=[file],
    )


def _create_dummy_simple_planner(
    acceleration: List[float], horizon_seconds: float = 10.0, sampling_time: float = 20.0
) -> SimplePlanner:
    """
    Create a dummy simple planner.
    :param acceleration: [m/s^2] constant ego acceleration, till limited by max_velocity.
    :param horizon_seconds: [s] time horizon being run.
    :param sampling_time: [s] sampling timestep.
    :return: dummy simple planner.
    """
    acceleration_np: npt.NDArray[np.float32] = np.asarray(acceleration)
    return SimplePlanner(
        horizon_seconds=horizon_seconds,
        sampling_time=sampling_time,
        acceleration=acceleration_np,
    )


def _create_dummy_simulation_history_buffer(
    scenario: AbstractScenario, iteration: int = 0, time_horizon: int = 2, num_samples: int = 2, buffer_size: int = 2
) -> SimulationHistoryBuffer:
    """
    Create dummy SimulationHistoryBuffer.
    :param scenario: Scenario.
    :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
    :param time_horizon: the desired horizon to the future.
    :param num_samples: number of entries in the future.
    :param buffer_size: size of buffer.
    :return: SimulationHistoryBuffer.
    """
    past_observation = list(
        scenario.get_past_tracked_objects(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
    )

    past_ego_states = list(
        scenario.get_ego_past_trajectory(iteration=iteration, time_horizon=time_horizon, num_samples=num_samples)
    )

    # Dummy history buffer
    history_buffer = SimulationHistoryBuffer.initialize_from_list(
        buffer_size=buffer_size,
        ego_states=past_ego_states,
        observations=past_observation,
        sample_interval=scenario.database_interval,
    )

    return history_buffer


def serialize_scenario(
    scenario: AbstractScenario, num_poses: int = 12, future_time_horizon: float = 6.0
) -> SimulationHistory:
    """
    Serialize a scenario to a list of scene dicts.
    :param scenario: Scenario.
    :param num_poses: Number of poses in trajectory.
    :param future_time_horizon: Future time horizon in trajectory.
    :return: SimulationHistory containing all scenes.
    """
    simulation_history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
    ego_controller = PerfectTrackingController(scenario)
    simulation_time_controller = StepSimulationTimeController(scenario)
    observations = TracksObservation(scenario)

    # Dummy history buffer
    history_buffer = _create_dummy_simulation_history_buffer(scenario=scenario)

    # Get all states
    for _ in range(simulation_time_controller.number_of_iterations()):
        iteration = simulation_time_controller.get_iteration()
        ego_state = ego_controller.get_state()
        observation = observations.get_observation()
        traffic_light_status = list(scenario.get_traffic_light_status_at_iteration(iteration.index))

        # Log play back trajectory
        current_state = scenario.get_ego_state_at_iteration(iteration.index)
        states = scenario.get_ego_future_trajectory(iteration.index, future_time_horizon, num_poses)
        trajectory = InterpolatedTrajectory(list(itertools.chain([current_state], states)))

        simulation_history.add_sample(
            SimulationHistorySample(iteration, ego_state, trajectory, observation, traffic_light_status)
        )
        next_iteration = simulation_time_controller.next_iteration()

        if next_iteration:
            ego_controller.update_state(iteration, next_iteration, ego_state, trajectory)
            observations.update_observation(iteration, next_iteration, history_buffer)

    return simulation_history


def visualize_scenario(
    scenario: NuPlanScenario, save_dir: str = '/tmp/scenario_visualization/', bokeh_port: int = 8899
) -> None:
    """
    Visualize a scenario in Bokeh.
    :param scenario: Scenario object to be visualized.
    :param save_dir: Dir to save serialization and visualization artifacts.
    :param bokeh_port: Port that the server bokeh starts to render the generate the visualization will run on.
    """
    map_factory = NuPlanMapFactory(get_maps_db(map_root=scenario.map_root, map_version=scenario.map_version))

    simulation_history = serialize_scenario(scenario)
    simulation_scenario_key = save_scenes_to_dir(
        scenario=scenario, save_dir=save_dir, simulation_history=simulation_history
    )
    visualize_scenarios([simulation_scenario_key], map_factory, Path(save_dir), bokeh_port=bokeh_port)


def visualize_scenarios(
    simulation_scenario_keys: List[SimulationScenarioKey],
    map_factory: NuPlanMapFactory,
    save_path: Path,
    bokeh_port: int = 8899,
) -> None:
    """
    Visualize scenarios in Bokeh.
    :param simulation_scenario_keys: A list of simulation scenario keys.
    :param map_factory: Map factory object to use for rendering.
    :param save_path: Path where to save the scene dict.
    :param bokeh_port: Port that the server bokeh starts to render the generate the visualization will run on.
    """

    def complete_message() -> None:
        """Logging to print once the visualization is ready."""
        logger.info("Done rendering!")

    def notebook_url_callback(server_port: Optional[int]) -> str:
        """
        Callback that configures the bokeh server started by bokeh.io.show to accept requests
        from any origin. Without this, running a notebook on a port other than 8888 results in
        scenario visualizations not being rendered. For reference, see:
            - show() docs: https://docs.bokeh.org/en/latest/docs/reference/io.html#bokeh.io.show
            - downstream usage: https://github.com/bokeh/bokeh/blob/aae3034/src/bokeh/io/notebook.py#L545
        :param server_port: Passed by bokeh to indicate what port it started a server on (random by default).
        :return: Origin string and server url used by bokeh.
        """
        if server_port is None:
            return "*"
        return f"http://localhost:{server_port}"

    def bokeh_app(doc: Document) -> None:
        """
        Run bokeh app in jupyter notebook.
        :param doc: Bokeh document to render.
        """
        # Change simulation_main_path to a folder where you want to save rendered videos.
        nuboard_file = NuBoardFile(
            simulation_main_path=save_path.name,
            simulation_folder='',
            metric_main_path='',
            metric_folder='',
            aggregator_metric_folder='',
        )

        experiment_file_data = ExperimentFileData(file_paths=[nuboard_file])
        # Create a simulation tile
        simulation_tile = SimulationTile(
            doc=doc,
            map_factory=map_factory,
            experiment_file_data=experiment_file_data,
            vehicle_parameters=get_pacifica_parameters(),
        )

        # Render a simulation tile
        simulation_scenario_data = simulation_tile.render_simulation_tiles(simulation_scenario_keys)

        # Create layouts
        simulation_figures = [data.plot for data in simulation_scenario_data]
        simulation_layouts = column(simulation_figures)

        # Add the layouts to the bokeh document
        doc.add_root(simulation_layouts)
        doc.add_next_tick_callback(complete_message)

    # bokeh.io.show starts a server on `bokeh_port`, but doesn't return a handle to it. If it isn't
    # shut down, we get a port-in-use error when generating the new visualization. Thus, we search for
    # any server currently running on the assigned port and shut it down before calling `show` again.
    for server_uuid, server in curstate().uuid_to_server.items():
        if server.port == bokeh_port:
            server.unlisten()
            logging.debug("Shut down bokeh server %s running on port %d", server_uuid, bokeh_port)

    show(bokeh_app, notebook_url=notebook_url_callback, port=bokeh_port)


def get_default_scenario_extraction(
    scenario_duration: float = 15.0,
    extraction_offset: float = -2.0,
    subsample_ratio: float = 0.5,
) -> ScenarioExtractionInfo:
    """
    Get default scenario extraction instructions used in visualization.
    :param scenario_duration: [s] Duration of scenario.
    :param extraction_offset: [s] Offset of scenario (e.g. -2 means start scenario 2s before it starts).
    :param subsample_ratio: Scenario resolution.
    :return: Scenario extraction info object.
    """
    return ScenarioExtractionInfo(DEFAULT_SCENARIO_NAME, scenario_duration, extraction_offset, subsample_ratio)


def get_default_scenario_from_token(
    data_root: str, log_file_full_path: str, token: str, map_root: str, map_version: str
) -> NuPlanScenario:
    """
    Build a scenario with default parameters for visualization.
    :param data_root: The root directory to use for looking for db files.
    :param log_file_full_path: The full path to the log db file to use.
    :param token: Lidar pc token to be used as anchor for the scenario.
    :param map_root: The root directory to use for looking for maps.
    :param map_version: The map version to use.
    :return: Instantiated scenario object.
    """
    timestamp = get_lidarpc_token_timestamp_from_db(log_file_full_path, token)
    map_name = get_lidarpc_token_map_name_from_db(log_file_full_path, token)
    return NuPlanScenario(
        data_root=data_root,
        log_file_load_path=log_file_full_path,
        initial_lidar_token=token,
        initial_lidar_timestamp=timestamp,
        scenario_type=DEFAULT_SCENARIO_NAME,
        map_root=map_root,
        map_version=map_version,
        map_name=map_name,
        scenario_extraction_info=get_default_scenario_extraction(),
        ego_vehicle_parameters=get_pacifica_parameters(),
    )


def get_scenario_type_token_map(db_files: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get a map from scenario types to lists of all instances for a given scenario type in the database.
    :param db_files: db files to search for available scenario types.
    :return: dictionary mapping scenario type to list of db/token pairs of that type.
    """
    available_scenario_types = defaultdict(list)
    for db_file in db_files:
        for tag, token in get_lidarpc_tokens_with_scenario_tag_from_db(db_file):
            available_scenario_types[tag].append((db_file, token))

    return available_scenario_types


def visualize_nuplan_scenarios(
    data_root: str, db_files: str, map_root: str, map_version: str, bokeh_port: int = 8899
) -> None:
    """
    Create a dropdown box populated with unique scenario types to visualize from a database.
    :param data_root: The root directory to use for looking for db files.
    :param db_files: List of db files to load.
    :param map_root: The root directory to use for looking for maps.
    :param map_version: The map version to use.
    :param bokeh_port: Port that the server bokeh starts to render the generate the visualization will run on.
    """
    from IPython.display import clear_output, display
    from ipywidgets import Dropdown, Output
    from ipywidgets import Button, Layout
    from ipywidgets import HBox
    from ipywidgets import Text, Textarea, Label
    from ipywidgets import interact
    from ipywidgets import BoundedIntText
    from ipywidgets import Select

    log_db_files = discover_log_dbs(db_files)

    scenario_type_token_map = get_scenario_type_token_map(log_db_files)

    # define context type and corresponding id
    context_dict = {
                'lane keep' : 0,
                'lane change towards left' : 1,
                'lane change towards right' : 2,
                'nudge from left' : 3,
                'nudge from right' : 4,
                'turn left at intersection' : 5,
                'turn right at intersection' : 6,
            }

    base_dirs = {
                'lane keep' : 'lane_keep',
                'lane change towards left' : 'lane_change_towards_left',
                'lane change towards right' : 'lane_change_towards_right',
                'nudge from left' : 'nudge_from_left',
                'nudge from right' : 'nudge_from_right',
                'turn left at intersection' : 'turn_left_at_intersection',
                'turn right at intersection' : 'turn_right_at_intersection',
            }


    out = Output()
    drop_down = Dropdown(description='选择scenario: ', 
                         options=sorted(scenario_type_token_map.keys()))

    label_frame_scroller = Label("选择db文件: ")
    btn_next = Button(
            description='next', 
            layout=Layout(width='10%', height='30px'), 
            #layout=Layout(flex='1 1 auto', width='auto', height='30px'),
            disabled=True,
            button_style='success')

    btn_prev = Button(
            description='prev', 
            layout=Layout(width='10%', height='30px'), 
            #layout=Layout(flex='1 1 auto', width='auto', height='30px'),
            disabled=True,
            button_style='success')

    btn_random = Button(
            description='random', 
            layout=Layout(width='10%', height='30px'), 
            #layout=Layout(flex='1 1 auto', width='auto', height='30px'),
            disabled=True,
            button_style='success')

    #text_area = Text(value='', description='当前处理: ')
    #hbox = HBox([label_frame_scroller, btn_next, btn_prev, text_area])
    label0 = Label('选择起始帧: ')
    label1 = Label('当前选中: ')
    label2 = Label('-', 
                   flex='1 1 auto',
                   style=dict(
                                #font_style='italic',
                                font_weight='bold',
                                #font_variant="small-caps",
                                text_color='red',
                                text_decoration='underline'
                             )
                  )
    hbox = HBox([label_frame_scroller, btn_next, btn_prev, btn_random, label1, label2])

    #text_int_start = BoundedIntText(value=0, min=0, max=10000, step=1, description='选择FOI: ')
    text_int_start = BoundedIntText(value=0, 
                                    min=0, 
                                    max=5000, 
                                    step=1, 
                                    layout=Layout(flex='1 1 auto', width='auto'))
    text_int_end = BoundedIntText(value=0, 
                                    min=0, 
                                    max=5000, 
                                    step=1,
                                    layout=Layout(flex='1 1 auto', width='auto'))
    btn_record = Button(
            description='record', 
            #layout=Layout(width='10%', height='30px'), 
            #layout=Layout(flex='1 1 auto', width='auto'),
            layout=Layout(flex='1 1 auto', width='30px'),
            disabled=True,
            button_style='success')

    '''
    context_select = Select(options=['lane keep', 'lane change'],
                            value='lane keep',
                            description='context',
                            disabled=False)
    '''
    context_desc_list = list(context_dict.keys())
    '''
    context_select = Select(options=context_desc_list,
                            value=context_desc_list[0],
                            description='context',
                            disabled=False)
    txt_hbox = HBox([label0, text_int_start, text_int_end, context_select, btn_record])
    '''
    context_dropdown = Dropdown(options=context_desc_list)
    # set default value of dropdown item
    context_dropdown.value = context_desc_list[0]

    txt_hbox = HBox([label0, text_int_start, text_int_end, context_dropdown, btn_record])

    label3 = Label('已处理记录: ')
    label4 = Label('-',
                   style=dict(
                                font_weight='bold',
                                text_color='blue',
                                text_decoration='underline'
                             )
                  )
    txt_hbox1 = HBox([drop_down, label3, label4])

    # scenario info records
    current_scenario = None
    current_scenario_type = None
    current_log_db_file = None
    current_token_idx = 0
    max_num_cur_scenario_tokens = 0
    total_num_chosen_tokens = 0

    # -----------------------------------
    def dict_to_json(js_filename, 
                    origin_scenario_desc,
                    origin_scenario_token_idx,
                    context_desc,
                    context_id,
                    log_db_file,
                    token,
                    clip_head_frm_idx,
                    clip_head_timestamp,
                    clip_tail_frm_idx,
                    clip_tail_timestamp):

        with open(js_filename, 'w') as f:
            '''
            eg.
            dic = {
                    'origin_scenario_desc' : 'change_lane',
                    'origin_scenario_token_idx' : 100,
                    'context' : {
                                  'desc' : 'turn left',
                                  'id' : 0
                                },
                    'log_db_file' : 'xxx.db',
                    'token' : '237jgh2424vv',
                    'clip_head' : {
                                    'frm_idx' : 0,
                                    'timestamp' : 1122423435
                                  },
                    'clip_tail' : {
                                    'frm_idx' : 100,
                                    'timestamp' : 1122450989
                                  }
                   }
            '''
            dic = {
                    'origin_scenario_desc' : origin_scenario_desc,
                    'origin_scenario_token_idx' : origin_scenario_token_idx,
                    'context' : {
                                  'desc' : context_desc,
                                  'id' : context_id
                                },
                    'log_db_file' : log_db_file,
                    'token' : token,
                    'clip_head' : {
                                    'frm_idx' : clip_head_frm_idx,
                                    'timestamp' : clip_head_timestamp
                                  },
                    'clip_tail' : {
                                    'frm_idx' : clip_tail_frm_idx,
                                    'timestamp' : clip_tail_timestamp
                                  }
                   }

            f.write(json.dumps(dic, indent=4))


    ###############
    # move to next
    ###############
    def next_click(sender):
        global current_scenario, current_scenario_type, current_log_db_file, current_token, current_token_idx, max_num_cur_scenario_tokens

        try:
            with out:
                clear_output()
                current_token_idx += 1
                current_token_idx = current_token_idx if current_token_idx<max_num_cur_scenario_tokens else max_num_cur_scenario_tokens-1
                log_db_file, token = scenario_type_token_map[current_scenario_type][current_token_idx]
                scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)
                visualize_scenario(scenario, bokeh_port=bokeh_port)
                # update
                current_scenario = scenario
                current_log_db_file = log_db_file
                current_token = token
                # enable btn_record
                btn_record.disabled = False
            #print("scenario: {}, db file idx: {}".format(current_scenario_type, current_token_idx))
            #text_area.value = "{}, db idx: {}".format(current_scenario_type, current_token_idx)
            label2.value = "{}, total # tokens: {}, chosen token idx: {}".format(current_scenario_type, max_num_cur_scenario_tokens, current_token_idx)

        except ValueError as ve:
            # 强行跳过该次token
            btn_record.disabled = True


    ####################
    # move to previous
    ####################
    def prev_click(sender):
        global current_scenario, current_scenario_type, current_log_db_file, current_token, current_token_idx, max_num_cur_scenario_tokens

        try:
            with out:
                clear_output()
                current_token_idx -= 1
                current_token_idx = current_token_idx if current_token_idx>=0 else 0
                log_db_file, token = scenario_type_token_map[current_scenario_type][current_token_idx]
                scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)
                visualize_scenario(scenario, bokeh_port=bokeh_port)
                # update
                current_scenario = scenario
                current_log_db_file = log_db_file
                current_token = token
                # enable btn_record
                btn_record.disabled = False
            #print("scenario: {}, db file idx: {}".format(current_scenario_type, current_token_idx))
            #text_area.value = "{}, db idx: {}".format(current_scenario_type, current_token_idx)
            label2.value = "{}, total # tokens: {}, chosen token idx: {}".format(current_scenario_type, max_num_cur_scenario_tokens, current_token_idx)

        except ValueError as ve:
            # 强行跳过该次token
            btn_record.disabled = True


    ###################
    # randomly choose
    ###################
    def random_click(sender):
        global current_scenario, current_scenario_type, current_log_db_file, current_token, current_token_idx, max_num_cur_scenario_tokens

        try:
            with out:
                clear_output()
                num_all_tokens = len(scenario_type_token_map[current_scenario_type])
                random_chosen_token_idx = random.choice(range(num_all_tokens))
                #log_db_file, token = random.choice(scenario_type_token_map[current_scenario_type])
                log_db_file, token = scenario_type_token_map[current_scenario_type][random_chosen_token_idx]
                scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)
                visualize_scenario(scenario, bokeh_port=bokeh_port)

                # update
                current_scenario = scenario
                current_log_db_file = log_db_file
                current_token = token
                current_token_idx = random_chosen_token_idx
                # enable btn_record
                btn_record.disabled = False
            #print("scenario: {}, db file idx: {}".format(current_scenario_type, current_token_idx))
            #text_area.value = "{}, db idx: {}".format(current_scenario_type, current_token_idx)
            label2.value = "{}, total # tokens: {}, chosen token idx: {}".format(current_scenario_type, max_num_cur_scenario_tokens, current_token_idx)

        except ValueError as ve:
            # 强行跳过该次token
            btn_record.disabled = True


    #############################
    # record frames of interest
    #############################
    def foi_record_click(sender):
        global current_scenario, current_scenario_type, current_log_db_file, current_token, current_token_idx, max_num_cur_scenario_tokens
        global total_num_chosen_tokens

        #context_desc = context_select.value
        context_desc = context_dropdown.value
        context_id = context_dict[context_desc]
        clip_head_frm_idx = text_int_start.value
        clip_head_timestamp = current_scenario.get_time_point(clip_head_frm_idx).time_us
        clip_tail_frm_idx = text_int_end.value
        clip_tail_timestamp = current_scenario.get_time_point(clip_tail_frm_idx).time_us

        base_dir = "./DATA"
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        save_dir = os.path.join(base_dir, base_dirs[context_desc])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        dt = str(datetime.utcnow()).split(' ')
        js_filename = dt[0]+'_'+dt[1]+".json"
        js_filepath = os.path.join(save_dir, js_filename)

        dict_to_json(js_filepath, 
                    current_scenario_type,
                    current_token_idx,
                    context_desc,
                    context_id,
                    os.path.basename(current_log_db_file),
                    current_token,
                    clip_head_frm_idx,
                    clip_head_timestamp,
                    clip_tail_frm_idx,
                    clip_tail_timestamp)

        # 计算记录了几次token("一鱼多吃"算多次)
        total_num_chosen_tokens += 1

        # disable button
        btn_record.disabled = True

        label4.value = "{}, recorded token idx: {}, total # records: {}".format(current_scenario_type, current_token_idx, total_num_chosen_tokens)
        #print("recorded start: {}, end: {}, context: {}".format(text_int_start.value, text_int_end.value, context_select.value))


    # --------------------------------------------
    # event bundle
    btn_next.on_click(next_click)
    btn_prev.on_click(prev_click)
    btn_random.on_click(random_click)
    btn_record.on_click(foi_record_click)


    ###
    def context_dropdown_handler(change: Any) -> None:
        """
        once change detect, enable btn_record
        """
        #print("old: {}, new: {}".format(str(change.old), str(change.new)))
        btn_record.disabled = False
        


    def scenario_dropdown_handler(change: Any) -> None:
        """
        Dropdown handler that randomly chooses a scenario from the selected scenario type and renders it.
        :param change: Object containing scenario selection.
        """
        global current_scenario, current_scenario_type, current_log_db_file, current_token, current_token_idx, max_num_cur_scenario_tokens
        global total_num_chosen_tokens

        with out:
            clear_output()

            #logger.info("Randomly rendering a scenario...")
            scenario_type = str(change.new)

            #log_db_file, token = random.choice(scenario_type_token_map[scenario_type])
            #print('log_db_file: {}, token: {}'.format(log_db_file, token))
            #print(len(scenario_type_token_map[scenario_type]))
            max_num_cur_scenario_tokens = len(scenario_type_token_map[scenario_type])

            current_token_idx = 0
            log_db_file, token = scenario_type_token_map[scenario_type][current_token_idx]
            #print('log_db_file: {}, token: {}'.format(basename(log_db_file), token))
            scenario = get_default_scenario_from_token(data_root, log_db_file, token, map_root, map_version)
            visualize_scenario(scenario, bokeh_port=bokeh_port)
            # update
            current_scenario = scenario
            current_scenario_type = scenario_type
            current_log_db_file = log_db_file
            current_token = token
            # 每选定一个场景,后续记录总共记录了几条数据(token)
            total_num_chosen_tokens = 0

            #print("scenario: {}, db file idx: {}".format(current_scenario_type, current_token_idx))
            #text_area.value = "{}, db idx: {}".format(current_scenario_type, current_token_idx)
            label2.value = "{}, total # tokens: {}, chosen token idx: {}".format(current_scenario_type, max_num_cur_scenario_tokens, current_token_idx)

            # enable buttons
            btn_next.disabled = False
            btn_prev.disabled = False
            btn_random.disabled = False
            btn_record.disabled = False


    #display(next_btn)
    #display(prev_btn)
    #display(drop_down)
    display(txt_hbox1)
    display(hbox)
    display(txt_hbox)
    display(out)
    drop_down.observe(scenario_dropdown_handler, names='value')
    context_dropdown.observe(context_dropdown_handler, names='value')

