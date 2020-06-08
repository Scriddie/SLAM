import sys
sys.path.append("MRS")
from gui.game import MobileRobotGame
from gui.human_controller import HumanController
from gui.ann_controller import ANNController
from genetic.ANN import ANN
from simulation.world_generator import WorldGenerator
import ParticleFilter
from grid import Grid
import argparse
import os
from copy import deepcopy
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--scenario", default="localization", choices=["localization", "evolutionary"],
                        help="Select a scenario: 'localization' or 'evolutionary'")

    parser.add_argument("--human", action="store_true", default=True,
                        help="manual robot control")

    parser.add_argument("--collision", action="store_true", default=True, help="collision with the walls")

    parser.add_argument("--model_name", default="pre_trained_models/model_random.p",
                        help="robot control model name")
    parser.add_argument("--world_name", default="random",
                        help="world name of the environment, options: rect_world, double_rect_world, trapezoid_world, "
                             "double_trapezoid_world, star_world, random")
    parser.add_argument("--snapshot", action="store_true", default=False,
                        help="take a snapshot")
    parser.add_argument("--snapshot_dir", default="_snapshots/latest.png",
                        help="file name for snapshot")
    parser.add_argument("--debug", action="store_true", default=False, help="Switch on debug information")

    args = parser.parse_args()

    use_human_controller = args.human
    debug = args.debug
    scenario = args.scenario
    collision = args.collision

    if scenario == "localization":
        # set up environment
        WIDTH = 500
        HEIGHT = 500
        world_name = "localization_maze"
        env_params = {"env_width": WIDTH, "env_height": HEIGHT}
        world_generator = WorldGenerator(WIDTH, HEIGHT, 20, world_name, scenario, collision)

        if use_human_controller:
            controller_func = HumanController

        # Game loop
        while True:
            world, robot = world_generator.create_world(random_robot=False)

            ### TODO initialize stuff for particle filter
            robo_pos = (robot.x, robot.y, robot.angle)
            grid_world = Grid((WIDTH, HEIGHT), 1)
            pf = ParticleFilter.ParticleFilter(deepcopy(list(robo_pos)), grid_world)
            # pf.particle_list[0].world.grid
            ###

            controller = controller_func(robot, scenario)
            env_params["world"] = world
            env_params["robot"] = robot
            env_params["robot_controller"] = controller
            env_params["scenario"] = scenario
            env_params["particle_filter"] = pf
            env_params["debug"] = debug

            game = MobileRobotGame(**env_params)
            game.init()
            game.run(args.snapshot, args.snapshot_dir)

            if not game.reset:
                break

    # elif scenario == "evolutionary":
    #     # set up environment
    #     WIDTH = 400
    #     HEIGHT = 400
    #     env_params = {"env_width": WIDTH, "env_height": HEIGHT}
    #     robot_kwargs = {"n_sensors": 12}
    #     world_generator = WorldGenerator(WIDTH, HEIGHT, 20, args.world_name, scenario, collision)

    #     if use_human_controller:
    #         controller_func = HumanController
    #     else:
    #         model_path = args.model_name
    #         controller_func = lambda robot, scenario: ANNController(
    #             robot, ANN.load(model_path))

    #     # Game loop
    #     while True:
    #         world, robot = world_generator.create_world(random_robot=True)

    #         controller = controller_func(robot, scenario)
    #         env_params["world"] = world
    #         env_params["robot"] = robot
    #         env_params["robot_controller"] = controller
    #         env_params["scenario"] = scenario
    #         env_params["debug"] = debug

    #         game = MobileRobotGame(**env_params)
    #         game.init()
    #         game.run(args.snapshot, args.snapshot_dir)

    #         if not game.reset:
    #             break
