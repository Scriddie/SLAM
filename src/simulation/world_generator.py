from simulation.world import World
from simulation.robot import Robot
from simulation.line_wall import LineWall
from simulation.beacon import Beacon
import numpy as np
import math
import random


def create_rect_walls(x, y, width, height):
    bottomLeft = (x - width / 2, y - height / 2)
    topLeft = (x - width / 2, y + height / 2)
    topRight = (x + width / 2, y + height / 2)
    bottomRight = (x + width / 2, y - height / 2)

    return [
        LineWall(bottomLeft, topLeft),
        LineWall(topLeft, topRight),
        LineWall(topRight, bottomRight),
        LineWall(bottomRight, bottomLeft)
    ]


def create_trapezoid_walls(x, y, height, bottom_width, top_width):
    leftBottom = (x - bottom_width / 2, y + height / 2)
    leftTop = (x - top_width / 2, y - height / 2)
    rightBottom = (x + bottom_width / 2, y + height / 2)
    rightTop = (x + top_width / 2, y - height / 2)

    return [
        LineWall(leftTop, rightTop),
        LineWall(rightTop, rightBottom),
        LineWall(rightBottom, leftBottom),
        LineWall(leftBottom, leftTop)
    ]


def create_star_walls(x, y, inner_radius, outer_radius, num_points=5):
    delta_angle = (math.pi * 2) / (num_points * 2)
    prev_x = x + math.cos(math.pi * 2 - delta_angle) * inner_radius
    prev_y = y + math.sin(math.pi * 2 - delta_angle) * inner_radius
    prev_radius = inner_radius
    curr_radius = outer_radius

    walls = []

    for i in range(num_points * 2):
        # Generate a wall
        new_x = x + math.cos(delta_angle * i) * curr_radius
        new_y = y + math.sin(delta_angle * i) * curr_radius

        walls.append(LineWall((prev_x, prev_y), (new_x, new_y)))
        prev_x = new_x
        prev_y = new_y

        # Swap the radius so we alternate between inner and outer radius
        tmp = curr_radius
        curr_radius = prev_radius
        prev_radius = tmp

    return walls

def create_localization_maze_walls_and_beacons(width, height, offset):
    walls = []
    # walls.append(LineWall((0+offset, height/5+offset),(width*2/3+offset, height/5+offset)))
    # walls.append(LineWall((width/3+offset, height*2/5+offset),(width+offset, height*2/5+offset)))
    # walls.append(LineWall((width/3+offset, height*2/5+offset),(width/3+offset, height*4/5+offset)))
    # walls.append(LineWall((width*2/3+offset, height*3/5+offset),(width*2/3+offset, height+offset)))

    # upper horizontal
    walls.append(LineWall((width*1/4+offset, height/4+offset),(width*2/4+offset, height/4+offset)))
    # lower horizontal
    walls.append(LineWall((width*1/4+offset, height*3/4+offset),(width*2/4+offset, height*3/4+offset)))
    # left vertical
    walls.append(LineWall((width*1/4+offset, height*1/4+offset),(width*1/4+offset, height*3/4+offset)))
    # right vertical
    walls.append(LineWall((width*2/4+offset, height*1/4+offset),(width*2/4+offset, height*3/4+offset)))

    beacons = []
    # beacons.append(Beacon((0+offset, 0+offset), identifier=0))
    # beacons.append(Beacon((width+offset, 0 + offset), identifier=1))
    # beacons.append(Beacon((0+offset, height/5 + offset), identifier=2))
    # beacons.append(Beacon((width*2/3+offset, height/5 + offset), identifier=3))
    # beacons.append(Beacon((width*1/3+offset, height*2/5 + offset), identifier=4))
    # beacons.append(Beacon((width+offset, height*2/5 + offset), identifier=5))
    # beacons.append(Beacon((width*2/3+offset, height*3/5 + offset), identifier=6))
    # beacons.append(Beacon((width*1/3+offset, height*4/5 + offset), identifier=7))
    # beacons.append(Beacon((0+offset, height + offset), identifier=8))
    # beacons.append(Beacon((width*2/3+offset, height + offset), identifier=9))
    # beacons.append(Beacon((width+offset, height + offset), identifier=10))

    return walls, beacons

class WorldGenerator:
    def __init__(self, width, height, robot_radius, world_name, scenario, collision):
        self.width = width
        self.height = height
        self.robot_radius = robot_radius
        self.world_name = world_name
        self.scenario = scenario
        self.collision = collision

    def create_rect_world(self, random_robot=True):
        walls = create_rect_walls(self.width / 2, self.height / 2, self.width, self.height)

        world = World(walls, self.width, self.height, self.scenario)

        min_x = self.robot_radius
        max_x = self.width - self.robot_radius
        min_y = self.robot_radius
        max_y = self.height - self.robot_radius
        margin = 20
        x = random.uniform(min_x + margin, max_x - margin)
        y = random.uniform(min_y + margin, max_y - margin)
        robot_start_loc = (x, y, np.random.uniform(0, 2 * np.pi))

        robot = self.__add_robot__(world, random_robot=random_robot, robot_start_loc=robot_start_loc)
        return world, robot

    def create_double_rect_world(self, random_robot=True):
        outer_walls = create_rect_walls(self.width / 2, self.height / 2, self.width, self.height)
        inner_walls = create_rect_walls(self.width / 2, self.height / 2, self.width / 2, self.height / 2)
        walls = [*outer_walls, *inner_walls]

        world = World(walls, self.width, self.height, self.scenario)

        x_left = self.width / 2 - self.width * 3 / 8
        x_right = self.width / 2 + self.width * 3 / 8
        y_up = self.height / 2 + self.height * 3 / 8
        y_down = self.height / 2 - self.height * 3 / 8

        robot_start_loc = []
        x_options = np.linspace(x_left, x_right, 100)
        for x in x_options:
            robot_start_loc.append((x, y_up, np.random.uniform(0, 2 * np.pi)))
            robot_start_loc.append((x, y_down, np.random.uniform(0, 2 * np.pi)))

        y_options = np.linspace(y_down, y_up, 100)
        for y in y_options:
            robot_start_loc.append((x_left, y, np.random.uniform(0, 2 * np.pi)))
            robot_start_loc.append((x_right, y, np.random.uniform(0, 2 * np.pi)))

        robot_start_loc = random.choice(robot_start_loc)

        robot = self.__add_robot__(world, random_robot=random_robot, robot_start_loc=robot_start_loc)
        return world, robot

    def create_trapezoid_world(self, random_robot=True):
        border = create_rect_walls(self.width / 2, self.height / 2, self.width, self.height)
        trapezoid = create_trapezoid_walls(self.width / 2, self.height / 2, self.height, self.width, self.width / 2)
        walls = [*border, *trapezoid]

        world = World(walls, self.width, self.height, self.scenario)

        min_x = self.robot_radius
        max_x = self.width - self.robot_radius
        min_y = self.robot_radius
        max_y = self.height - self.robot_radius
        margin_x = self.width / 3
        margin_y = 20
        x = random.uniform(min_x + margin_x, max_x - margin_x)
        y = random.uniform(min_y + margin_y, max_y - margin_y)
        robot_start_loc = (x, y, np.random.uniform(0, 2 * np.pi))

        robot = self.__add_robot__(world, random_robot=random_robot, robot_start_loc=robot_start_loc)
        return world, robot

    def create_double_trapezoid_world(self, random_robot=True):
        border = create_rect_walls(self.width / 2, self.height / 2, self.width, self.height)
        outer_walls = create_trapezoid_walls(self.width / 2, self.height / 2, self.height, self.width, self.width / 2)
        inner_walls = create_trapezoid_walls(self.width / 2, self.height / 2, self.height / 2, self.width / 2,
                                             self.width / 4)
        walls = [*border, *outer_walls, *inner_walls]

        world = World(walls, self.width, self.height, self.scenario)

        x_left_down = self.width / 2 - self.width * 3 / 8
        x_right_down = self.width / 2 + self.width * 3 / 8
        x_left_up = self.width / 2 - self.width * 1 / 5
        x_right_up = self.width / 2 + self.width * 1 / 5
        y_up = self.height / 2 + self.height * 3 / 8
        y_down = self.height / 2 - self.height * 3 / 8

        robot_start_loc = []
        x_options = np.linspace(x_left_down, x_right_down, 100)
        for x in x_options:
            robot_start_loc.append((x, y_up, np.random.uniform(0, 2 * np.pi)))

        x_options = np.linspace(x_left_up, x_right_up, 100)
        for x in x_options:
            robot_start_loc.append((x, y_down, np.random.uniform(0, 2 * np.pi)))

        robot_start_loc = random.choice(robot_start_loc)

        robot = self.__add_robot__(world, random_robot=random_robot, robot_start_loc=robot_start_loc)
        return world, robot

    def create_star_world(self, random_robot=True):
        border = create_rect_walls(self.width / 2, self.height / 2, self.width, self.height)
        star = create_star_walls(self.width / 2, self.height / 2, self.height / 4, self.height / 2)
        walls = [*border, *star]

        world = World(walls, self.width, self.height, self.scenario)

        radius = min(self.width / 6, self.height / 6)
        radius = random.random() * radius
        angle = random.random() * 2 * math.pi

        robot_start_loc = (
            int(self.width / 2 + radius * math.cos(angle)), int(self.height / 2 + radius * math.sin(angle)),
            np.random.uniform(0, 2 * np.pi))

        robot = self.__add_robot__(world, random_robot=random_robot, robot_start_loc=robot_start_loc)
        return world, robot

    def create_localization_maze(self, random_robot=False):
        border_buffer = 10
        effective_width = self.width - border_buffer*2
        effective_height = self.height - border_buffer*2
        border = create_rect_walls(self.width / 2, self.height / 2, effective_width, effective_height)
        internal_walls, internal_beacons = create_localization_maze_walls_and_beacons(effective_width, effective_height, border_buffer)

        walls = [*border, *internal_walls]

        beacons = [*internal_beacons]

        world = World(walls, self.width, self.height, self.scenario, beacons=beacons)
        robot_start_loc = ((effective_width/6) + border_buffer, (effective_height/10) + border_buffer, 0)
        robot = self.__add_robot__(world, random_robot=random_robot, robot_start_loc=robot_start_loc)
        return world, robot

    def create_random_world(self, random_robot=True):
        world_func = random.choice([
            self.create_rect_world,
            self.create_double_rect_world,
            self.create_trapezoid_world,
            self.create_double_trapezoid_world,
            self.create_star_world
        ])

        return world_func(random_robot=random_robot)

    def create_world(self, random_robot=True):
        if self.world_name == "rect_world":
            return self.create_rect_world(random_robot)
        elif self.world_name == "double_rect_world":
            return self.create_double_rect_world(random_robot)
        elif self.world_name == "trapezoid_world":
            return self.create_trapezoid_world(random_robot)
        elif self.world_name == "double_trapezoid_world":
            return self.create_double_trapezoid_world(random_robot)
        elif self.world_name == "star_world":
            return self.create_star_world(random_robot)
        elif self.world_name == "random":
            return self.create_random_world(random_robot)
        elif self.world_name == "localization_maze":
            return self.create_localization_maze(random_robot)
        else:
            raise ValueError("Wrong world name")

    def __add_robot__(self, world, random_robot=True, robot_start_loc=None):
        if random_robot and (robot_start_loc is None):
            min_x = self.robot_radius
            max_x = self.width - self.robot_radius
            min_y = self.robot_radius
            max_y = self.height - self.robot_radius
            margin = 20
            
            while True:
                rand_x = random.uniform(min_x + margin, max_x - margin)
                rand_y = random.uniform(min_y + margin, max_y - margin)
    
                collisions = world.circle_collision((rand_x, rand_y), self.robot_radius)
                if len(collisions) == 0:
                    break
        elif robot_start_loc is None:
            robot_start_loc = (self.width / 2, self.height / 2, 0)

        # Place robot randomly until no collisions occur
        robot = Robot(*robot_start_loc, scenario=self.scenario, radius=self.robot_radius,collision=self.collision)
        world.set_robot(robot)
        
        return robot
