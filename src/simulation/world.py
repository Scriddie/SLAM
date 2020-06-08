from simulation.line_wall import LineWall
from simulation.dustgrid import DustGrid
from pygame.math import Vector2
import numpy as np
import math


class World:
    def __init__(self, walls, width, height, scenario, beacons=None):
        self.walls = walls
        self.scenario = scenario
        if scenario == "evolutionary":
            self.dustgrid = DustGrid(width, height, 5)
        if scenario == "localization":
            self.beacons = beacons

    def set_robot(self, robot):
        self.robot = robot
        # The robot is now part of this world, so make sure he can access it
        self.robot.world = self

    def update(self, delta_time):
        robo_diff = self.robot.update(delta_time)
        if self.scenario == "evolutionary":
            self.dustgrid.clean_circle_area(self.robot.x, self.robot.y, self.robot.radius)
        return robo_diff

    def get_beacons(self, x, y, r):
        # Return the beacons in range
        beacons_in_range = []
        for beacon in self.beacons:
            b_x = beacon.location[0]
            b_y = beacon.location[1]
            # Check if in range
            distance_from_robot = math.sqrt((b_x - x) ** 2 + (b_y - y) ** 2)
            if distance_from_robot <= r:
                # Beacon is in range, check if the sensor collides with a wall. We cannot see beacons through walls
                hit = self.raycast_beacon((x, y), (b_x, b_y))
                if not hit:
                    # No walls in the way
                    beacons_in_range.append((beacon, distance_from_robot))

        return beacons_in_range

    def raycast(self, x, y, angle, max_length):
        # angle is in radians
        # Calculate the start from x and y
        start = Vector2(x, y)

        # Calculate the direction from angle
        direction = Vector2(math.cos(angle), math.sin(angle))
        end = start + direction * max_length

        closest_inter = None
        closest_dist = max_length
        closest_line = None
        for wall in self.walls:
            inter, dist, line = wall.check_line_intercept(start, end)

            # Check if the intersection is the closest to our start
            if (inter is not None) and (dist < closest_dist) and (line is not None):
                closest_inter = inter
                closest_dist = dist
                closest_line = line

        return closest_inter, closest_dist, closest_line

    def raycast_beacon(self, start, beacon_loc):
        # angle is in radians
        # Calculate the start from x and y
        start = Vector2(start)
        beacon_loc = Vector2(beacon_loc)

        for wall in self.walls:
            inter, dist, line = wall.check_line_intercept(start, beacon_loc)


            if inter is not None:
                # Check if the intersect is at the beacon location +- some error
                margin = 0.001
                if math.sqrt((inter[0] - beacon_loc[0])**2 + (inter[1] - beacon_loc[1])**2) < margin:
                    # Beacon is at the intersection
                    continue
                else:
                    # Beacon not in line of sight
                    return True

        return False

    def circle_collision(self, circle_position, radius):
        circle_position = Vector2(circle_position)

        collisions = []
        # prev_intercept = False
        for wall in self.walls:
            offset = wall.check_circle_intercept(circle_position, radius)
            if offset is not None:
                collisions.append((wall, offset))

        return collisions

    def slide_collision(self, circle_position, r_circle_position, radius):
        circle_position = Vector2(circle_position)
        r_circle_position = Vector2(r_circle_position)

        collisions = self.circle_collision(r_circle_position, radius)
        if len(collisions) == 0:
            return None

        # Check if the slide locations do not cause interceptions, take the first slide location that did not cause
        # an intercept
        for wall, offset in collisions:
            slide_loc = wall.calculate_sliding(r_circle_position, radius)

            free_from_all = True
            for wall in self.walls:
                intercept = wall.check_circle_intercept(slide_loc, radius)
                if intercept:
                    # Not free from all walls
                    free_from_all = False

            if free_from_all:
                # This slide position is free from all walls
                return slide_loc

        # At this point all the slide positions are behind walls, return the old location
        return circle_position
