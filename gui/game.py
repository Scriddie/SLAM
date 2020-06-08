import pygame
import pygame.gfxdraw
import math
from .fps_counter import FPSCounter
import numpy as np
from gui.dustgrid_sprite import DustGridSprite
from gui.debug_display import DebugDisplay
from gui.localization_path import LocalizationPath
import time


def ti(arr):
    """
    Very short functionname to convert float-arrays to int
    Since pygame doesnt accept floats on its own
    """
    return np.rint(arr).astype(int).tolist()


class MobileRobotGame:

    def __init__(self, env_width, env_height, world, robot, robot_controller, scenario, particle_filter, debug):
        self.done = False
        self.debug = debug
        self.scenario = scenario
        self.particle_filter = particle_filter

        self.env_width = env_width
        self.env_height = env_height
        self.screen_width = env_width
        self.screen_height = env_height

        self.world = world
        self.robot = robot
        self.robot_controller = robot_controller
        self.start_time = time.time()

        self.fps_tracker = FPSCounter()
        self.reset = False

        if scenario == "evolutionary":
            self.robo_lines = [[self.robot.x, self.robot.y, self.robot.x, self.robot.y]]

        if scenario == "localization":
            self.surface = pygame.Surface((env_width, env_height))
            self.surface.fill(pygame.Color('white'))
            self.robo_lines_color = pygame.Color('black')
            self.robo_line_buffer = (self.robot.x, self.robot.y)  # This is an faster implementation

    def init(self):
        # Initialize pygame and modules that we want to use
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.fps_font = pygame.font.SysFont('Arial', 16)

        # Initialize sprites
        self.debug_display = DebugDisplay(self)
        if self.scenario == "evolutionary":
            self.dust_sprite = DustGridSprite(self.robot, self.world.dustgrid)
        elif self.scenario == "localization":
            self.localization_path = LocalizationPath(self, "green")
            self.dead_reckoning_path = LocalizationPath(self, "orange")

    def run(self, snapshot=False, snapshot_dir=""):
        # Main game loop
        ticks_last_frame = pygame.time.get_ticks()
        counter = 0
        while (not self.done and not self.reset):
            self.handle_events()

            # Update
            t = pygame.time.get_ticks()
            delta_time = (t - ticks_last_frame) / 1000.0
            ticks_last_frame = t
            self.update(delta_time)

            if snapshot:
                counter += 1
                if counter > 50000:
                    self.draw()
                    pygame.image.save(self.screen, snapshot_dir)
                    pygame.quit()
                    break
            else:
                self.draw()
            # Pygame uses double buffers
            # This swaps the buffers so everything we've drawn will now show up on the screen
            pygame.display.flip()
            self.fps_tracker.tick()

        pygame.quit()

    def update(self, delta_time):
        robo_diff = self.world.update(delta_time)
        self.robot_controller.update(delta_time)
        if self.scenario == "evolutionary":
            self.dust_sprite.update(delta_time)
            self.robo_lines[-1][-2] = self.robot.x
            self.robo_lines[-1][-1] = self.robot.y
            self.robo_lines.append([self.robot.x, self.robot.y, self.robot.x, self.robot.y])

        if self.scenario == "localization":  # Not to break the evolutionary part
            pygame.draw.line(self.surface, self.robo_lines_color, self.robo_line_buffer, (self.robot.x, self.robot.y))
            self.robo_line_buffer = (self.robot.x, self.robot.y)
            # self.localization_path.update(delta_time)

            # particle filter
            robo_pos = (self.robot.x, self.robot.y, self.robot.angle)
            self.particle_filter.update(robo_diff, robo_pos, self.robot.sensor_data)
            particle_pos = self.particle_filter.get_particle_pos()
            self.localization_path.update_loc(particle_pos)
            self.dead_reckoning_path.update_loc((self.robot.ideal_x, self.robot.ideal_y))

    def draw(self):
        if self.scenario == "evolutionary":
            self.dust_sprite.draw(self.screen)
        elif self.scenario == "localization":
            # Fix the screen updating
            self.screen.blit(self.surface, (0, 0), (0, 0, self.screen_width, self.screen_height))
            # Draw the beacon lines
            # for beacon in self.robot.beacons:
            #     pygame.draw.line(self.screen, pygame.Color('green'), beacon[0].location, (self.robot.x, self.robot.y),2)
            
            # draw particle filter map
            for i in self.particle_filter.draw_blobs():
                pygame.draw.circle(self.screen, pygame.Color("blue"), i, 2)
            # pf_robo_loc = self.particle_filter.robo_loc()
        
        self.__draw_robot__()

        if self.scenario == "localization":
            for i in self.particle_filter.draw_robos():
                pygame.draw.circle(self.screen, pygame.Color("green"), i, 3)
            orientation = self.particle_filter.draw_orientation()
            rotated_x = self.robot.x + math.cos(orientation) * (self.robot.radius - 1)
            rotated_y = self.robot.y + math.sin(orientation) * (self.robot.radius - 1)
            pygame.gfxdraw.line(self.screen, *ti((self.robot.x, self.robot.y)), *ti((rotated_x, rotated_y)),
                    pygame.Color('darkgreen'))

        # Draw walls
        for wall in self.world.walls:
            pygame.draw.line(self.screen, pygame.Color('black'), wall.start, wall.end, 1)

        # # Draw beacons
        # if self.scenario == "localization":
        #     if self.world.beacons is not None:
        #         for beacon in self.world.beacons:
        #             pygame.draw.circle(self.screen, pygame.Color('blue'),
        #                                (int(beacon.location[0]), int(beacon.location[1])), 5)

        if self.debug:
            self.debug_display.draw(self.screen)
            
        self.localization_path.draw(self.screen)
        self.dead_reckoning_path.draw(self.screen)

    def __draw_robot__(self):
        if self.scenario == "evolutionary":
            # draw ICC
            R, icc = self.robot.R, self.robot.icc
            if (max(icc) < 10e8 and min(icc) > -10e8):
                # In bounds
                pygame.draw.circle(self.screen, pygame.Color('orange'), ti(icc), 5)

            # Draw sensors
            for hit, dist in self.robot.sensor_data:
                if hit is None:
                    continue
                pygame.gfxdraw.line(self.screen, *ti((self.robot.x, self.robot.y)), *ti(hit), pygame.Color('red'))

        # For Particle Filter
        for hit, dist in self.robot.sensor_data:
            if hit is None:
                continue
            pygame.gfxdraw.line(self.screen, *ti((self.robot.x, self.robot.y)), *ti(hit), pygame.Color('red'))

        # Draw the shape of the robot as an circle with an line marking its rotation
        rotated_x = self.robot.x + math.cos(self.robot.angle) * (self.robot.radius - 1)
        rotated_y = self.robot.y + math.sin(self.robot.angle) * (self.robot.radius - 1)

        pygame.gfxdraw.filled_circle(self.screen, *ti((self.robot.x, self.robot.y)), self.robot.radius,
                                     pygame.Color('lightblue'))
        pygame.gfxdraw.line(self.screen, *ti((self.robot.x, self.robot.y)), *ti((rotated_x, rotated_y)),
                            pygame.Color('black'))
        pygame.gfxdraw.circle(self.screen, *ti((self.robot.x, self.robot.y)), self.robot.radius, pygame.Color('black'))

        if self.scenario == "evolutionary":
            for i in self.robo_lines:
                pygame.gfxdraw.line(self.screen, *ti(i), pygame.Color('black'))

    def handle_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.done = True
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset = True
                    return
                if event.key == pygame.K_t:
                    self.debug = not self.debug

        self.robot_controller.handle_events(events)
