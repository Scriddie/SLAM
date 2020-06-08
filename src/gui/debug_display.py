import pygame
import time
from statistics import median

class V_QUEUE:
    # Class to get the mean q value for printing
    def __init__(self):
        self.v_queue = list()
        self.v_queue_length = 20

    def median(self):
        if (len(self.v_queue) < 1):
            return 0.0

        return median(self.v_queue)

    def update(self, v):
        self.v_queue.insert(0, v)
        if len(self.v_queue) > self.v_queue_length:
            self.dequeue()

    def dequeue(self):
        if len(self.v_queue) > 0:
            return self.v_queue.pop()
        return ("Queue Empty!")

class DebugDisplay:
    def __init__(self, game):
        self.game = game
        self.robot = game.robot
        self.world = game.world
        
        self.start_time = time.time()
        self.debug_color = pygame.Color('red')
        self.debug_font = pygame.font.SysFont('Arial', 16)
        self.v_queue = V_QUEUE()

    def update(self, delta_time):
        self.v_queue.update(self.robot.velocity)

    def draw(self, s):
        # Elapsed time since start
        elapsed_time = time.time() - self.start_time

        text_y_left = 20
        text_y_right = 20
        x_loc_right = self.game.screen_width - 100

        self.__render_text__(time.strftime("%M:%S", time.gmtime(elapsed_time)), s, x_loc_right, text_y_right)
        text_y_right += 20
        
        # FPS
        fps = self.game.fps_tracker.get_fps()
        self.__render_text__(f"FPS: {fps:3.0f}", s, 30, text_y_left)
        text_y_left += 20
        
        if self.game.scenario == "evolutionary":
            self.__render_text__(f"Vl: {round(self.world.robot.vl, 2)}", s, 30, text_y_left)
            text_y_left += 20
            
            self.__render_text__(f"Vl: {round(self.world.robot.vr, 2)}", s, 30, text_y_left)
            text_y_left += 20

        self.v_queue.update(self.robot.velocity)
        self.__render_text__(f"V: {round(self.v_queue.median() * 500, 1)}", s, 30, text_y_left)
        text_y_left += 20
        
        self.__render_text__(f"x: {round(self.robot.x, 2)}", s, 30, text_y_left)
        text_y_left += 20
        
        self.__render_text__(f"y: {round(self.robot.y, 2)}", s, 30, text_y_left)
        text_y_left += 20
        
        self.__render_text__(f"angle: {round(self.robot.angle, 2)}", s, 30, text_y_left)
        text_y_left += 20

        self.__render_text__(f"p_x: {round(self.robot.localizer.state_mu[0,0], 2)}", s, x_loc_right, text_y_right)
        text_y_right += 20

        self.__render_text__(f"p_y: {round(self.robot.localizer.state_mu[1,1], 2)}", s, x_loc_right, text_y_right)
        text_y_right += 20

        self.__render_text__(f"p_angle: {round(self.robot.localizer.state_mu[2,2], 2)}", s, x_loc_right, text_y_right)
        text_y_right += 20



    def __render_text__(self, text, screen, x, y):
        text_surface = self.debug_font.render(text, False, self.debug_color)
        screen.blit(text_surface, (x, y))