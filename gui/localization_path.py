import pygame
import math

# Draws a dashed curve
# Works by using the fraction variable to keep track of the dash strokes
# fraction from 0 to 1 means dash
# fraction from 1 to 2 means no dash
def draw_dashed_curve(surf, color, start, end, fraction, dash_length=10):
    start = pygame.Vector2(start)
    end = pygame.Vector2(end)
    
    delta = end - start
    length = delta.length()
    if length < 0.0000001:
        return fraction + length
    
    new_fraction = fraction + length / dash_length
    slope = delta / length
    if fraction < 1:
        # If we're in the middle of drawing an dash, finish or continue it
        dash_end = start + slope * (min(1 - fraction, new_fraction - fraction)) * dash_length
        pygame.draw.line(surf, color, start, dash_end, 2)
    
    # Draw the remaining full-dashes
    for index in range(2, int(new_fraction), 2):
        dash_start = start + (slope * index * dash_length)
        dash_end = start + (slope * (index + 1) * dash_length)
        pygame.draw.line(surf, color, dash_start, dash_end, 2)
    
    if (new_fraction % 2) < 1:
        # There is still an half finished dash left to draw
        dash_start = start + slope * int(new_fraction - fraction) * dash_length
        pygame.draw.line(surf, color, dash_start, end, 2)
    
    return new_fraction % 2

class LocalizationPath:
    def __init__(self, game):
        self.game = game
        self.robot = game.robot
        self.localizer = self.robot.localizer
        
        self.path_surface = pygame.Surface((game.screen_width, game.screen_height), pygame.SRCALPHA)
        self.path_color = pygame.Color('orange')
        self.old_pos = (self.localizer.state_mu[0], self.localizer.state_mu[1])
        self.passed_time = 0
        self.dash_fraction = 0

    def update(self, delta_time):
        new_pos = (self.localizer.state_mu[0], self.localizer.state_mu[1])
        self.dash_fraction = draw_dashed_curve(surf=self.path_surface, color=self.path_color, start=self.old_pos, end=new_pos, fraction=self.dash_fraction)
        self.old_pos = new_pos
        
        # Freeze the uncertainty ellipse after a set amount of time
        self.passed_time += delta_time
        if self.passed_time > 2:
            self.__draw_uncertainty_ellipse__(self.path_surface)
            self.passed_time = 0
    
    def draw(self, surface):
        surface.blit(self.path_surface, (0,0), (0,0, self.game.screen_width, self.game.screen_height))
        self.__draw_uncertainty_ellipse__(surface)
        
    def __draw_uncertainty_ellipse__(self, surface):
        x_mu = self.localizer.state_mu[0]
        y_mu = self.localizer.state_mu[1]
        x_std = self.localizer.state_std[0,0]
        y_std = self.localizer.state_std[1,1]
        
        pygame.gfxdraw.ellipse(surface, int(x_mu), int(y_mu), int(x_std), int(y_std), self.path_color)