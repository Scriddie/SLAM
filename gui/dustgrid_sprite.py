import pygame

class DustGridSprite:
    def __init__(self, robot, dustgrid):
        self.robot = robot
        self.dustgrid = dustgrid

        self.surface = pygame.Surface((dustgrid.width, dustgrid.height))
        self.surface2 = pygame.Surface((dustgrid.width, dustgrid.height), pygame.SRCALPHA)
        self.surface.fill(pygame.Color('white'))

        self.cleaned_color = pygame.Color('green')

    def update(self, delta_time):
        x_start, x_end, y_start, y_end = self.dustgrid.x_start, self.dustgrid.x_end, self.dustgrid.y_start, self.dustgrid.y_end
        pygame.draw.rect(self.surface, self.cleaned_color, (x_start, y_start, x_end - x_start, y_end - y_start))
        self.surface2.set_at((int(round(self.robot.x)), int(round(self.robot.y))), pygame.Color("black"))

    def draw(self, target_surface):
        target_surface.blit(self.surface, (0,0), (0,0, self.dustgrid.width, self.dustgrid.height))
        target_surface.blit(self.surface2, (0, 0), (0, 0, self.dustgrid.width, self.dustgrid.height))