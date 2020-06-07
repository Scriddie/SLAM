from pygame.math import Vector2


class Beacon:
    def __init__(self, location, identifier):
        self.location = Vector2(location)
        self.x = location[0]
        self.y = location[1]
        self.id = identifier
