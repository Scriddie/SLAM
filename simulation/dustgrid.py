import numpy as np

class DustGrid:
    def __init__(self, width, height, cell_size):
        # Make sure the rectangle can be perfectly divided into cells
        # We could remove this check, but that could result in missing rows and columns
        assert width % cell_size == 0, "The width has to be divisible by cell_size"
        assert height % cell_size == 0, "The height has to be divisible by cell_size"
        
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        self.cells = np.ones((height // cell_size, width // cell_size), dtype=np.bool)
        self.cleaned_cells = 0
        
    def clean_circle_area(self, circle_x, circle_y, radius):
        # Calculate the area that is cleaned by the circle
        # Note that even partially occupied cells are going to be cleaned
        x_start = int((circle_x - radius) // self.cell_size)
        x_end = int((circle_x + radius) // self.cell_size + 1)
        y_start = int((circle_y - radius) // self.cell_size)
        y_end = int((circle_y + radius) // self.cell_size + 1)
        
        # Clean area
        self.cleaned_cells += np.sum(self.cells[y_start:y_end, x_start:x_end])
        self.cells[y_start:y_end, x_start:x_end] = 0
        
        self.x_start = x_start * self.cell_size
        self.x_end = x_end * self.cell_size
        self.y_start = y_start * self.cell_size
        self.y_end = y_end * self.cell_size