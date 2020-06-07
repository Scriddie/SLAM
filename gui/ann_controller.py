import numpy as np

def exponential_decay(x, start=1, end_factor=0.1, factor=30):
    """
        Taken from the slides, dont ask me how it works
    """
    x = np.array(x)
    return start + (start * end_factor - start) * (1 - np.exp(-x/factor))

def get_action(robot, ann, feedback):
    inp = exponential_decay([dist for hit, dist in robot.sensor_data])
    return ann.predict(inp.reshape(-1, 1), feedback).reshape(-1) * 2 - 1

def apply_action(robot, ann, feedback):
    action = get_action(robot, ann, feedback)
    
    # The network outputs values between 0 and 1 for each motor
    # We treat those outputs as normalized velocities
    robot.vl = action[0] * robot.max_v
    robot.vr = action[1] * robot.max_v

class ANNController:
    def __init__(self, robot, ann, step_size_ms=270, feedback=True):
        self.robot = robot
        self.ann = ann
        self.feedback = feedback
        self.step_size = step_size_ms / 1000
        self.passed_time = 0.0
        
    def update(self, delta_time):
        self.passed_time += delta_time
        if self.passed_time > self.step_size:
            apply_action(self.robot, self.ann, self.feedback)
            self.passed_time = 0
                    
    def handle_events(self, events):
        pass