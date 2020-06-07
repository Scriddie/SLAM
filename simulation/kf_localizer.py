import numpy as np


class KFLocalizer:
    def __init__(self, state_mu, state_std, motion_model):
        self.state_mu = np.array(state_mu)
        self.state_std = np.array(state_std)

        self.motion_model = motion_model

    def predict(self, action, delta_time, motion_noise):
        # Note the real formula is A * state_mu + B * action
        # We assume A is an identity-matrix and B * action is represented by the motion model
        self.state_mu = self.motion_model(self.state_mu, action, delta_time)
        # Note the real formula is A * state_std * A^T + motion_noise
        # TODO: double check if this makes sense
        self.state_std += motion_noise
        
    def correct(self, z, sensor_noise):
        z = np.array(z)
        # C is identity in all calculations
        K = np.matmul(self.state_std, np.linalg.inv(self.state_std + sensor_noise))
        self.state_mu = self.state_mu + np.matmul(K, z - self.state_mu)
        self.state_std = np.matmul((np.identity(K.shape[0]) - K), self.state_std)