import random
from copy import deepcopy
import numpy as np
from scipy.stats import norm
from bresenham import bresenham
import threading
import math
import concurrent


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


class Particle:
    def __init__(self, pos, world):
        self.pos = pos
        # world should be all zeros initially
        self.world = world

    def update_position(self, pos_delta):
        """ update position """
        dx, dy, dangle = pos_delta
        # TODO somehow dangle doesn't work at all!
        # print(dx, dy, dangle)
        sigma = 1
        # sample from around predicted location
        if not ((dx == 0) and (dy == 0) and (dangle == 0)):
            self.pos[0] += dx + random.gauss(0, sigma)
            self.pos[1] += dy + random.gauss(0, sigma)
            self.pos[2] += dangle + random.gauss(0, 0.1*sigma)

    def closest_point(self, point):
        point = np.array(list(point))
        min_dist = np.inf
        for i in self.world.grid.keys():
            dist = np.linalg.norm(np.array(i) - point)
            min_dist = min(min_dist, dist)
        return min_dist

    def likelihood_field(self, sensor_data):
        """ to what extend does our map fit the observations? """
        pnorm = norm(0, 5).pdf
        q = 1.0
        for hit, dist in sensor_data:
            if hit is None:
                continue
            min_dist = self.closest_point(hit)
            q = q * pnorm(min_dist)
        return q

    def update_map(self, robo_pos, sensor_data, remove_objects=True):
        """ update occupancy grid, add sensor noise """
        no_contact_list = []
        for hit, dist in sensor_data:
            if hit is None:
                continue
            else:
                # we need to do this relative to imaginary robo pos and robo direction!!!
                # add some noise
                dx = hit[0] - robo_pos[0] + random.gauss(0, 1)
                dy = hit[1] - robo_pos[1] + random.gauss(0, 1)
                dx, dy = rotate((0, 0), (dx, dy), (robo_pos[2] - self.pos[2]))

                x = self.pos[0] + dx
                y = self.pos[1] + dy
                x = int(np.round(x))
                y = int(np.round(y))
                self.world.grid[(x, y)] = 1

                if remove_objects:
                    robo_x = int(np.round(robo_pos[0]))
                    robo_y = int(np.round(robo_pos[1]))
                    no_contact_list += list(bresenham(robo_x, robo_y, x, y))[1:-1]
        if remove_objects:
            for k in self.world.grid.keys():
                if k in no_contact_list:
                    self.world.grid[k] = 0


class ParticleFilter:
    def __init__(self, bot_pos, world, n_part=30):
        # draw a bunch of particles
        self.resample_ratio = 0.9
        self.n_part = n_part
        self.weights = np.ones((n_part), dtype=float) / n_part
        p = Particle(bot_pos, deepcopy(world))
        self.particle_list = []
        # copy the same guy
        for i in range(n_part):
            self.particle_list.append(deepcopy(p))
        self.best_particle = self.particle_list[0]
    
    def update(self, pos_delta, robo_pos, sensor_data):
        dx, dy, dangle = pos_delta
        if not ((dx == 0) and (dy == 0) and (dangle == 0)):
            field = np.zeros((self.n_part), dtype=float)
            threads = []
            for idx, i in enumerate(self.particle_list):
            #     threads.append(threading.Thread(target=i.update_position, args=(pos_delta,)))
            # for t in threads:
            #     t.start()
            # for t in threads:
            #     t.join()
                i.update_position(pos_delta)
            
            threads = []
            for idx, i in enumerate(self.particle_list):
            #     threads.append(threading.Thread(target=i.update_map, args=(robo_pos, sensor_data,)))
            # for t in threads:
            #     t.start()
            # for t in threads:
            #     t.join()
                i.update_map(robo_pos, sensor_data)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for idx, i in enumerate(self.particle_list):
                    future = executor.submit(i.likelihood_field, sensor_data)
                    field[idx] = future.result()
                # field[idx] = i.likelihood_field(sensor_data)
            
            # resample only when stuff happens & only 2 in 100 rounds to preserve connections
            if np.random.uniform() > self.resample_ratio:
                self.resample()

            self.weights = field / np.sum(field)
            
            # remember our best particle
            self.best_particle = self.particle_list[np.argmax(self.weights)]

    def resample(self):
        # which ones have we resampled already
        map_rec = np.zeros((self.n_part))

        # resample according to weights
        re_id = np.random.choice(self.n_part, self.n_part, p=list(self.weights))
        
        # create new particle list
        new_particle_list = []
        for i in range(self.n_part):
            new_particle_list.append(deepcopy(self.particle_list[re_id[i]]))
        
        self.particle_list = new_particle_list
        self.weights = np.ones((self.n_part), dtype=float) / self.n_part

    def get_particle_pos(self):
        return (self.best_particle.pos[0], self.best_particle.pos[1])

    def draw_robos(self):
        return [(int(a.pos[0]), int(a.pos[1])) for a in self.particle_list]

    def draw_orientation(self):
        return self.best_particle.pos[2]

    def draw_blobs(self):
        """ draw occupied grid cells """
        all_blobs = []
        # for i in self.particle_list:
        i = self.best_particle
        all_blobs += [i for i, v in i.world.grid.items() if v==1]
        return all_blobs

    # one more def??



    
