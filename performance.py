"""Measure performance of our robot"""

from world_creator import WorldCreator
import time

def stop_time(world, robot, num_steps=10000):
    start_time = time.time()
    fps_s = time.time()
    for step in range(num_steps):
        robot.update(1000 / 60)

        if(step % 100 == 0):
            fps_e = time.time()
            fps_dif = fps_e - fps_s
            fps = 1.0/(fps_dif / 100)
            print("fps:",fps)
            fps_s = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Needed {elapsed_time}s for {num_steps}steps")
    print(f"{num_steps / elapsed_time} steps per second")
    
if __name__ == "__main__":
    WIDTH = 1000
    HEIGHT = 650
    env_params = {
        "env_width": WIDTH,
        "env_height": HEIGHT
    }
    creator = WorldCreator(WIDTH, HEIGHT)
    world, robot = creator.create_random_world()
    
    stop_time(world, robot, num_steps=10000)
    