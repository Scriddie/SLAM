import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from scipy.optimize import minimize
import genetic.functions as utils
import imageio

def plot_3d(fn, x1_low, x1_high, x2_low, x2_high, stepsize=0.1):
    # Create 2d raster
    x1_steps = np.arange(x1_low, x1_high, stepsize)
    x2_steps = np.arange(x2_low, x2_high, stepsize)
    x1, x2 = np.meshgrid(x1_steps, x2_steps)
    
    # Plot
    y = fn(x1, x2)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(x1, x2, y, cmap=cm.plasma, linewidth=0, antialiased=False)
    plt.show()

def visualize_heatmap(fn, history, extent, trail_lenght = 20,
    fname="particles.gif", output = "show"):
    fig = plt.figure()
    ax = plt.axes()
    plt.xlim(-2.0, 2.0)
    plt.ylim(-2.0, 2.0)

    # Step number needs to be global for the interactive stepping
    global step_num
    step_num = 0

    
    # these are matplotlib.patch.Patch properties for the textboax
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # Create heatmap
    X = np.arange(extent[0], extent[1], 0.1)
    Y = np.arange(extent[2], extent[3], 0.1)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = fn([X_grid, Y_grid])
    patch = plt.imshow(Z, extent=extent, cmap=cm.jet, zorder=0, origin="lower")
    fig.colorbar(patch, ax=ax)

    # Draw a star for the minimum   
    minimum = minimize(fn, [0, 0])
    ax.plot(minimum.x[0], minimum.x[1], "r*")

    # Draw a star for the average plot
    average_x = np.mean([p["pos"][0] for p in history[0]])
    average_y = np.mean([p["pos"][1] for p in history[0]])
    avg_pos, = ax.plot(average_x, average_y, "y*")
    
    # Create initial scatterplot
    x_points = [p["pos"][0] for p in history[0]]
    y_points = [p["pos"][1] for p in history[0]]
    sc = ax.scatter(x=x_points, y=y_points, color="black", zorder=2)

    # Create the initial textstring
    avg_mse = utils.distance_mse([average_x], [average_y], minimum.x[0], minimum.x[1])
    sum_mse = utils.distance_mse(x_points, y_points, minimum.x[0], minimum.x[1])

    textstr = f'Step        : {step_num}\nAvg MSE : {avg_mse:.4f}\nSum MSE: {sum_mse:.4f}'
    
    # Create initial lineplots
    num_particles = len(history[0])
    lines = []
    for i in range(num_particles):
        lines.append(ax.plot(0, 0, color="grey", zorder = 1)[0])

    # Create the initial text plot
    # place a text box in upper left in axes coords
    label = ax.text(0.08, 0.94, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    
    # Function for animating scatterplot
    def animate(i):
        state = history[i]

        # update particles
        x_points = [p["pos"][0] for p in state]
        y_points = [p["pos"][1] for p in state]
        sc.set_offsets(np.c_[x_points,y_points])

        # update the position of the mean star
        average_x = np.mean(x_points)
        average_y = np.mean(y_points)
        avg_pos.set_data(average_x, average_y)

        # Update the text box
        avg_mse = utils.distance_mse([average_x], [average_y], minimum.x[0], minimum.x[1])
        sum_mse = utils.distance_mse(x_points, y_points, minimum.x[0], minimum.x[1])

        textstr = f'Step        : {step_num}\nAvg MSE : {avg_mse:.4f}\nSum MSE: {sum_mse:.4f}'
        label.set_text(textstr)
    
    if(output == "step"):
        # Step through the frames   
        
        def on_keyboard(event):
            global step_num
            if event.key == 'right':
                if(step_num < len(history)-1):
                    step_num += 1
            elif event.key == 'left':
                if(step_num != 0):
                    step_num -= 1
                
            animate(step_num)

            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.gcf().canvas.mpl_connect('key_press_event', on_keyboard)
        plt.show()

    else:
        anim = animation.FuncAnimation(fig, animate, len(history), interval=20, blit=False)
        
        if(output == "show"):
            plt.show()
        elif(output == "save"):
            anim.save(fname, writer='imagemagick', fps=60)


def visualize_3D(fn, history):
    # TODO: this whole thing about the plot is still not quite right
    # (0 point is different)
    buffer = []
    for state in history:
        plt.close("all")
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-2, 2, 0.1)
        X, Y = np.meshgrid(X, Y)
        a = 0
        b = 1000
        Z = utils.rastrigin(X, Y)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0,
        antialiased=False)

        # visualize particles
        x_points = [i["pos"][0] for i in state]
        y_points = [i["pos"][1] for i in state]
        z_points = [i["fit"] for i in state]
        ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv')

        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        buffer.append(image)

    imageio.mimsave("particles.gif", buffer, )


if __name__ == "__main__":
    plot_3d(utils.rosenbrock, -2, 2, -2, 2)
    plot_3d(utils.rastrigin, -2, 2, -2, 2)
