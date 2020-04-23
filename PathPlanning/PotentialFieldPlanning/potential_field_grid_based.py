from potential_field_metric import PotentialFieldPlanner
from utilities import draw_heatmap
from scenarios_params import define_scenario

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import time

# Parameters
# AREA_WIDTH = 0.0  # potential area width [m]
START_INVERSE_KP = 1.0            # attractive potential gain
START_LIN_INFLUENCE_DISTANCE = 200.0    # [m]
# GOAL_INFLUENCE_DISTANCE = 200.0 # [m]
GOAL_KP = 0.1
ETA = 1000.0        # repulsive potential gain
ETA2 = 500.0
MIN_DISTANCE = 0.1
MIN_OBSTACLE_DISTANCE = 5.0
DESIRED_DISTANCE = 10
MAX_OBSTACLE_DISTANCE = 50
MAX_POTENTIAL = 50

show_animation = False
show_result = True


class PotentialFieldPlannerGrid(PotentialFieldPlanner):
    # override
    def set_problem(self, grid, sx, sy, gx, gy, ox, oy, resolution):
        self.grid = grid
        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy
        # self.ox = ox
        # self.oy = oy
        self.resolution = resolution

        # transform to indices
        self.minx = 0
        self.miny = 0
        self.maxx = self.grid.shape[0]
        self.maxy = self.grid.shape[1]

        self.xw = self.grid.shape[0]
        self.yw = self.grid.shape[1]

        self.sx_id = (sx - self.minx) / resolution
        self.sy_id = (sy - self.miny) / resolution
        self.gx_id = (gx - self.minx) / resolution
        self.gy_id = (gy - self.miny) / resolution
        # self.ox_id = ox
        # self.oy_id = oy

        if self.sx_id < self.minx or self.sx_id >= self.maxx or self.gx_id < self.minx or self.gx_id >= self.maxx:
            print "start or goal exceed grid dimensions!"
            return False

        print ("start id: {} , {}".format(self.sx_id, self.sy_id))
        print ("goal id: {} , {}".format(self.gx_id, self.gy_id))
        # print ("ox id: {}".format(self.ox_id))
        # print ("oy id: {}".format(self.oy_id))
        print ("minx : {}".format(self.minx))
        print ("miny : {}".format(self.miny))
        print ("maxx : {}".format(self.maxx))
        print ("maxy : {}".format(self.maxy))
        return True

    def set_motion_model(self, motion_model):
        self.motion_model = motion_model

    def calc_potential_field(self, obstacle_field, goal_field, start_field_inv, start_field_lin):
        start = time.time()
        self.potential = np.zeros((self.xw, self.yw))
        print ("potential size: {}".format(self.potential.shape))

        distance_array = scipy.ndimage.distance_transform_edt(self.grid)
        # self.potential = distance_array
        if obstacle_field:
            uo = self.repulsive_potential(distance_array)
            self.potential += uo

        if goal_field:
            dist_goal = self.calc_distance_to_index(self.grid, self.gx_id, self.gy_id)
            self.potential += self.goal_attractive_potential(dist_goal)

        if start_field_inv:
            dist_start = self.calc_distance_to_index(self.grid, self.sx_id, self.sy_id)
            self.potential += self.start_repulsive_potential_inverse(dist_start)

        if start_field_lin:
            dist_start = self.calc_distance_to_index(self.grid, self.sx_id, self.sy_id)
            self.potential += self.start_repulsive_potential_linear(dist_start)

        end = time.time()
        print ("Caulculate potential time: {} s".format((end - start)))

        if show_animation:
            draw_heatmap(self.potential)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [
                exit(0) if event.key == 'escape' else None])
            plt.plot(self.sx_id, self.sy_id, "*k")
            plt.plot(self.gx_id, self.gy_id, "*m")

        return self.potential

    def calc_distance_to_index(self, grid, ix, iy):
        dist = np.zeros((grid.shape[0], grid.shape[1]))

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                dist[x, y] = np.hypot(x - ix, y - iy) * self.resolution

        return dist

    def repulsive_potential(self, d):
        conds = [d <= MIN_OBSTACLE_DISTANCE, (d > MIN_OBSTACLE_DISTANCE) & (d <= MAX_OBSTACLE_DISTANCE),
                 d > MAX_OBSTACLE_DISTANCE]
        funcs = [lambda x: MAX_POTENTIAL, lambda x: 0.5 * ETA * (x**-1 - DESIRED_DISTANCE**-1) ** 2,
                 lambda x: 0.0]
        return np.minimum(np.piecewise(d, conds, funcs), MAX_POTENTIAL)

    def goal_attractive_potential(self, d):
        # influence_distance_pixels = GOAL_INFLUENCE_DISTANCE/self.resolution
        # slope = MAX_POTENTIAL/influence_distance_pixels
        return np.minimum(GOAL_KP * d, MAX_POTENTIAL)

    def start_repulsive_potential_inverse(self, d):
        return np.minimum(START_INVERSE_KP * np.reciprocal(np.maximum(d, MIN_DISTANCE)), MAX_POTENTIAL)

    def start_repulsive_potential_linear(self, d):
        slope = MAX_POTENTIAL / START_LIN_INFLUENCE_DISTANCE
        return np.maximum(MAX_POTENTIAL - slope * d, 0)

    def potential_field_planning(self):
        self.potential_profile = []

        # search path
        d = np.hypot(self.sx - self.gx, self.sy - self.gy)
        # ix = round((sx - minx) / self.resolution)
        # iy = round((sy - miny) / self.resolution)
        # gix = round((gx - minx) / self.resolution)
        # giy = round((gy - miny) / self.resolution)
        ix = self.sx_id
        iy = self.sy_id

        if show_animation:
            draw_heatmap(self.potential)
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [
                exit(0) if event.key == 'escape' else None])
            plt.plot(self.sx_id, self.sy_id, "*k")
            plt.plot(self.gx_id, self.gy_id, "*m")

        rx, ry = [self.sx], [self.sy]
        previous_id = [(None, None)] * 3

        while d >= self.resolution:
            minp = float("inf")
            minix, miniy = -1, -1
            for i, _ in enumerate(self.motion_model):
                inx = int(ix + self.motion_model[i][0])
                iny = int(iy + self.motion_model[i][1])
                if inx >= len(self.potential) or iny >= len(self.potential[0]) or inx < 0 or iny < 0:
                    p = float("inf")  # outside area
                    print ("outside potential!")
                else:
                    p = self.potential[inx][iny]
                if minp > p:
                    minp = p
                    minix = inx
                    miniy = iny
            ix = minix
            iy = miniy
            xp = ix * self.resolution + self.minx
            yp = iy * self.resolution + self.miny
            d = np.hypot(self.gx - xp, self.gy - yp)
            rx.append(xp)
            ry.append(yp)
            self.potential_profile.append(minp)

            if ((None, None) not in previous_id and
                    (previous_id[0] == previous_id[1] or previous_id[1] == previous_id[2]
                        or previous_id[0] == previous_id[2])):
                print ("Oscillation detected!!!")
                print previous_id
                break

            # roll previous
            previous_id[0] = previous_id[1]
            previous_id[1] = previous_id[2]
            previous_id[2] = (ix, iy)

            if show_animation:
                plt.plot(ix, iy, ".r")
                plt.pause(0.001)

        print("Finish!!")

        return rx, ry

    def draw_potential_profile(self, potential_function, xlim):
        d = np.linspace(0, xlim, 500)
        p = np.array(map(potential_function, d))

        plt.figure()
        plt.plot(d, p)
        plt.ylim(-1, MAX_POTENTIAL+1.0)
        plt.grid()

    def draw_executed_potential_profile(self):
        plt.figure()
        plt.plot(self.potential_profile)
        plt.ylim(-1, MAX_POTENTIAL+1.0)
        plt.grid()


def main():
    print("potential_field_planning start")

    # define problem
    sx, sy, gx, gy, grid, ox, oy, resolution = define_scenario()

    print ("grid size: {}".format(grid.shape))
    print grid
    if ox is None:
        print "image could not be loaded"
        return False

    potential_planner = PotentialFieldPlannerGrid()
    if not potential_planner.set_problem(grid, sx, sy, gx, gy, ox, oy, resolution):
        return False

    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    potential_planner.set_motion_model(motion)

    # potential_planner.draw_potential_profile(potential_planner.repulsive_potential, 500)
    # potential_planner.draw_potential_profile(potential_planner.goal_attractive_potential, 500)
    # potential_planner.draw_potential_profile(potential_planner.start_repulsive_potential_inverse, 200)
    # potential_planner.draw_potential_profile(potential_planner.start_repulsive_potential_linear, 300)

    if show_animation or show_result:
        plt.figure()
        plt.grid(True)
        plt.axis("equal")
        plt.grid()

    # calc potential field
    potential_planner.calc_potential_field(obstacle_field=True, goal_field=True,
                                           start_field_inv=False, start_field_lin=True)

    # path generation
    rx, ry = potential_planner.potential_field_planning()
    print len(rx)
    print len(ry)

    if show_result:
        draw_heatmap(potential_planner.potential)
        plt.plot(potential_planner.sx_id, potential_planner.sy_id, "*k")
        plt.plot(potential_planner.gx_id, potential_planner.gy_id, "*m")
        for i in range(len(rx)):
            plt.plot(rx[i]+0.5, ry[i]+0.5, ".r")
        # for i in range(len(ox)):
        plt.plot(ox+0.5, oy+0.5, ".g")

        # potential_planner.draw_executed_potential_profile()

    if show_animation or show_result:
        plt.show()

    return True


if __name__ == '__main__':
    print(__file__ + " start!!")
    if main():
        print(__file__ + " Done!!")
    else:
        print(__file__ + " Failed!!")
