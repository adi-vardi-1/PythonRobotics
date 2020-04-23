from utilities import draw_heatmap, load_image

import numpy as np
import matplotlib.pyplot as plt
import time

# Parameters
AREA_WIDTH = 30.0  # potential area width [m]
KP = 1.0            # attractive potential gain
ETA = 1000.0        # repulsive potential gain
# ETA2 = 500.0
MIN_DISTANCE = 0.1
MIN_OBSTACLE_DISTANCE = 5.0
DESIRED_DISTANCE = 10
MAX_OBSTACLE_DISTANCE = 50
MAX_POTENTIAL = 5

show_animation = True


class PotentialFieldPlanner(object):
    def set_problem(self, sx, sy, gx, gy, ox, oy, resolution, use_goal_field, use_start_field, use_obstacle_field):
        self.use_goal_field = use_goal_field
        self.use_start_field = use_start_field
        self.use_obstacle_field = use_obstacle_field
        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy
        self.ox = ox
        self.oy = oy
        self.resolution = resolution

        # transform to indices
        self.minx = min(min(ox), sx, gx) - AREA_WIDTH / 2.0
        self.miny = min(min(oy), sy, gy) - AREA_WIDTH / 2.0
        self.maxx = max(max(ox), sx, gx) + AREA_WIDTH / 2.0
        self.maxy = max(max(oy), sy, gy) + AREA_WIDTH / 2.0

        self.xw = int(round((self.maxx - self.minx) / resolution))
        self.yw = int(round((self.maxy - self.miny) / resolution))

        self.sx_id = (sx - self.minx) / resolution
        self.sy_id = (sy - self.miny) / resolution
        self.gx_id = (gx - self.minx) / resolution
        self.gy_id = (gy - self.miny) / resolution
        self.ox_id = (ox - self.minx) / resolution
        self.oy_id = (oy - self.miny) / resolution

        print ("start id: {} , {}".format(self.sx_id, self.sy_id))
        print ("goal id: {} , {}".format(self.gx_id, self.gy_id))
        print ("ox id: {}".format(self.ox_id))
        print ("oy id: {}".format(self.oy_id))
        print ("minx : {}".format(self.minx))
        print ("miny : {}".format(self.miny))
        print ("maxx : {}".format(self.maxx))
        print ("miny : {}".format(self.miny))

    def set_motion_model(self, motion_model):
        self.motion_model = motion_model

    def calc_potential_field(self):
        start = time.time()
        self.potential = np.zeros((self.xw, self.yw))
        print ("potential size: {}".format(self.potential.shape))

        # calc each potential
        for ix in range(self.xw):
            x = ix * self.resolution + self.minx
            for iy in range(self.yw):
                y = iy * self.resolution + self.miny

                ug = (self.calc_attractive_potential(x, y, self.gx, self.gy) if self.use_goal_field else 0)
                uo = (self.calc_repulsive_potential(x, y, self.ox, self.oy) if self.use_start_field else 0)
                us = (self.calc_start_repulsive_potential(x, y, self.sx, self.sy) if self.use_obstacle_field else 0)
                uf = ug + uo + us
                self.potential[ix][iy] = uf

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

    def calc_attractive_potential(self, x, y, gx, gy):
        return 0.5 * KP * np.hypot(x - gx, y - gy)

    def calc_start_repulsive_potential(self, x, y, sx, sy):
        d = np.hypot(x - sx, y - sy)
        if d <= MIN_DISTANCE:
            d = MIN_DISTANCE
        return 0.5 * KP * (1/d)

    def repulsive_potential(self, distance):
        if distance <= MIN_OBSTACLE_DISTANCE:
            p = MAX_POTENTIAL
        # elif distance <= DESIRED_DISTANCE:        # same as next
        #     p = 0.5 * ETA * (1.0 / distance - 1.0 / DESIRED_DISTANCE) ** 2
        elif distance <= MAX_OBSTACLE_DISTANCE:
            p = 0.5 * ETA * (1.0 / distance - 1.0 / DESIRED_DISTANCE) ** 2
            # p *= (distance - MAX_OBSTACLE_DISTANCE)
        else:
            p = 0

        # calc repulsive potential, only for points which should be in collision with robot
        # original potential
        # if dq <= MIN_OBSTACLE_DISTANCE:
        #     if dq <= MIN_DISTANCE:
        #         dq = MIN_DISTANCE

        #     return 0.5 * ETA * (1.0 / dq - 1.0 / MIN_OBSTACLE_DISTANCE) ** 2
        # else:
        #     return 0.0

        return min(p, MAX_POTENTIAL)

    def calc_repulsive_potential(self, x, y, ox, oy):
        # search nearest obstacle
        minid = -1
        dmin = float("inf")
        for i, _ in enumerate(ox):
            d = np.hypot(x - ox[i], y - oy[i])
            if dmin >= d:
                dmin = d
                minid = i

        # calc distance to obstacle
        dq = np.hypot(x - ox[minid], y - oy[minid])

        return self.repulsive_potential(dq)

    def plan(self):
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
                    plt.pause(0.01)

            print("Goal!!")

            return rx, ry

    def draw_potential_profile(self):
        d = np.linspace(0, 2*MAX_OBSTACLE_DISTANCE, 500)
        p = np.array(map(self.repulsive_potential, d))

        fig = plt.figure()
        plt.plot(d, p)
        plt.ylim(-5, MAX_POTENTIAL)


def main():
    print("potential_field_planning start")

    # define problem
    sx = 40.0  # start x position [m]
    sy = 0.0  # start y positon [m]
    gx = 40.0  # goal x position [m]
    gy = 70.0  # goal y position [m]
    resolution = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    if robot_radius > DESIRED_DISTANCE:
        print "robot_radius > DESIRED_DISTANCE"
        return

    grid, ox, oy = load_image('./map.png')
    if ox is None:
        print "image could not be loaded"
        return

    potential_planner = PotentialFieldPlanner()
    potential_planner.set_problem(sx, sy, gx, gy, ox, oy, resolution, False, True, True)

    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
            # [0, -1],
            # [-1, -1],
              [-1, 1],
            # [1, -1],
              [1, 1]]

    potential_planner.set_motion_model(motion)

    potential_planner.draw_potential_profile()

    if show_animation:
        fig = plt.figure()
        plt.grid(True)
        plt.axis("equal")

    # calc potential field
    potential_planner.calc_potential_field()

    # path generation
    _, _ = potential_planner.plan()

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
