"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

This is the simple code for path planning class

"""



import math

import matplotlib.pyplot as plt

show_animation = 0


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr, fc_x, fc_y, tc_x, tc_y,rc_x,rc_y):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution # get resolution of the grid
        self.rr = rr # robot radis
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model() # motion model for grid search expansion
        self.calc_obstacle_map(ox, oy)

        self.fc_x = fc_x
        self.fc_y = fc_y
        self.tc_x = tc_x
        self.tc_y = tc_y
        self.rc_x = rc_x
        self.rc_y = rc_y
        

        self.Delta_C1 = 0.3 # cost intensive area 1 modifier
        self.Delta_C2 = 0.15 # cost intensive area 2 modifier
        self.Delta_C3 = -0.05 # cost intensive area 2 modifier

        self.costPerGrid = 1 


    class Node: # definition of a sinle node
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy,task2_i):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x), # calculate the index based on given position
                               self.calc_xy_index(sy, self.min_y), 0.0, -1) # set cost zero, set parent index -1
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x), # calculate the index based on given position
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict() # open_set: node not been tranversed yet. closed_set: node have been tranversed already
        open_set[self.calc_grid_index(start_node)] = start_node # node index is the grid index

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(self, goal_node,
                                                                     open_set[
                                                                         o])) # g(n) and h(n): calculate the distance between the goal node and openset
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            # reaching goal
            if current.x == goal_node.x and current.y == goal_node.y:
                print("-----------------------------------------------------------")
                print("For blue area from " + str(task2_i) + " to " + str(task2_i + 5) +", Trip time required -> ",current.cost )
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                if current.cost <= lowest_task2[2]:
                    lowest_task2[0]=task2_i
                    lowest_task2[1]=task2_i+5
                    lowest_task2[2]=current.cost
                print("lowest cost area for task 2: "+str(lowest_task2[0])+" to "+str(lowest_task2[1])+" with cost :"+str(lowest_task2[2]))
                print("-----------------------------------------------------------")
                #result_print(current.cost)
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # print(len(closed_set))

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion): # tranverse the motion matrix
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2] * self.costPerGrid, c_id)
                
                ## add more cost in cost intensive area 1 / time consuming area
                if self.calc_grid_position(node.x, self.min_x) in self.tc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.tc_y:
                        # print("cost intensive area!!")
                        node.cost = node.cost + self.Delta_C1 * self.motion[i][2]
                
                # add more cost in cost intensive area 2 / fuel consuming area
                if self.calc_grid_position(node.x, self.min_x) in self.fc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.fc_y:
                        # print("cost intensive area!!")
                        node.cost = node.cost + self.Delta_C2 * self.motion[i][2]
                    # print()
                
                # add more cost in cost intensive area 3 / task 2 cost reducing area
                if self.calc_grid_position(node.x, self.min_x) in self.rc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.rc_y:
                        # print("cost intensive area!!")
                        node.cost = node.cost + self.Delta_C3 * self.motion[i][2]
                    # print()    
                     
                
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        # print(len(closed_set))
        # print(len(open_set))

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)] # save the goal node as the first point
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(self, n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        d = d * self.costPerGrid
        return d
    
    def calc_heuristic_maldis(n1, n2):
        w = 1.0  # weight of heuristic
        dx = w * math.abs(n1.x - n2.x)
        dy = w *math.abs(n1.y - n2.y)
        return dx + dy

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x) 

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        #print("min_x:", self.min_x)
        #print("min_y:", self.min_y)
        #print("max_x:", self.max_x)
        #print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        #print("x_width:", self.x_width)
        #print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)] # allocate memory
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x) # grid position calculation (x,y)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy): # Python’s zip() function creates an iterator that will aggregate elements from two or more iterables. 
                    d = math.hypot(iox - x, ioy - y) # The math. hypot() method finds the Euclidean norm
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True # the griid is is occupied by the obstacle
                        break

    @staticmethod
    def get_motion_model(): # the cost of the surrounding 8 points
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

def result_print(time):
    """
    Output the cost for each type of aircraft as well as the lowest one
    """
    # Ask for scenario No. 
    # If invalid scenario no. is inputed, ask again
    loop = 1
    while loop :
        test_input = int(input("Please select the Scenario (1/2/3): "))-1
        print("-----------------------------------------------------------")
        if test_input <= 2 and test_input >= 0 :
            loop = 0
    
    # Name(0)    Fuel(1)    Pax(2)     Time_L(3)  Time_M(4)  Time_H(5)  Fix_C(6)
    # 321neo     54         200        10         15         20         1800      
    # 339        84         300        15         21         27         2000   
    # 359        90         350        20         27         34         2500
    #
    # Set up the Cost Speccification Array
    aircraft_stat = [
    ["A321neo",54,200,10,15,20,1800],
    ["A339",84,300,15,21,27,2000],
    ["A359",90,350,20,27,34,2500]
    ] 

    #        Pax(0)     Limit(1)   Time(2)    Fuel(3)
    # Scene1 3300       13         M          0.85
    # Scene2 1500       7*4        H          0.96   
    # Scene3 2250       25         L          0.78
    #

    # Set up the Scenario Array
    scene_stat = [
    [3300,13,"M",0.85],
    [1500,7*4,"H",0.96],
    [2250,25,"L",0.78]
    ]
    
    # Set up variable flight_count for counting flight no. needed
    flight_count=0
    # Set up array to store aircraft typr w/ lowest cost
    flight_min_cost = ["none",2**31-1]
    # Get Passenger target no. from scenario array
    pax_target = scene_stat[test_input][0]
    
    # Calculate cost for each flight Order: A321 --> A339 --> A359
    for j in range(0,3):
        flight_count = math.ceil(scene_stat[test_input][0]/aircraft_stat[j][2])
        if flight_count <= scene_stat[test_input][1]:
            cost = scene_stat[test_input][3] * aircraft_stat[j][1] * time + aircraft_stat[j][6]
            if scene_stat[test_input][2] == "L":
                cost= cost + time * aircraft_stat[j][3]
            elif scene_stat[test_input][2] == "M":
                cost= cost + time * aircraft_stat[j][4]
            elif scene_stat[test_input][2] == "H":
                cost= cost + time * aircraft_stat[j][5]
            total_cost = cost*flight_count
            print("Single trip cost of " + aircraft_stat[j][0] + " in scene " + str(test_input+1) + ": "+ str(cost))
            print("Total trip cost of " + aircraft_stat[j][0] + " in scene " + str(test_input+1) + ": "+ str(total_cost))
            print("-----------------------------------------------------------")
            # If total cost < stored lowest cost (i.e. lowest cost by now) then update the array
            if total_cost < flight_min_cost[1]:
                flight_min_cost = [aircraft_stat[j][0],total_cost]
        else:
            print( aircraft_stat[j][0] + " is not viable")
            print("-----------------------------------------------------------")
    print("In scene " + str(test_input+1) + ", the operation cost of " + flight_min_cost[0] + " is the lowest, with the cost of " + str(flight_min_cost[1]))
    print("-----------------------------------------------------------")


def main(task2_i):
    print(__file__ + " start the A star algorithm demo !!") # print simple notes

    # start and goal position
    sx = 50.0  # [m]
    sy = 50.0  # [m]
    gx = 0.0  # [m]
    gy = 0.0  # [m]
    grid_size = 1.0  # [m]
    robot_radius = 1.0  # [m]

    # set obstacle positions for group 8
    # ox, oy = [], []
    # for i in range(-10, 60): # draw the button border 
    #     ox.append(i)
    #     oy.append(-10.0)
    # for i in range(-10, 60):
    #     ox.append(60.0)
    #     oy.append(i)
    # for i in range(-10, 61):
    #     ox.append(i)
    #     oy.append(60.0)
    # for i in range(-10, 61):
    #     ox.append(-10.0)
    #     oy.append(i)
    # for i in range(-10, 40):
    #     ox.append(20.0)
    #     oy.append(i)
    # for i in range(0, 40):
    #     ox.append(40.0)
    #     oy.append(60.0 - i)


    # set obstacle positions for group 9
    ox, oy = [], []
    for i in range(-10, 60): # draw the bottom border 
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60): # draw the right border
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 60): # draw the top border
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 60): # draw the left border
        ox.append(-10.0)
        oy.append(i)

    for i in range(-10, 20): # draw the free border
        ox.append(i)
        oy.append(int(-2/3 * i + 70/3))

    for i in range(30, 40):
        ox.append(i)
        oy.append(int(-2 * i + 120))
    
    for i in range(30, 40):
        ox.append(i)
        oy.append(-1 * i + 60)
    
    # for i in range(40, 45): # draw the button border 
    #     ox.append(i)
    #     oy.append(30.0)


    # set cost intesive area 1
    tc_x, tc_y = [], []
    for i in range(10, 30):
        for j in range(30, 50):
            tc_x.append(i)
            tc_y.append(j)
    
    # set cost intesive area 2
    fc_x, fc_y = [], []
    for i in range(30, 60):
        for j in range(0, 20):
            fc_x.append(i)
            fc_y.append(j)
    
    # set cost reducing area (Task 2)
    rc_x, rc_y = [], []
    for i in range(-10, 60):
        for j in range(task2_i, task2_i + 5):
            rc_x.append(i)
            rc_y.append(j)
    

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k") # plot the obstacle
        plt.plot(sx, sy, "og") # plot the start position 
        plt.plot(gx, gy, "xb") # plot the end position
        
        plt.plot(fc_x, fc_y, "oy") # plot the cost intensive area 1
        plt.plot(tc_x, tc_y, "or") # plot the cost intensive area 2
        plt.plot(rc_x, rc_y, "ob") # plot the cost reducing area Task 2

        plt.grid(True) # plot the grid to the plot panel
        plt.axis("equal") # set the same resolution for x and y axis 

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius, fc_x, fc_y, tc_x, tc_y,rc_x,rc_y)
    rx, ry = a_star.planning(sx, sy, gx, gy,task2_i)

    #test_input = int(input("Scene? : "))
    #print("test scene"+ str(test_input))
    #result_print()

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r") # show the route 
        plt.pause(0.001) # pause 0.001 seconds
        plt.show() # show the plot


if __name__ == '__main__':
    lowest_task2 = [99,99,2**31]
    for task2_i in range(-10,56):
        main(task2_i)
