from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
     
class anim_array():
    def __init__(self, grid_sys, data_list):
        self.grid_sys = grid_sys
        self.data_list = data_list
        self.figure = None
        self.ax = None
        self.cax = None
        #self.create_graph()
        self.homemade_anim()
    
    def homemade_anim(self):
        self.figure, self.ax = plt.subplots(1,1)
        for i in range(len(self.data_list)):
            if i % 5 == 0:
                i = self.ax.pcolormesh(self.grid_sys.x_level[0], self.grid_sys.x_level[1], self.data_list[i].T, shading='gouraud', cmap = 'plasma')

            plt.pause(0.1)
        plt.show()
            