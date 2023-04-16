import numpy as np
import gtsam
import cv2
import torch
import time
import glob
import matplotlib.pyplot as plt
directory = "/root/ommo_ws/src/Loc-NeRF-plus-plus-OMMO/errors_log_horn_random_start"
def plot_error(error_type = 'Position'):
    """
    Plot the error change
    total_error_dataset: list of list. [per img in a dataset[per iteration in one img]]
    img number in one dataset is default = 5.
    """
    original_file = np.load(directory +"/"+ error_type + "_errors.npy")[0]
    adaptive_file = np.load(directory +"/"+ error_type + "_errors_adaptive.npy")[0]
    annealing_file = np.load(directory +"/"+ error_type + "_errors_refining.npy")[0]
    adapt_anneal = np.load(directory +"/"+ error_type + "_errors_adaptive_refining.npy")[0]
    iteration_num = len(original_file)
    print(adapt_anneal.shape)
    x = list(range(0, iteration_num))
    plt.plot(x, original_file, label = "loc-nerf")
    plt.plot(x, adaptive_file, label = "loc-nerf++")
    # x_annealing = list(range(0, len(annealing_file)))
    # annealing_file = annealing_file[:iteration_num]
    plt.plot(x, annealing_file, label = "loc-nerf w/ refining")
    plt.plot(x, adapt_anneal, label = "loc-nerf++ w/ refining")
    plt.xlabel("Iteration number")
    if error_type == 'Position':
        plt.ylabel("Position Error")
    elif error_type == 'Rotation':
        plt.ylabel("Rotation Error")
    plt.legend()
    plt.grid()
    plt.show()

def plot_particle_number():
    original_file = np.load(directory +"/num_particles.npy")[0]
    adaptive_file = np.load(directory +"/num_particles_adaptive.npy")[0]
    annealing_file = np.load(directory +"/num_particles_refining.npy")[0]
    adapt_anneal = np.load(directory +"/num_particles_adaptive_refining.npy")[0]
    
    iteration_num = len(original_file)
    x = list(range(0, iteration_num))
    plt.plot(x, original_file, label = "loc-nerf")
    plt.plot(x, adaptive_file, label = "loc-nerf++")
    # x_annealing = list(range(0, len(annealing_file)))
    # annealing_file = annealing_file[:iteration_num]
    plt.plot(x, annealing_file, label = "loc-nerf w/ refining")
    plt.plot(x, adapt_anneal, label = "loc-nerf++ w/ refining")
    plt.xlabel("Iteration number")
    plt.ylabel("Number of Particles")
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == "__main__":
    plot_error(error_type='Position')
    plot_error(error_type='Rotation')
    plot_particle_number()