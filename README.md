# Loc-NeRF-plus

An Enhanced Monte Carlo Localization using Neural Radiance Fields. 

## Coordinate Frames
To be consistent throughout the code and in the yaml files, we define coordinates using the camera frame commonly used for NeRF (x right, y up, z inward from the perspective of the camera) unless stated otherwise. Coordinates are FROM Camera TO World unless otherwise stated. Note this is not the same as the more common camera frame used in robotics (x right, y down, z outward).

## Publications
Reference:
[Loc-NeRF: Monte Carlo Localization using Neural Radiance Fields](https://arxiv.org/abs/2209.09050)

# 1. Installation
We suggest to use Docker to run our algorithm. Pull a Docker image with ROS Noetic from Docker Hub (https://hub.docker.com/).
```bash
# Pull a docker image
sudo docker pull osrf/ros:noetic-desktop-full

# Run a docker container
docker run --gpus all -it -v /mnt:/mnt --device=/dev/dri --group-add video --volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" --name=ros_noetic_loc_nerf osrf/ros:noetic-desktop-full /bin/bash

# Install and run tmux
sudo apt-get update
sudo apt install tmux
tmux new -s new_window
```

# 2. Loc-NeRF installation
```bash
# Setup catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin init

# Clone repo
cd ~/catkin_ws/src
git clone https://github.com/JunShao0104/Loc-NeRF-plus

# Compile code
catkin build

# Source workspace
source ~/catkin_ws/devel/setup.bash

# Install dependencies:
cd ~/catkin_ws/src/Loc-NeRF-plus
pip install -r requirements.txt

# Update numpy with version >=1.19.5 and <1.27.0
pip uninstall numpy
pip install numpy==1.19.5

# Update tkinter
sudo apt-get install python3-tk
sudo apt-get install tk-dev
```

## Starting Loc-NeRF-plus
We will use ROS and rviz as a structure for running Loc-NeRF and for visualizing performance. 
As a general good practice, remember to source your workspace for each terminal you use.

  0. Open a new terminal in the host machine and run:
  ```bash
  xhost +
  ```

  1. Open a new terminal and run: `roscore`

  2. In another terminal, launch Loc-NeRF:
  ```bash
  roslaunch locnerf navigate.launch parameter_file:=<param_file.yaml>
  ```

  3. In another terminal, launch rviz for visualization:
  ```bash
  rviz -d $(rospack find locnerf)/rviz/rviz.rviz
  ```

  4. If you are not running with a rosbag, i.e. you are using LLFF data, then Loc-NeRF should start and you should be set. If you are using a rosbag, continue to the next steps.

  5. In another terminal launch VIO.

  6. Finally, in another terminal, play your rosbag:
  ```bash
  rosbag play /PATH/TO/ROSBAG
  ```

## Provided config files
We provide a few yaml files in /cfg to get you started. 

```llff.yaml``` runs Loc-NeRF on the LLFF dataset as described in the paper.

```llff_global.yaml``` runs Loc-NeRF on the LLFF dataset with a wider spread of particles to test the ability to perform global localization as described in the paper.

```llff_adaptive.yaml``` runs Loc-NeRF++ on the LLFF dataset and remember to modify the nav_node.py to enable the adaptve version.

```llff_global_adaptive.yaml``` runs Loc-NeRF++ on the LLFF dataset with a wider spread of particles to test the ability to perform global localization and also remember to modify the nav_node.py to enable the adaptive version.

# 3. Usage
The fastest way to start running Loc-NeRF is to download LLFF data with pre-trained NeRF weights. We also test our algorithm on OMMO dataset (Outdoor Multi-Modal Dataset).

## Using LLFF data

Download LLFF images and pretrained NeRF-Pytorch weights from [NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch). If you download our fork of iNeRF here: 
[iNeRF](https://github.com/Dominic101/inerf) then the configs and ckpts folder will already be setup correctly with the pre-trained weights, and you just need to add the data folder from NeRF-Pytorch.

Place data using the following structure:

```
├── configs   
│   ├── ...
├── ckpts                                                                                                       
│   │   ├── fern
|   |   |   └── fern.tar                                                                                                                     
│   │   ├── fortress
|   |   |   └── fortress.tar                                                                                   
│   │   ├── horns
|   |   |   └── horns.tar   
│   │   ├── room
|   |   |   └── room.tar   
|   |   └── ...                                                                                 
                                                                                            
├── data                                                                                                                                                                                                       
│   ├── nerf_llff_data                                                                                                  
│   │   └── fern  # downloaded llff dataset                                                                                                                         
│   │   └── fortress  # downloaded llff dataset                                                                                  
│   │   └── horns   # downloaded llff dataset
|   |   └── room   # downloaded llff dataset
|   |   └── ...
```

After updating your yaml file ```llff.yaml``` with the directory where you placed the data and any other params you want to change, you are ready to run Loc-NeRF! By default, Loc-NeRF will estimate the camera pose of 5 random images from each of fern, fortress, horns, room (20 images in total). You can use rviz to provide real-time visualization of the ground truth pose and the particles.

  # Third-party code:
 Parts of this code were based on [this pytorch implementation of iNeRF](https://github.com/salykovaa/inerf) and [NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch).

 ```
 NeRF-Pytorch:
 
 @misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
 ```
