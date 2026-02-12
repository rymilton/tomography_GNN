# Overview
This repository uses a modified version of MLPF (Machine Learning Particle Flow) to do muon tomography. The original MLPF paper is found [here](https://link.springer.com/article/10.1140/epjc/s10052-021-09158-w).

We use MLPF to take in simulated muons passing through a voxel grid of varying densities to predict the densities. The workflow is as follows:
1. Convert the local, reconstructed theta and phi values from a detector to the global coordinate system.
2. Get the positions of each muon above and below the voxel 3D grid.
3. Run Siddon's algorithm to get the voxels each muon passes through and the path length of the muons in those voxels. The implementation of Siddon's algorithm can be found in [this repository](https://github.com/rymilton/raytrace/).
4. Preprocess the data by re-scaling the input features to be in the range [0,1].

MLPF then forms a graph of a group of muons, and for each muon node, outputs a set of density predictions for each voxel that the muon passes through. This number is multiplied by the path length of the muon in that voxel. If a voxel has multiple muons passing through it, the contributions from each muon (model prediction*path length) are summed. The training label is the true densities of all the voxels that received hits. The data can include geometries with and without objects present.

During inference, the model predictions per voxels are averaged across all batches.
## Setting up + environment
### Cloning
To clone this repository, use `git clone --recurse-submodules https://github.com/rymilton/tomography_GNN.git`. You need the `--recurse-submodules` to get the raytrace repository.

### Environment
#### CUDA
CUDA is needed to install this software. CUDA needs to be in your PATH and LD_LIBRARY_PATH variables. To do this for example on the UCR GPU machine, you can add the following to your ~/.bashrc:
```
export PATH=$PATH:/usr/local/cuda-11.8/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
```

#### Python environment
To set up your Python virtual environment simply do the following after cloning the repository:
```
python3 -m venv tomography_GNN_venv
source tomography_GNN_venv/bin/activate
pip install -r requirements.txt
```

#### Installation
To install the raytracing repository, you need to do the following: `cd raytrace & pip install . & cd ..`

### Scripts for preparing data
As mentioned in the Overview section, there are four steps to preparing the data for training. There are different scripts to do these steps, all contained in `scripts`. The `prepare_data.sh` script carries out all of the steps in one convenient script. It is highly recommended to use this script.

Below, each set of steps is covered in detail:

1. Convert the local, reconstructed theta and phi values from a detector to the global coordinate system.
2. Get the positions of each muon above and below the voxel 3D grid.

These two steps are handled by `scripts/transform_local_to_global.py`. The input muon data is in the local coordinate system of the detector that detected the particle. In general, the detector could be rotated with respect to the global coordinate system. For every muon, we get the positions above and below the voxels in z. To correctly calculate this, we need the positions and the angles of the detectors with respect to the global coordinate system. These are set in `scripts/data_config.yaml`, along with the z locations you want the muon positions at. There are some default detector values in this file, but they can be adjusted based on the detector index in the data.

3. Run Siddon's algorithm to get the voxels each muon passes through and the path length of the muons in those voxels.

Now that we have the intial and final positions of the muons, we can run them through Siddon's algorithm. This is done in `scripts/get_voxels.py`. This script requires the positions of the voxels and the shape of the voxels. To get the shape, the voxel density file is used. The file produced from `scripts/transform_local_to_global.py` should be used as input here. Some plots can be drawn to illustrate the voxel identification. This can be done using the `--draw_plots` and `--plot_directory` flags. Some example plots are shown below.
<img width="300" height="300" alt="muon_ray_000_withobject" src="https://github.com/user-attachments/assets/25328575-90d8-476d-805d-6d04d8538d46" />
<img width="300" height="300" alt="muon_ray_001_withobject" src="https://github.com/user-attachments/assets/4ab3cfca-da65-4f5c-a558-2a951a20c0ce" />

4. Preprocess the data by re-scaling the input features to be in the range [0,1].

Finally, the input features of the data is re-scaled to be in the range [0,1] using the script `scripts/preprocess_data.py`. This script can also split files into train/test. To enable this, set `TRAIN_TEST_SPLIT: True` and set the `TEST_FRACTION` in `scripts/data_config.yaml`. This script adds the voxel densities to the final .pkl file, so the voxel file is needed as input with the `--input_voxel_densities_file` flag. If you're working with a case with no object, use the `--no_object` flag to set the densities of all the voxels to 0. The shape of the voxels will be the same as the voxels in `--input_voxel_densities_file`. Note an object ID will be added to the data, with 0 representing no object and 1 being an object is present.

The files from `scripts/preprocess_data.py` are the final files you'll need to train the model. Again, it's recommended to use the `scripts/prepare_data.sh` script to run all of these scripts for you.
