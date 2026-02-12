# Overview
This repository uses a modified version of MLPF (Machine Learning Particle Flow) to do muon tomography. The original MLPF paper is found [here](https://link.springer.com/article/10.1140/epjc/s10052-021-09158-w).

We use MLPF to take in simulated muons passing through a voxel grid of varying densities to predict the densities. The workflow is as follows:
1. Convert the local, reconstructed theta and phi values from a detector to the global coordinate system.
2. Get the positions of each muon above and below the voxel 3D grid.
3. Run Siddon's algorithm to get the voxels each muon passes through and the path length of the muons in those voxels. The implementation of Siddon's algorithm can be found in [this repository](https://github.com/rymilton/raytrace/).
4. Preprocess the data by re-scaling the input features to be in the range [0,1].

MLPF then forms a graph of a group of muons, and for each muon node, outputs a set of density predictions for each voxel that the muon passes through. This number is multiplied by the path length of the muon in that voxel. If a voxel has multiple muons passing through it, the contributions from each muon (model prediction*path length) are summed. The training label is the true densities of all the voxels that received hits. The data can include geometries with and without objects present.

During inference, the model predictions per voxels are averaged across all batches.
