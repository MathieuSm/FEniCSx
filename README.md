# FEniCSx Examples Repository

This repository provides various examples of finite element analyses using **FEniCSx**, the new version of the FEniCS finite element library. The repository covers different simulation scenarios and utility functions to support preprocessing, simulation, and post-processing tasks.

## Repository Structure

The repository is organized into the following directories:

### 1. UnitCube
This directory contains simulations of different material models applied to a unit cube. The implemented and planned models include:

- **Linear elasticity (Implemented)**: Basic linear elastic material analysis.
- **Hyperelasticity (Implemented)**: Advanced non-linear material behavior.
- **Post-yield behaviors (Ongoing):**
  - Perfect plasticity
  - Densification
- **Anisotropic material models (Implemented)**: Implementation of anisotropic material properties.

### 2. TensileExperiment
This directory provides a pipeline from a 3D scan to a full finite element simulation of a tensile experiment. The pipeline includes:

1. **Image Cropping**
2. **Filtering**
3. **Segmentation**
4. **Cleaning**
5. **Meshing**
6. **Simulation (Linear Elastic)**: Currently implemented only for the linear elastic case, with future plans to extend it to non-linear behaviors.

### 3. Homogenization
This module investigates homogenization techniques for:

- **3D Cubic Regions of Interest (ROI):** Computation of effective material properties from a 3D sample.
- **2D Segmented Images:** Different material properties assigned to different segments.

### 4. Utils
A collection of utility functions that facilitate various operations required throughout the simulations, including:

- **Time Printing:** Utilities for tracking computation times.
- **Image Processing:** Functions for preprocessing image data.
- **Image Reading:** Handling and loading images for analysis.
- **Tensor Calculus:** Operations related to tensor mathematics.
- **Meshing:** Utilities to generate and manipulate finite element meshes.

## Installation

To use this repository, install the provided conda environment on **WSL (Windows Subsystem for Linux)**:

```bash
conda env create -f FEniCS.yml
conda activate fenics
```

Ensure WSL is properly set up to run Linux applications on Windows.

## Usage

To write...

Refer to individual script documentation for further details.

## Future Work

- Complete implementation of postyield behaviors (plastic, densification).
- Expand the tensile experiment pipeline for nonlinear material properties.

## Contributing

To write as well...
<!---
Contributions are welcome! Feel free to submit pull requests or raise issues for discussion.
-->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

