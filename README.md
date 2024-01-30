# OCTSimulator
This MATLAB-based OCT simulator uses the single-scattering OCT Forward model to generate OCT tomograms given a sample's scattering potential.

The definition of the sample's scattering potential is based on Rayleigh scattering and Jonez-matrix formalism to model polarization-dependent backscattering. The modular implementation allows further implementation of different scattering models such as Mie scattering.

## Folder structure

**Scripts** contains a demo script that simulates a layered sample with birefringent and depolarizing samples.
*matlab* contains the modular functions required to generate the sample's scattering potential and emulate its corresponding OCT tomogram.

## Contact
Sebastian Ruiz-Lopera, Wellman Center for Photomedicine, Boston MA. srulo@mit.edu. 

## Version
**1.0** - Release version
