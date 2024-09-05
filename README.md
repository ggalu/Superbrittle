# Superbrittle - a FEM code for brittle fracture

![til](./Superbrittle.webp)


## Superbrittle is a high performance code which runs on GPUs. It has these features:
* written in Python
* uses the Taichi extension for parallelization and to leverage the computing power of GPUs
* explicit time integration
* constant stress elements
* a pinball contact algorithm with surface normals
* elements are converted to particles upon failure to conserve momentum and interact with failed matter

## Current limitations
This software is at a very early development stage. Currently, only 2D simulations are supported, the material model is limited to corotational elasticity.

## Try it out
### Set up a Python environment 
The developers use Python==3.10 and the following packages, which are listed in `requirements.txt`
````
gmsh==4.13.1
matplotlib==3.8.4
meshio==5.3.5
numpy==2.1.1
progressbar33==2.4
scipy==1.14.1
taichi==1.7.1
````
### Run an example
Run the spallation example in `examples/01_spall2d`
