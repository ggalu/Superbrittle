# Spall experiment

In the spall experiment two elastic bodies collide. Body 1 is half as thick as body 2.
This leads to the superposition of tensile stress waves in body 2, which causes fracture.

Run this example like so from the spall example directory (`$` is the terminal prompt):

## 1. Generate the spall input mesh
The preprocessor can set up the spall example with an arbitrary mesh size, 0.01 in this case:

`$ python3 ../../2D/preprocess.py SPALL 1.0e-2`

This produces the input mesh file `mesh.pkl`

## 2. Run the spall experiment
`run.py` contains the simulation instructions and invokes the 2D Superbrittle solver.

`$ python3 run.py`

# Notes on benchmark results
This example is used to benchmark performance. A meshsize of 0.00125 is used, leading to approx. 2.1 million elements. Timings are as follows
* 420 seconds on Nvidia RTX 4060 (Sep 4 2024)
* 120 seconds on Nvidia L40 (Sep 4 2024)

An as-close as possible comparison was set up in LS-Dyna: linear triangular elements, 2D problem with 2D contact, but no conversion of failed elements to particles. The input file can found in the subfolder `LS-Dyna-comparison`. Reference timings for this simulation are as follows
* 38 Minutes on 32 cores, Intel Xeon Gold 6346 CPU @ 3.10GHz (fastest result, more cores increase wall time due to paralellization overhead)