#!/bin/sh
find . -name "*.npz" -exec rm {} \;
find . -name "*.xdmf" -exec rm {} \;
find . -name "*.h5" -exec rm {} \;
find . -name "*.vtu" -exec rm {} \;
find . -name "*.png" -exec rm {} \;
find . -name "*.jpeg" -exec rm {} \;
find . -name "*.pyc" -exec rm {} \;
find . -name "d3plot*" -exec rm {} \;
find . -name "d3dump*" -exec rm {} \;
find . -name "d3hsp" -exec rm {} \;
find . -name "messag" -exec rm {} \;
find . -name "mesh.pkl" -exec rm {} \;
find . -name "all_data.npy" -exec rm {} \;
find . -name "profile_*" -exec rm {} \;
find . -name "tf.png" -exec rm {} \;
find . -name "reaction_forces.dat" -exec rm {} \;
find . -name "log.dat" -exec rm {} \;