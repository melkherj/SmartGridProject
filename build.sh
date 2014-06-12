#!/bin/bash

cd evaluate_visualize_model
./build_pyx.sh
cd ../
cd piecewise
./build_pyx.sh
cd ../
