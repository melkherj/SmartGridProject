#!/bin/bash

echo ""
echo "### Building ###"
python setup.py build_ext --inplace
echo ""

# Clean
rm -rf build
#rm piecewise.c
mv src/piecewise/*.so .
rmdir src/piecewise
rmdir src
