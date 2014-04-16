#!/bin/bash

echo ""
echo "### Building ###"
python setup.py build_ext --inplace
echo ""

# Clean
rm -rf build
rm process_just_errors.c
rm process_date_time_errors.c
mv src/evaluate_visualize_model/*.so .
rmdir src/evaluate_visualize_model
rmdir src
