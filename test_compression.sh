#!/bin/bash

wd=`pwd`
cd $SMART_GRID_SRC

cat "${compression_data_dir}/temp_tiny.txt" | ./model_power_data/compress_all_tags/reduce.py > "${compression_data_dir}/temp_tiny_compressed.txt"
python ./evaluate_visualize_model/process_date_time_errors_exe.py "${compression_data_dir}/temp_tiny_compressed.txt"

cd $wd
