#!/bin/bash

wd=`pwd`
cd $SMART_GRID_SRC

#~/part-00000
cat "${compression_data_dir}/part-00099" | ./model_power_data/compress_all_tags/reduce.py > "${compression_data_dir}/out.txt"
./evaluate_visualize_model/process_date_time_errors.py "${compression_data_dir}/out.txt"

cd $wd
