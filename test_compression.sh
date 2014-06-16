#!/bin/bash

wd=`pwd`
cd $SMART_GRID_SRC

#Compress
cat "${compression_data_dir}/outside_temp_preprocessed" | ./model_power_data/compress_best_tags/reduce.py > "${compression_data_dir}/outside_temp_meta_compressed.txt"
#python ./evaluate_visualize_model/process_date_time_errors_exe.py "${compression_data_dir}/outside_temp_compressed.txt"
python ./evaluate_visualize_model/process_meta_date_time_errors_exe.py "${compression_data_dir}/outside_temp_meta_compressed.txt"

#Evaluate
#cat "${compression_data_dir}/outside_temp_preprocessed" | ./model_power_data/evaluate_all_tags/reduce.py > "${compression_data_dir}/outside_temp_space_err"

cd $wd
