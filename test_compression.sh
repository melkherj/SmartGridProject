#!/bin/bash

wd=`pwd`
cd $SMART_GRID_SRC

#~/part-00000
cat /home/melkherj/SmartGridProject/src/part-00099 | ./model_power_data/compress_all_tags/reduce.py > "${data_dir}/out.txt"
./evaluate_visualize_model/process_date_time_errors.py "${data_dir}/out.txt"

cd $wd
