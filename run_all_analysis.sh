#!/bin/bash

### PRECOMPUTATIONS ###
# Preprocess data
./stream.sh preprocess_power_data/stream_config.sh
# Create a hash of seek locations in hdfs part files where tags are located
./stream.sh model_power_data/get_tag_part_seek/stream_config.sh
hdfs -getmerge /user/melkherj/tag_part_seek "$SMART_GRID_DATA/summary_data/phase1/tag_part_seek"

### ANALYSIS ###

# Run All compression models, and evaluate error/space
./stream.sh model_power_data/per_tag_analysis/stream_config.sh
space_err_results="$SMART_GRID_DATA/summary_data/phase1/space_errors/all_compressed.txt"
hdfs -getmerge /user/melkherj/all_compressed.txt $space_err_results
./evaluate_visualize_model/process_date_time_errors.py $space_err_results


