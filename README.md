SmartGridProject
================

A system for compressing smart grid data, and allowing queries on the compressed representation.  

# Getting Started #
Create a directory, let's say SmartGridProject.  

Clone this github directory into SmartGridProject/src

Create another directory SmartGridProject/data.  

In `setup.sh` replace `ion_username` with your ion-21-14 username

In SmartGridProject/src, run `source setup.sh`

# Load Data into HDFS #
```bash
hdfs -copyFromLocal /oasis/projects/nsf/csd181/melkherj/PI_data/PI_datasets/oledb_phase1 /user/melkherj/unprocessed_power_csvs
```
This won't work if the file `/user/melkherj/unprocessed_power_csvs` in hdfs already exists

# Preprocess data in HDFS #
Run: 
```bash
./stream.sh preprocess_power_data/stream_config.sh
```

# Create Seek Location Hash #
This allows for more efficient random access to the original sensor data on HDFS
    
```bash
./stream.sh model_power_data/get_tag_part_seek/stream_config.sh
hdfs -getmerge /user/melkherj/tag_part_seek "$SMART_GRID_DATA/summary_data/tag_part_seek"
````

# Run Compression Evaluation


# TODO #
Remove `melkherj` references in stream_config.sh files.  Make these refer to `ion_username` in `setup.sh`
