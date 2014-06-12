SmartGridProject
================

A system for compressing smart grid data, and allowing queries on the compressed representation.  

# Getting Started #
Create a directory, let's say SmartGridProject.  

Clone this github directory into SmartGridProject/src

Create another directory SmartGridProject/data.  

In `setup.sh` replace `ion_username` with your ion-21-14 username.  The directions below are for using the system with username melkherj on ion-21-14.  If you need to preprocess this data again from scratch, you'll need to change `melkherj` to your username.  You will also need to change `melkherj` references in `stream_config.sh` to your username.  

In SmartGridProject/src, run `source setup.sh`

# Load Data into HDFS #
Run 
```bash 
hdfs -ls /user/melkherj/unprocessed_power_csvs
```
This should report: `No such file or directory.`  If it doesn't, you should either: skip this step and use the existing data, or `hdfs -rmr` the existing data and 

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
