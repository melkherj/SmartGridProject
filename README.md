SmartGridProject
================

A system for compressing smart grid data, and allowing queries on the compressed representation.  

# Getting Started #
Create a directory, let's say SmartGridProject.  

Clone this github directory into SmartGridProject/src

Create another directory SmartGridProject/data.  

In SmartGridProject/src, run `source setup.sh`

# Load Data into HDFS #
Run: `./stream.sh preprocess_power_data/stream_config.sh`

# Preprocess data in HDFS #
Change your ion_username
