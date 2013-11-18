#Environment variables used for running the hadoop streaming command
mapper="$SMART_GRID_SRC/preprocess_power_data/map.py"
reducer=/bin/cat
map_tasks=1000
reduce_tasks=1000
input=/user/melkherj/unprocessed_power_csvs
output=/user/melkherj/preprocessed_power_data
inputformat=TextInputFormat
# This is how you handle keying on values
partitioner=org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner
## How many fields total
map_out_fields=3
## Tell hadoop how it knows how to parse map output so it can key on it
map_out_field_separator=^
## You can key on your map fields separately
map_out_key_separator=^
## Numer of map output keys for grouping on, the rest are sorted
map_out_keys=1
