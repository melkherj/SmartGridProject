#Environment variables used for running the hadoop streaming command
mapper="$SMART_GRID_SRC/model_power_data/get_tag_part_seek/map.py"
reducer=/bin/cat
map_tasks=50 #make sure input file size is smaller than mapred.min.split.size
reduce_tasks=50
input=/user/melkherj/preprocessed_power_data
output=/user/melkherj/tag_part_seek
inputformat=TextInputFormat
# This is how you handle keying on values
partitioner=org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner
## How many fields total
map_out_fields=3
## Tell hadoop how it knows how to parse map output so it can key on it
map_out_field_separator=^
## You can key on your map fields separately
map_out_key_separator=^
## Number of map output keys for grouping on, the rest are sorted
map_out_keys=0
