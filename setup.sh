# The file containing the mapping from tags to files 
export part_tag_path="${SMART_GRID_DATA}/summary_data/phase1/tag_part_seek"
# The hdfs directory containing the oledb_tag_aggregate part files
export hdfs_part_root_dir="/user/melkherj/preprocessed_power_data"
# The directory containing space/error files
export space_err_dir="${SMART_GRID_DATA}/summary_data/phase1/space_errors"
# Directory to store output data.  An example: the compressed pandas/pickle files
export data_dir="$(dirname $(pwd))/data"
