ion_username="melkherj"
ion="ion-21-14.sdsc.edu"

# We're not on the io-node on gordon
if [ `hostname` != $ion ]; then
    # For accessing hdfs
    export sshion="ssh ${ion_username}@${ion}"
    export SMART_GRID_DATA="$(pwd)/../data"
fi

# Smart Grid Compression source directory
export SMART_GRID_SRC=`pwd`
# The hdfs directory containing the oledb_tag_aggregate part files
export hdfs_part_root_dir="/user/melkherj/preprocessed_power_data"
# Directory to store output data.  An example: the compressed pandas/pickle files
export compression_data_dir="${SMART_GRID_DATA}/compression"
# The file containing the mapping from tags to files 
export part_tag_path="${SMART_GRID_DATA}/summary_data/tag_part_seek"
# The directory containing space/error files
export space_err_dir="${SMART_GRID_DATA}/summary_data/space_errors"
