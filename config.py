import os

dat = os.environ['SMART_GRID_DATA']
# The file containing the mapping from tags to files 
part_tag_path = dat+'/summary_data/phase1/tag_part_seek'
# The hdfs directory containing the oledb_tag_aggregate part files
hdfs_part_root_dir = '/user/melkherj/preprocessed_power_data'
# The directory containing space/error files
space_err_dir = dat + '/summary_data/phase1/space_errors'
