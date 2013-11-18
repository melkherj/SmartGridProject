#!/bin/bash

#shopt -s expand_aliases
#source /oasis/projects/nsf/csd181/hadoop/hadoop_shared/hadoop_bashrc.sh

if [ $# -ne 1 ]; then
    echo "usage: $0 <stream_config path>"
    exit -1
fi

source $1

echo $partitioner

if [ -z "$partitioner" ]
then
    echo "\$partitioner is not defined"
    # $partitioner is not defined, no partitioner
    had jar /oasis/projects/nsf/csd181/hadoop/hadoop_installation/contrib/streaming/hadoop-*streaming*.jar \
    -D mapred.map.tasks=$map_tasks \
    -D mapred.reduce.tasks=$reduce_tasks \
    -D mapred.text.key.partitioner.options=-k1 \
    -file $mapper -mapper $mapper \
    -file $reducer -reducer $reducer \
    -input $input \
    -output $output \
    -inputformat $inputformat
else
    echo "\$partitioner is defined"
    # $partitioner is defined
    had jar /oasis/projects/nsf/csd181/hadoop/hadoop_installation/contrib/streaming/hadoop-*streaming*.jar \
    -D mapred.map.tasks=$map_tasks \
    -D mapred.reduce.tasks=$reduce_tasks \
    -D stream.map.output.field.separator=$map_out_field_separator \
    -D stream.num.map.output.key.fields=$map_out_fields \
    -D map.output.key.field.separator=$map_out_key_separator \
    -D mapred.text.key.partitioner.options=-k1,$map_out_keys \
    -file $mapper -mapper $mapper \
    -file $reducer -reducer $reducer \
    -input $input \
    -output $output \
    -inputformat $inputformat \
    -partitioner $partitioner 
fi

# -jobconf num.key.fields.for.partition=$map_out_keys \
