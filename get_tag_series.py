import sys, os
sys.path.append(os.environ['SMART_GRID_SRC'])
import numpy as np
import pandas as pd
from serialize_tag_date import decode_line
import datetime
import subprocess

def get_part_tag_dict():
    ''' Load a dictionary mapping tag->(part,seek) from a file '''
    part_tag_dict = {}
    with open(os.environ['part_tag_path'], 'r') as f:
        for line in f:
            part, tag, seek = line.split('^')
            part_tag_dict[tag] = (part, int(seek))
    return part_tag_dict

part_tag_dict = get_part_tag_dict()

def hdfs_file_stream(path):
    ''' Given a path to a file in hdfs, return a stream reading from this file '''
    hdfs = os.environ['hdfs'].split()
    cat = subprocess.Popen(hdfs + ["-cat",path], stdout=subprocess.PIPE)
    return cat.stdout
 
def get_next_tag_series(f, first_line=None):
    ''' Given a file handle, read lines until a new tag appears 
        Return a DataFrame, and the first line of the next tag series
        '''
    tag_date_series = []
    tags = []
    dates = []
    current_tag = None
    next_line = None
    def generator_prepend(gen, prepend):
        ''' Return a generator that first returns prepend, then 
            runs generator 'gen' '''
        yield prepend
        for value in gen:
            yield value
    if not first_line is None:
        f = generator_prepend(f, first_line)
    for line in f:
        tag, date, series = decode_line(line)
        if (current_tag != tag) and (not current_tag is None):
            next_line = line #Tag has changed.  Terminate
            break
        else:
            series = np.reshape(series, (1,1440))
            tag_date_series.append(series) #We've found the line
            tags.append(tag)
            dates.append(date)
        current_tag = tag
    if len(tag_date_series) == 0:
        return None, None
    series = np.concatenate(tag_date_series, axis=0)
    # Create a data frame with index tag/date, and values given by the series
    df = pd.DataFrame(series)
    df['tag'] = tags
    df['date'] = [np.datetime64(datetime.date(*d)) for d in dates]
    df = df.set_index(['tag','date'])
    return df, next_line
   
def get_tag_series(tag):
    ''' Get the series and dates corresponding to the given tag'''
    part, seek = part_tag_dict[tag]
    hdfs_path = os.environ['hdfs_part_root_dir']+'/'+part
    f = hdfs_file_stream(hdfs_path)
    f.read(seek)
    df, _ = get_next_tag_series(f)
    return df
