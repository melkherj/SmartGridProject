#!/usr/bin/env python

# A mapper for counting the number of each kind of tag in the dataset

import sys, os
sys.path.append(os.environ['SMART_GRID_SRC'])
from serialize_tag_date import encode_line
import datetime
import numpy as np

if 'map_input_file' in os.environ:
    filepath = os.environ['map_input_file']
    filename = os.path.split(filepath)[-1]
else:
    filename = ''

def series_from_string(series_str):
    ''' Given the junk input csv format, transform to numpy array '''
    series_spl = series_str.split(',')
    if len(series_spl) == 1440:
        try:
            return np.array([float(spl) for spl in series_spl], dtype=np.float32)
        except ValueError:
            return None
    return None

if not 'time_taken' in filename: #Ignore PI timing collection files
    # Exctract date from filename
    (month, day, year) = map(int, filename.split('.')[0].split('_')[-3:])
    date = (year, month, day) #Switch the order to something more logical
    # Check that the date is valid.  There were some leap-year issues
    try:
        datetime.date(*date) #Creating datetime objects checks for valid dates
        valid_date = True
    except ValueError:
        valid_date = False
    if valid_date:
        # Statistics about the number of daily time series skipped
        counter_total = 0
        counter_unpacked = 0
        counter_preprocessed = 0
        ###
        for line in sys.stdin:
            counter_total += 1
            unpack = line.split(',',1)
            if len(unpack) == 2:
                counter_unpacked += 1
                tag, series_str = unpack
                tag = tag.strip()
                series_str = series_str.strip()
                series = series_from_string(series_str)
                if not series is None:
                    counter_preprocessed += 1
                    print encode_line(tag, date, series)
    else:
        # We need to read the input stream, or hadoop streaming yells at us
        for line in sys.stdin:
            pass
# Print stats about the number of daily time series skipped
#        sys.stderr.write('%d,%d,%d\n'%(counter_total, counter_unpacked, counter_preprocessed))
