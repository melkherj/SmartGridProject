import sys, os, re
sys.path.append(os.environ['SMART_GRID_SRC'])
from serialize_tag_date import encode_date
import numpy as np
from numpy.linalg import norm

### Generic Functions ###

def encode_timestamp_date(d):
    return encode_date(( d.year, d.month, d.day ))

def compute_print_space_error(compression_type, df, prediction, space):
    errors = (df - prediction).apply(lambda row:norm(row), axis=1)
    for i in range(len(df)):
        tag, date = df.index[i]
        date = encode_timestamp_date(date)
        print '%s^%s^%s^%.5f^%.5f'%(compression_type, tag, 
            date, space[i], errors[i])

### One function for each compression method ###
#     Every such function takes a data frame of the form:
#     tag/date/values... for each row, and returns
#     a vector space, one value for each row, and a reconstructed version
#     of the values matrix, after compressing/decompressing

def constant_compress(df):
    ''' Use Predict mean '''
    space = 1/1440.0 # only one value per day is needed
    space = np.array([space]*len(df))
    day_means = df.mean(axis=1)
    prediction = np.vstack([day_means for _ in range(df.shape[1])]).T
    return space, prediction

def perfect_step_compress(df):
    space = np.sum(np.diff(df, axis=1) > 0, axis=1) + 1 #The number of jumps in 
    space = space / 1440.0
    prediction = df 
    return space, prediction

def mean_compress(df):
    ''' Predict the mean '''
    mean = df.mean(axis=0)
    #Compression ratio = 1 day space / all days space = 1 / #days
    space = 1/float(len(df)) 
    space = np.array([space]*len(df))
    # stack mean len(df) times
    prediction = np.vstack([mean for _ in range(len(df))])
    return space, prediction

### Run all compression methods ###

tag_compression_models = {
    'constant':constant_compress,
    'perfect_step':perfect_step_compress,
    'mean':mean_compress
}

def compress_all(df):
    for compression_type, compress in tag_compression_models.iteritems():
        space, prediction = compress(df)
        compute_print_space_error(compression_type, df, prediction, space)
