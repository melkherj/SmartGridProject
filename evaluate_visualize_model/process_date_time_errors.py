#!/usr/bin/env python

import sys, os
sys.path.append(os.environ['SMART_GRID_SRC'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from collections import defaultdict

''' Given the path to a text file of the form:
        predictor type^tag name^date^space^error

    Export a vocabulary of tags, predictors, and dates, along with a dataframe
    containing the space/error data.  

    The dataframe/vocabulary files exported are in the same directory as
        the input file, but with .txt replaced by .pandas and .pkl

'''

def hash_line(line, predictor_hash, tag_hash, date_hash):
    predictor, tag, date, compressed, space, error, quantile_lower, \
        quantile_upper = line.split('^')
    predictor = predictor_hash[predictor]
    tag = tag_hash[tag]
    date = date_hash[date]
    space = float(space)
    error = float(error)
    quantile_lower = float(quantile_lower)
    quantile_upper  = float(quantile_upper)
    return (predictor, tag, date, compressed, space, error, quantile_lower, 
        quantile_upper)

def load_compressed_df(filename, save=False):
    ''' Load the compressed representation of sensor data in a dataframe '''
    # Create vocabularies for tags, dates, and predictors.  Count number of lines
    tag_set = set([])
    date_set = set([])
    predictor_set = set([])
    aggregates = defaultdict(list)
    with open(filename, 'r') as f:
        for i,line in enumerate(f): #f is a lazy generator of lines
            if line[:3] == '^^^':
                predictor, aggregate = line[3:].split('^')
                aggregates[predictor].append(aggregate)
                continue
            if i % 100000 == 0:
                print i
            predictor, tag, date, _,  _, _, _, _ = line.split('^')
            predictor_set.add(predictor)
            tag_set.add(tag)
            date_set.add(date)
    n = i + 1 # Number of lines in the file.  Needed to pre-allocate a numpy array
    predictor_list = sorted(list(predictor_set))
    predictor_hash = dict((predictor, i) 
        for i,predictor in enumerate(predictor_list))
    tag_list = sorted(list(tag_set))
    tag_hash = dict((tag, i) for i,tag in enumerate(tag_list))
    date_list = sorted(list(date_set))
    date_hash = dict((date, i) for i,date in enumerate(date_list))

    with open(filename, 'r') as f:
        E = [hash_line(line, predictor_hash, tag_hash, date_hash) for line in f if not line[:3] == '^^^']
        predictors, tags, dates, all_compressed, spaces, errors, quantile_lower, \
            quantile_upper = zip(*E)
        predictors = pd.Series(predictors, dtype=np.int32, name='predictor') 
        tags = pd.Series(tags, dtype=np.int32, name='tag') 
        dates = pd.Series(dates, dtype=np.int32, name='date') 
        all_compressed = pd.Series(all_compressed, dtype=np.object, 
            name='compressed') 
        spaces = pd.Series(spaces, dtype=np.float32, name='space') 
        errors = pd.Series(errors, dtype=np.float32, name='error') 
        quantile_lower = pd.Series(quantile_lower, 
            dtype=np.float32, name='quantile_lower') 
        quantile_upper = pd.Series(quantile_upper, 
            dtype=np.float32, name='quantile_upper') 

    # Convert to DataFrame 
    df = pd.concat([predictors, tags, dates, all_compressed, spaces, errors, 
        quantile_lower, quantile_upper], axis=1)
    df = df.set_index(['tag','date','predictor']).unstack(level=-1)
    context = {
             'predictor_list':predictor_list, 
             'predictor_hash':predictor_hash, 
             'tag_list':tag_list, 
             'tag_hash':tag_hash, 
             'date_list':date_list,
             'date_hash':date_hash,
             'aggregates':aggregates
    }
    if save:
        # Save DataFrame/vocabulary lists to files
        df.save(filename.replace('txt','pandas'))
        with open(filename.replace('txt','pkl'),'wb') as f:
            pickle.dump(context, f)
    return context, df

if __name__ == '__main__':
    load_compressed_df(sys.argv[1], save=True)
