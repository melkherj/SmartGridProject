#!/usr/bin/env python
import sys, os
sys.path.append(os.environ['SMART_GRID_SRC'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

''' Given the path to a text file of the form:
        predictor type^tag name^date^space^error

    Export a vocabulary of tags, predictors, and dates, along with a dataframe
    containing the space/error data.  

    The dataframe/vocabulary files exported are in the same directory as
        the input file, but with .txt replaced by .pandas and .pkl

'''

if __name__ == '__main__':
    filename = sys.argv[1]
    
    # Create vocabularies for tags, dates, and predictors.  Count number of lines
    predictor_set = set([])
    tag_set = set([])
    date_set = set([])
    with open(filename, 'r') as f:
        for i,line in enumerate(f): #f is a lazy generator of lines
            if i % 100000 == 0:
                print i
            predictor, tag, date, _, _ = line.split('^')
            predictor_set.add(predictor)
            tag_set.add(tag)
            date_set.add(date)
    n = i + 1 # Number of lines in the file.  Needed to pre-allocate a numpy array
    predictor_list = sorted(list(predictor_set))
    predictor_hash = dict((predictor, i) for i,predictor in enumerate(predictor_list))
    tag_list = sorted(list(tag_set))
    tag_hash = dict((tag, i) for i,tag in enumerate(tag_list))
    date_list = sorted(list(date_set))
    date_hash = dict((date, i) for i,date in enumerate(date_list))
   
    # Create the numpy array of space/errors per predictor 
    prediction_errors = np.zeros(shape=(n,5), dtype=np.float32)
    with open(filename, 'r') as f:
        for i,line in enumerate(f): #f is a lazy generator of lines
            if i % 10000 == 0:
                print i
            predictor, tag, date, space, error = line.split('^')
            predictor = predictor_hash[predictor]
            tag = tag_hash[tag]
            date = date_hash[date]
            space = float(space)
            error = float(error)
            prediction_errors[i,:] = (predictor, tag, date, space, error)
   
    # Convert to  
    df = pd.DataFrame(prediction_errors,  columns=['predictor', 'tag', 'date', 'space', 'error'])
    df = df.set_index(['tag','date','predictor']).unstack(level=-1)
    # Save DataFrame/vocabulary lists to files
    df.save(filename.replace('txt','pandas'))
    with open(filename.replace('txt','pkl'),'wb') as f:
        pickle.dump(
            {
             'predictor_list':predictor_list, 
             'predictor_hash':predictor_hash, 
             'tag_list':tag_list, 
             'tag_hash':tag_hash, 
             'date_list':date_list,
             'date_hash':date_hash
            }, f
        )
