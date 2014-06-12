#!/usr/bin/env python
import numpy as np
cimport numpy as np
import sys, os
sys.path.append(os.environ['SMART_GRID_SRC'])
import pandas as pd
import random
import pickle
from collections import defaultdict
import time

FTYPE = np.float32
ctypedef np.float32_t FTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

''' Given the path to a text file of the form:
        compressor type^tag name^date^space^error

    Export a vocabulary of tags, compressors, and dates, along with a dataframe
    containing the space/error data.  

    The dataframe/vocabulary files exported are in the same directory as
        the input file, but with .txt replaced by .pandas and .pkl

'''

t0 = time.time()
def print_time(s):
    print '%.3f: %s'%(time.time() - t0,s)

def hash_line(line, compressor_hash, tag_hash, date_hash):
    compressor, tag, date, compressed, space, error, _, _ = line.split('^')
    compressor = compressor_hash[compressor]
    tag = tag_hash[tag]
    date = date_hash[date]
    space = float(space)
    error = float(error)
    return (compressor, tag, compressed, date, space, error)

def load_compressed_df(filename, save=False):
    ''' Load the compressed representation of sensor data in a dataframe '''

    cdef int i
    cdef ITYPE_t tag_index, date_index, compressor_index
    cdef FTYPE_t space, error

    print_time('Creating date/tag/compression method vocabulary...')
    # Create vocabularies for tags, dates, and compressors.  Count number of lines
    tag_set = set([])
    date_set = set([])
    compressor_set = set([])
    n = 0
    aggregates = defaultdict(dict)
    with open(filename, 'r') as f:
        for i,line in enumerate(f): #f is a lazy generator of lines
            if line[:3] == '^^^':
                compressor, tag, aggregate = line[3:].split('^')
                aggregates[compressor][tag] = aggregate
            else:
                compressor, tag, date, _,  _, _, _, _ = line.split('^')
                compressor_set.add(compressor)
                tag_set.add(tag)
                date_set.add(date)
                n += 1
    print_time('%d lines to process\n'%n)
    compressor_list = sorted(list(compressor_set))
    compressor_hash = dict((compressor, i) 
        for i,compressor in enumerate(compressor_list))
    tag_list = sorted(list(tag_set))
    tag_hash = dict((tag, i) for i,tag in enumerate(tag_list))
    date_list = sorted(list(date_set))
    date_hash = dict((date, i) for i,date in enumerate(date_list))

    print_time('Allocating tag/date/space/err/compressor arrays...')
    cdef np.ndarray tags = np.zeros((n,),dtype=ITYPE)
    cdef np.ndarray dates = np.zeros((n,),dtype=ITYPE)
    cdef np.ndarray spaces = np.zeros((n,),dtype=FTYPE)
    cdef np.ndarray errors = np.zeros((n,),dtype=FTYPE)
    cdef np.ndarray compressors = np.zeros((n,),dtype=ITYPE)
    # There isn't a nice cython type for objects, unfortunately
    all_compressed = np.empty((n,),dtype=np.object)
    
    with open(filename, 'r') as f:
        print_time('Setting values in tag/date/space/err/compressor arrays...')
        i = 0 #number of non-aggregate lines
        for line in f:
            if i % 1000 == 0:
                sys.stdout.write(('%20d'%i)+'\r'*20); sys.stdout.flush(); 
            if not line[:3] == '^^^':
                # c is the compressor index
                c, tag_index, compressed, date_index, space, \
                    error = hash_line(line, compressor_hash, tag_hash, date_hash)
                tags[i] = tag_index
                dates[i] = date_index
                all_compressed[i] = compressed
                spaces[i] = space
                errors[i] = error
                compressors[i] = c
                i += 1

    # Convert to DataFrame 
    print_time('Combining arrays into a single DataFrame...')
    index = pd.MultiIndex.from_arrays([tags, dates],names=['tag','date'])
    df = pd.DataFrame({
        'compressed':all_compressed,
        'space':spaces,
        'error':errors,
        'compressor':compressors
    },index=index)
    context = {
             'compressor_list':compressor_list, 
             'compressor_hash':compressor_hash, 
             'tag_list':tag_list, 
             'tag_hash':tag_hash, 
             'date_list':date_list,
             'date_hash':date_hash,
             'aggregates':aggregates
    }
    print_time('Saving to Disk...')
    if save:
        # Save DataFrame/vocabulary lists to files
        df.save(filename.replace('txt','pandas'))
        with open(filename.replace('txt','pkl'),'wb') as f:
            pickle.dump(context, f)
    return context, df
