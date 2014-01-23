import sys, os, re
sys.path.append(os.environ['SMART_GRID_SRC'])
from serialize_tag_date import encode_date, decode_date, \
    b64_encode_series, b64_decode_series
import numpy as np
from numpy.linalg import norm
import pandas as pd
import sys
import datetime

global p
p = 0.05

### Generic Functions ###

def encode_timestamp_date(d):
    return encode_date(( d.year, d.month, d.day ))


### One function for each compression method ###
#     Every such function takes a data frame of the form:
#     tag/date/values... for each row, and returns
#     a vector space, one value for each row, and a reconstructed version
#     of the values matrix, after compressing/decompressing

class Compressor:
    name = None #overwrite this

    def compress_evaluate(self, df):
        global p
        aggregate, df_compressed = self.compress(df)
        df_compressed = pd.DataFrame(df_compressed, columns=['compressed'])
        df_reconstructed = self.decompress(aggregate, df_compressed)
        aggregate = b64_encode_series(aggregate)
        err = df-df_reconstructed
        df_compressed['error'] = err.apply(norm, axis=1)
        df_compressed['space'] = df_compressed['compressed'].apply(len) + \
            len(aggregate)/float(len(df_compressed))
        p = p/float(len(df))
        df_compressed['quantile_lower'] = err.quantile(p, axis=1)
        df_compressed['quantile_upper'] = err.quantile(1-p, axis=1)
        return aggregate, df_compressed

    def serialize_compressed(self, aggregate, df_compressed, outfile=sys.stdout):
        # Print the serialized aggregate compression information.  This is
        #  the compressed data shared across the entire tag
        tag = df_compressed.index[0][0]
        outfile.write('^^^%s^%s^%s\n'%(self.name, tag, aggregate) )
        # Print each serialized compressed tag/date
        for index, row in df_compressed.iterrows():
            tag, date = index
            date = encode_timestamp_date(date)
            outfile.write('%s^%s^%s^%s^%.5f^%.5f^%.5f^%.5f\n'%(self.name, tag, date, 
                row['compressed'], row['space'], row['error'], row['quantile_lower'], row['quantile_upper']))

    def compress(self, df):
        ''' Returns a compressed dataframe, and another numpy array storing
            aggregate data for the entire group 
            '''
        raise NotImplementedError("Please Implement this method")

    def decompress(self, df, aggregate):
        ''' Given compressed DataFrame, produce dataframe '''
        raise NotImplementedError("Please Implement this method")
        

class TagConstantCompressor(Compressor):
    name = 'tag_constant'
    
    def compress(self, df):
        compressed = df.ix[:,0].map(lambda row:'-')
        mean = df.mean(axis=1).mean(axis=0, dtype=np.float32)
        return np.array([mean]), compressed

    def decompress(self, aggregate, df_compressed):
        aggregate = pd.Series(aggregate*np.ones(shape=(1440,)))
        return df_compressed['compressed'].apply(lambda row:aggregate)

class ConstantCompressor(Compressor):
    name = 'constant'
    
    def compress(self, df):
        df_compressed = pd.Series(df.mean(axis=1), dtype=np.float32)
        df_compressed = df_compressed.apply(b64_encode_series)
        return np.array([]), df_compressed

    def decompress(self, aggregate, df_compressed):
        df = df_compressed.copy()
        df['const'] = pd.Series(df['compressed'].apply(b64_decode_series), 
            dtype=np.float32)
        for i in range(1440):
            df[i] = df['const']
        del df['compressed'], df['const']
        return df
    
class StepCompressor(Compressor):
    name = 'step'

    def __init__(self, steps=10):
        self.steps = steps

    def compress_series(self, series):
        diff = series.diff()
        # 1 - s/1440 gives the proportion of series values
        #  to be kept in step function
        d = diff.quantile(1 - self.steps/1440.0)
        diff = diff.fillna(d)
        series = series[diff > d]
        series.sort()
        return b64_encode_series(series.index)+'^'+b64_encode_series(series)
        
    def decompress_series(self, compressed):
        index = b64_decode_series(compressed).tolist()
        index.append(1440)
        series = pd.Series([0]*1440, dtype=np.float32)
        for i in range(len(index)-1):
            series[index[i]:index[i+1]] = series[index[i]]
        return series

    def compress(self, df):
        compressed = df.apply(self.compress_series, axis=1)
        return np.array([]), compressed
        
    def decompress(self, aggregate, df_compressed):
        df = df_compressed['compressed'].apply(self.decompress_series)
        diff = df.diff(axis=1)
        d = diff.quantile(1 - self.steps/1440.0)
        diff.fillna(d)
        compressed = df[diff > d]
        return df

class MeanCompressor(Compressor):
    name = 'mean'

    def compress(self, df):
        compressed = df.ix[:,0].map(lambda row:'-')
        mean = df.mean(axis=0).values
        return mean, compressed
        
    def decompress(self, aggregate, df_compressed):
        aggregate = pd.Series(aggregate)
        return df_compressed['compressed'].apply(lambda row:aggregate)

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
tag_constant_compressor = TagConstantCompressor()
constant_compressor = ConstantCompressor()
step_compressor = StepCompressor()
mean_compressor = MeanCompressor()
all_compressors = dict( (compressor.name, compressor) for compressor in
    [tag_constant_compressor, constant_compressor, mean_compressor]) #, step_compressor] )

def compress_serialize_all(df, outfile=sys.stdout):
    ''' For each registered compressor, compress the given DataFrame <df>
        Serialize the compressed data, and output to <outfile> '''
    for compressor in all_compressors.values():
        aggregate, df_compressed = compressor.compress_evaluate(df)
        compressor.serialize_compressed(aggregate, 
            df_compressed, outfile=outfile)


def decompress_df(df_compressed, context, compressor):
    def decompress_group(df):
        tag = context['tag_list'][int(df.name)]
        aggregate = b64_decode_series(aggregates[tag])
        return compressor.decompress(aggregate, df)
    aggregates = context['aggregates'][compressor.name]
    compressor_index = context['predictor_list'].index(compressor.name)
    df_compressed = df_compressed.xs(compressor_index, axis=1, level=1)
    df_series = df_compressed.groupby(df_compressed.index.get_level_values(0)) \
        .apply(decompress_group)
    # Delete non-integer columns
    for column in df_series.columns:
        try:
            _ = int(column)
        except:
            del df_series[column]
    df_series['tag'] = [context['tag_list'][i] 
        for i in df_compressed.index.get_level_values(0)]
    df_series['date'] = [context['date_list'][i] 
        for i in df_compressed.index.get_level_values(1)]
    df_series['date'] = df_series['date'].apply(
        lambda d: np.datetime64(datetime.date(*decode_date(d))) )
    df_series = df_series.set_index( ['tag','date']  )
    return df_series

