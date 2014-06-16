from scipy.stats import linregress
import pylab
import sys, os
sys.path.append(os.environ['SMART_GRID_SRC'])
from matplotlib.pyplot import *
import pandas as pd
import random, pickle
from get_tag_series import part_tag_dict, get_tag_series
from serialize_tag_date import decode_date
import datetime
import numpy as np
from numpy.linalg import norm
from operator import itemgetter
#import statsmodels.tsa.api as tsa
from evaluate_visualize_model.load_compressed_tag_errs import load_compressed_tag_errs
from compression_methods import decompress_df, compressors, SVDCompressor, WaveletCompressor, fill_coeffs, Compressor
from evaluate_visualize_model.analyze_tag_errs import sample_df, plot_daily_series
import pywt
from scipy.linalg import svd
from serialize_tag_date import encode_date, decode_date, \
    b64_encode_series, b64_decode_series
import random
from compression_methods import fill_coeffs
from collections import defaultdict
import nltk
import re
import pywt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from nltk import FreqDist
from IPython.display import HTML

pylab.rcParams['figure.figsize'] = (12.0, 7.0)

toggle_sections = HTML("""
<script type="text/javascript">
     show=true;
     function toggle(){
         if (show){
             $('div.input').hide();
             $('div.output_area').hide();
             $("#toggle").parent().parent().show()
         }else{
             $('div.input').show();
             $('div.output_area').show();
         }
         show = !show
     }
 </script>
 <a id="toggle" href="javascript:toggle()" target="_self">toggle sections</a>""")

def tag_type(tag):
    ''' The type of sensor, eg air flow/outside temperature/...'''
    return tag.split('.')[-1]

def choose_compressor(se, max_error):
    ''' Return the best compressor and complexity hyperparameter k
        best is measured by lowest error, bounding space'''
    tradeoff = get_space_mse_tradeoff(se)
    best = tradeoff[tradeoff['error'] < max_error].head(n=1)[['compressor','k']]
    return best

def get_space_mse(se):
    ''' Deserialize 64-bit-encoded space/error.  return (space,error)'''
    se = b64_decode_series(se)
    n = len(se)
    return se[:n/2],se[n/2:]

def get_space_mse_tradeoff(se):
    ''' Given a dataframe <se> with columns 'compressor' and 'space_err',
        'compressor' giving the compression method used
        'space_err' giving a numpy array for the tradeoff between space and error
        (first half of array gives #floats taken, second half gives errors)
        
        Return a DataFrame <tradeoff> with columns 'compressor','error','k','space'
        <tradeoff> is sorted increasing by 'space'
        space is the space taken per compression method, 'compressor' is the method', 'error' is the error (we've used MSE so far, could be anything), and 'k' is the complexity hyperparameter of the compression methods used.  
        '''
    spaces = []
    errs = []
    compressors = []
    ks = []
    for _,row in se.iterrows():
        s,e = get_space_mse(row['space_err'])
        c = [row['compressor']]*len(s)
        spaces += s.tolist()
        errs += e.tolist()
        compressors += c
        ks += range(1,len(s)+1)
    tradeoff = pd.DataFrame({'space':spaces,'error':errs,'k':ks,
                         'compressor':compressors}).sort(columns=['space'])
    l = 0
    while len(tradeoff) != l:
        l = len(tradeoff)
        decreased = tradeoff['error'].diff() < 0
        decreased[0] = True
        tradeoff = tradeoff[decreased]
    tradeoff.index = tradeoff['space']
    return tradeoff

def choose_space_binning(tradeoff,k=10):
    ''' reduce space/error tradeoff dimensionality by clustering space in log-space
        Given a tradeoff curve, return <k> spaces that give a good representation of
        the entire space curve '''
    spaces = tradeoff['space'].values
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(np.reshape(np.log(spaces),(len(spaces),1)))
    labels_seen = set([]) #clusters labels we've seen so far
    spaces2 = [] #just k representative spaces chosen, after clustering in log space
    # choose 10 spaces: smallest space where cluster label was seen
    for l,s in zip(labels,spaces):
        if not (l in labels_seen):
            spaces2.append(s)
            labels_seen |= set([l])
    return np.array(spaces2)

def plot_space_mse_tradeoff(tradeoff):
    ''' <tradeoff> is a dataframe returned by get_space_mse_tradeoff
        This plots the space/error tradeoff on a log-log plot, 
        colored by compression method.  
        See the description of the tradeoff dataframe in the 
        get_space_mse_tradeoff function '''
    clf()
    xscale('log')
    yscale('log')
    compressor_list = list(set(tradeoff['compressor']))
    colors = iter(cm.rainbow(np.linspace(0, 1, len(compressor_list))))
    for c in compressor_list:
        t = tradeoff[tradeoff['compressor'] == c]
        scatter(t['space'],t['error'],label=c,color=next(colors),s=20,marker='.')
    legend(bbox_to_anchor=(1.3,0.5))
    xlabel('space')
    ylabel('error')

def apply_space_binning(spaces2, tradeoff):
    ''' Given the binning of spaces <spaces2>, and a tradeoff curve 
        with spaces not necessily in the range of spaces2, 
        map the space index of tradeoff to spaces2
        This gives the error e each s in spaces2, where e is the smallest
        error in tradeoff such that the corresponding space in tradeoff 
        <= s '''
    k = len(spaces2) #number of space/err points to keep
    spaces = np.zeros((k,))
    i = 0
    s_last = tradeoff.index[0]
    for s in tradeoff.index:
        while (i < k) and (s > spaces2[i]):
            spaces[i] = s_last
            i += 1
        if i == k:
            break
        # s2[i] is now the smallest space >= s
          # e is monotonically decreasing
        spaces[i] = s
        # so e2[s2[i]] is now the smallest err we've examined so far
        # such that space does not exceed s2[i]
        s_last = s
    if i < k-1:
        spaces[i:] = spaces[i]
    return tradeoff.ix[spaces]

def space_err_features(spaces, se):
    ''' create a feature vector from the space-error tradeoff curve
        this is log(spaces), log(errors) concatenated
        '''
    tradeoff = get_space_mse_tradeoff(se)
    tradeoff = apply_space_binning(spaces,tradeoff)
    #return np.concatenate([np.log(tradeoff['space'].values),
    return tradeoff['error'].values

def choose_space_mse(se):
    s,e = get_space_mse(se)
    i = random.randint(0,len(s)-1)
    return s[i],e[i]

def meta_decompress_tag(df_compressed, tag, context):
    ''' Given meta-compressed representation in df_compressed and context, 
        return a decompressed representation for one tag '''
    tag_index = context['tag_hash'][tag]
    df_compressed2 = df_compressed[df_compressed.index.get_level_values(0) == tag_index]
    compressor_index = df_compressed2['compressor'].values[0]
    compressor_name = context['compressor_list'][compressor_index]
    return decompress_df(df_compressed2, context, compressor_name)
