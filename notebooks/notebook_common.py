import sys, os
sys.path.append(os.environ['SMART_GRID_SRC'])
from matplotlib.pyplot import *
import pandas as pd
import random, pickle
from get_tag_series import part_tag_dict, get_tag_series
from serialize_tag_date import decode_date
import datetime
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

from IPython.display import HTML
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
    ''' Given a part of the space vs. error dataframe'''
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
    return tradeoff

def bin_space_mse_tradeoff(tradeoff):
    ''' Binning by compression ratios of 1 + [10**(-2) .. 10**5]
        Select the lowest error with at least the given compression ratio
        Return just the errors '''
    crs = 1.0+10.0**arange(-3,4) #compression ratios
    i = 0
    errs = ones(len(crs))
    for _,row in tradeoff.iterrows():
        if row['compression_ratio'] >= crs[i]:
            errs[i] = row['error']
            i += 1
            if i >= len(crs):
                break
    return pd.Series(errs)

def choose_space_mse(se):
    s,e = get_space_mse(se)
    i = random.randint(0,len(s)-1)
    return s[i],e[i]

def plot_space_err(se):
    # Choose random space/error from this sensor-compressor
    
    compressor_list = sorted(list(set(se['compressor'])))
    compressor_vocab = dict((v,i) for i,v in enumerate(compressor_list))
    colors = iter(mpl.cm.rainbow(np.linspace(0, 1, len(compressor_list))))
    
    pylab.rcParams['figure.figsize'] = (8.0, 7.0)
    for i,compressor in enumerate(compressor_list):
        # Choose random sensor-space compressor pairs
        se2 = sample_df(se[se['compressor'] == compressor],100)
        spaces,errs = zip(*se2['space_err'].apply(choose_space_mse))
        scatter(spaces,errs,label=compressor,color=next(colors),s=10,marker='.')
        #c=map(compressor_vocab.__getitem__,se2['compressor']),
    legend(bbox_to_anchor=(1.5,0.5))
    xlabel('compression ratio')
    ylabel('error')
    xscale('log')
    yscale('log')
    ylim(10**(-5),10**5)
    xlim(10**(-2),10**7)
    show()

def meta_decompress_tag(df_compressed, tag, context):
    ''' Given meta-compressed representation in df_compressed and context, 
        return a decompressed representation for one tag '''
    tag_index = context['tag_hash'][tag]
    df_compressed2 = df_compressed[df_compressed.index.get_level_values(0) == tag_index]
    compressor_index = df_compressed2['compressor'].values[0]
    compressor_name = context['compressor_list'][compressor_index]
    return decompress_df(df_compressed2, context, compressor_name)
