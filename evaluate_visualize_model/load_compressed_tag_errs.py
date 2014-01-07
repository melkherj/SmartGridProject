import pandas as pd
import pickle
from process_date_time_errors import *

def load_compressed_tag_errs(err_file_basename):
    all_vars = {'df': pd.load(err_file_basename+'.pandas') }
    with open(err_file_basename+'.pkl','r') as f:
        all_vars.update(pickle.load(f))
    return all_vars
    
if __name__ == '__main__':
    v = load_compressed_tag_errs(sys.argv[1])
    globals().update(v)
