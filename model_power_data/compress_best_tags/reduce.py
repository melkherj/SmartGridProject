#!/usr/bin/env python

import sys, os, re
sys.path.append(os.environ['SMART_GRID_SRC'])
from get_tag_series import get_next_tag_series
import scipy as sp
import numpy as np
from numpy.linalg import norm
from compression_methods import compress_serialize_best

next_line = None
eof = False
while (not eof):
    df, next_line = get_next_tag_series(sys.stdin, first_line=next_line)
    if df is None:
        break
    else:
        # Compress tag, print out compression
        compress_serialize_best(df)
