# cython: profile=True

import numpy as np
cimport numpy as np
import pandas as pd
cimport cython

FTYPE = np.float32
ctypedef np.float32_t FTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

cdef int max_segment_length = 100 #optimal takes too long to compute

@cython.boundscheck(False)
def precompute_L(np.ndarray[FTYPE_t, ndim=1] x,np.ndarray[FTYPE_t, ndim=2] L):
    ''' Precompute MSE for each constant interval '''
    cdef int i,l,T
    T = len(x)
    # default to positive infinity
    cdef np.ndarray[FTYPE_t,ndim=1] stds
    cdef np.ndarray z
    #for l in range(1,T+1): #optimal requires considering segments up to length T
    for l in range(1,max_segment_length+1): #only consider 100-length-segments 
        #stds = pd.rolling_std(x,l,ddof=0)[l-1:]
        #for i,s in enumerate(stds):
        #    L[i,i+l-1] = s**2
        stds = np.array(pd.rolling_std(x,l,ddof=0),dtype=np.float32)
        for i in range(l-1,T):
            L[i-(l-1),i] = stds[i]**2
   
@cython.boundscheck(False) 
def optimal_piecewise_constant(np.ndarray[FTYPE_t, ndim=1] x, int k):
    ''' Given a time-series <x>, return ts,xs
        ts the *start* indices of constant intervals
        xs gives the values of each of the intervals
        len(ts) is <k>, the number of intervals '''
        
    # Compute the minimal cost and best breakpoints
    cdef ITYPE_t j,t2,t,T,dec,curr_dec
    cdef FTYPE_t opt,curr_opt
    T = len(x)
    #best MSE at each point
    cdef np.ndarray[FTYPE_t, ndim=2] OPT = np.zeros((T,k),dtype=FTYPE) 
    #optimal decision at each time t -- previous segment break-point
    cdef np.ndarray[ITYPE_t, ndim=2] DEC = np.zeros((T,k),dtype=ITYPE) 
    cdef np.ndarray[FTYPE_t, ndim=2] L = np.ones((T,T),dtype=FTYPE) / np.zeros((T,T),dtype=FTYPE)
    # Precompute L
    precompute_L(x,L)
    for t in range(T):
        OPT[t,0] = L[0,t]
    for j in range(1,k):
        for t in range(j+2,T):
            #Find the best previous segment breakpoint t
            #dec,opt are this breakpoint and the total cost respectively
            dec = t-1
            opt = float("+inf")
            # since length of segment bounded, only go down to t-max_seg+1
            #for curr_dec in range(t,j,-1): #t downto j+1 inclusive
            t2 = max(j,t-max_segment_length)
            for curr_dec in range(t,t2,-1): #t downto j+1 inclusive
                if L[curr_dec,t] >= opt: #can't do better, L will only increase
                    break
                curr_opt = OPT[curr_dec-1,j-1]+L[curr_dec,t]
                if curr_opt < opt:
                    dec = curr_dec
                    opt = curr_opt
            DEC[t,j] = dec
            OPT[t,j] = opt
                
    # create the segments ts and xs by running backward through the data
    ts = [] #beginnings of segments
    xs = [] #values of segments
    t = T-1
    # Find the least space needed.  This is the smallest 
    #   number of segments to get the minimum error
    #   in some cases, adding segments doesn't decrease error
    j = np.nonzero(OPT[T-1,:] == min(OPT[T-1,:]))[0][0]
    while t > 0:
        t_prev = t
        t = DEC[t,j]
        j -= 1
        ts = [t] + ts
        xs = [np.mean(x[t:t_prev+1])] + xs
        t -= 1 #now it's the end of the next interval
    
    return ts,xs       

