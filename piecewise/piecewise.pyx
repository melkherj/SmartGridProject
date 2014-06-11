import numpy as np
cimport numpy as np
import pandas as pd

FTYPE = np.float32
ctypedef np.float32_t FTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

def L_fun(x,i1,i2):
    ''' Cost of best-fit line for range of values [i1,i2) '''
    return np.std(x[i1:i2+1])**2

def precompute_L(x):
    cdef int i,l,T
    T = len(x)
    # default to positive infinity
    cdef np.ndarray L = np.ones((T,T),dtype=FTYPE) / np.zeros((T,T),dtype=FTYPE)
    cdef np.ndarray stds
    for l in range(1,T+1):
        stds = pd.rolling_std(x,l,ddof=0)[l-1:]
        for i,s in enumerate(stds):
            L[i,i+l-1] = s**2
    return L
    
def optimal_piecewise_constant(np.ndarray x,int k):
    ''' Given a time-series <x>, return ts,xs
        ts the *start* indices of constant intervals
        xs gives the values of each of the intervals
        len(ts) is <k>, the number of intervals '''
        
    # Compute the minimal cost and best breakpoints
    cdef np.ndarray L
    cdef ITYPE_t j,t,T,dec,curr_dec
    cdef FTYPE_t opt,curr_opt
    T = len(x)
    #best MSE at each point
    cdef np.ndarray OPT = np.zeros((T,k),dtype=FTYPE) 
    #optimal decision at each time t -- previous segment break-point
    cdef np.ndarray DEC = np.zeros((T,k),dtype=ITYPE) 
    # Precompute L
    L = precompute_L(x)
    for t in range(T):
        OPT[t,0] = L[0,t]
    for j in range(1,k):
        for t in range(j+2,T):
            #Find the best previous segment breakpoint t
            #dec,opt are this breakpoint and the total cost respectively
            dec = t-1
            opt = float("+inf")
            for curr_dec in range(t,j,-1): #t downto j+1 inclusive
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

