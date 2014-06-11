import numpy as np

def L(x,i1,i2):
    ''' Cost of best-fit line for range of values [i1,i2) '''
    return np.std(x[i1:i2+1])**2

def update(x,OPT,t,j):
    ''' Find the best previous segment breakpoint t
        return this breakpoint and the total cost '''
    min_t2 = t-1
    min_so_far = float("+inf")
    for t2 in range(t,j,-1): #t downto j+1 inclusive
        opt = OPT[t2-1,j-1]+L(x,t2,t)
        if opt < min_so_far:
            min_t2 = t2
            min_so_far = opt
    return (min_t2,min_so_far)
    
def optimal_piecewise_constant(x,k):
    ''' Given a time-series <x>, return ts,xs
        ts the *start* indices of constant intervals
        xs gives the values of each of the intervals
        len(ts) is <k>, the number of intervals '''
        
    # Compute the minimal cost and best breakpoints
    T = len(x)
    OPT = np.zeros((T,k)) #best MSE at each point
    DEC = np.zeros((T,k),dtype=np.int32) #optimal decision at each time t -- previous segment break-point
    for t in range(T):
        OPT[t,0] = L(x,0,t)
    for j in range(1,k):
        for t in range(j+2,T):
            dec,opt = update(x,OPT,t,j)
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

