import matplotlib.pyplot as plt
import random
import numpy as np

def f():
    plt.plot(range(5))

def sample_df(df, samples):
    ''' Samples rows of a dataframe '''
    if samples >= len(df):
        return df
    else:
        rows = random.sample(df.index, samples)
        return df.ix[sorted(rows)]

def plot_daily_series(df, samples=5, title=None, ylim=None):
    ''' Given a dataframe containing daily series for sensors, 
        randomly sample <samples> days, and plot the daily variation for those days'''
    if not samples is None:
        df = sample_df(df, samples)
    X = df.copy()
    # Find the minimum and maximum
    ymax = X.max().max()
    ymin = X.min().min()
    yrange = ymax - ymin
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(221)
    fig.suptitle('yrange: %.5f'%yrange)
    if yrange == 0:
        yrange = 1
    #Add a small amount of random noise, so we can see overlapping lines
    X.index = X.index.droplevel(0)
    X.T.plot(ax=ax, grid=False, legend=False, title=title)
    ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fancybox=True, shadow=True)
    # Set reasonable x/y limits.  y slightly above/below the largest values
    plt.xlim([min(df.columns),max(df.columns)])
    if ylim is None:
        if yrange == 0:
            ymax, ymin = (-1,1)
        else:
            ymax += 0.1*yrange
            ymin -= 0.1*yrange
        ylim = [ymin, ymax]  
    plt.ylim(ylim)
    ts = df.columns
    # downsample time to 12 points
    if len(ts) > 12:
        ts = [ts[i] for i in range(0,len(ts),len(ts)/12)]
    plt.xticks(ts,['%2d:%02d'%(t/60,t%60) for t in ts],rotation=70)
