# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#use ginput to generate dataset of 5 points

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cols = ['a{}'.format(i) for i in range(10)] + ['b{}'.format(i) for i in range(10)] + ['c{}'.format(i) for i in range(10)]
df_x,df_y = pd.DataFrame(columns = cols, index = np.arange(0,50)),pd.DataFrame(columns = cols, index = np.arange(0,50))

fig,axe = plt.subplots(figsize = (5,5))
axe.axvline(0.05,linestyle = '--')
axe.axhline(0.05,linestyle = '--')
for i in range(30):
    coor = plt.ginput(50,timeout = 60)
    df_x[cols[i]] = np.array(coor)[:,0]
    df_y[cols[i]] = np.array(coor)[:,1]
    print(i)
    
df_x[df_x < 0.05] = np.nan
df_y[df_y < 0.05] = np.nan

#save x and y data
df_x.to_csv('xdata_abc.csv',index = False)
df_y.to_csv('ydata_abc.csv',index = False)