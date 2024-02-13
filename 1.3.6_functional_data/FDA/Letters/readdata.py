# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:51:48 2022

@author: Admin
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate

class letterData:
    
    def __init__(self,x,y):
        self.colors = ['blue','orange','green','red','purple','brown','pink','grey','olive','cyan']
        self.x,self.y = pd.read_csv(x),pd.read_csv(y)
        self.letters = self.find_letters()
        self.preprocess()
        self.x_smooth,self.y_smooth = self.x.copy(),self.y.copy()
        self.coefficients = pd.DataFrame(index = np.arange(0,self.x.shape[1]),columns = ['tx','cx','kx','ty','cy','ky'])
        for c_n,col in enumerate(self.x.columns):
            self.bSpline(c_n,col)
        
    def find_letters(self):
        return self.x.columns.str[0].unique()
    
    def preprocess(self):
        """ 
        to retain de shape of the letters, scale minmax 0-1 over x axis 
        for lower case i or l, this will skew data massively
        for the specific case of letters, we could centre x at zero, then scale via y 0-1 for example
        """
        
        cols = self.x.columns
        for c in cols:
            x,y = self.x[c],self.y[c]
            x,y = x[~np.isnan(x)],y[~np.isnan(y)]
            scale_diff = np.max(x) - np.min(x)
            x_sc = (x - np.min(x)) / scale_diff
            y_sc = (y - np.min(y)) / scale_diff
            self.x[c].iloc[:len(x)] = x_sc
            self.y[c].iloc[:len(x)] = y_sc
    
    def subset(self,letter = 'a'):
        return [self.x[[c for c in self.x.columns if letter in c]],
                self.y[[c for c in self.x.columns if letter in c]],
                self.x_smooth[[c for c in self.x.columns if letter in c]],
                self.y_smooth[[c for c in self.x.columns if letter in c]]]
            
    def single_plotx(self,letter = 'a'):
        """ Plot all x curves for a letter (10 spots) """
        fig,axe = plt.subplots(2,5,figsize = (15,3))
        datax = self.subset(letter)[0]
        datax_smooth = self.subset(letter)[2]
        n = np.min([10,datax.shape[1]])
        for i in range(n):
            x = np.array(datax.iloc[:,i])
            xs = np.array(datax_smooth.iloc[:,i])
            x = x[~np.isnan(x)]
            xs = xs[~np.isnan(xs)]
            axe[i//5,i%5].scatter(np.arange(0,len(x)),x)
            axe[i//5,i%5].plot(np.arange(0,len(xs)),xs,color = 'orange')
    
    def single_ploty(self,letter = 'a'):
        """ Plot all y curves for a letter (10 spots) """
        fig,axe = plt.subplots(2,5,figsize = (15,3))
        datay = self.subset(letter)[1]
        n = np.min([10,datay.shape[1]])
        for i in range(n):
            y = np.array(datay.iloc[:,i])
            y = y[~np.isnan(y)]
            axe[i//5,i%5].plot(np.arange(0,len(y)),y)
    
    def single_plot_letters(self,letter = 'a'):
        """ Plot all letter curves for a letter (10 spots) """
        fig,axe = plt.subplots(2,5,figsize = (15,5))
        [datax,datay] = self.subset(letter)[2:]
        n = np.min([10,datax.shape[1]])
        for i in range(n):
            x = np.array(datax.iloc[:,i])
            y = np.array(datay.iloc[:,i])
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]
            axe[i//5,i%5].plot(x,y)
    
    def agg_plotx(self):
        """ For all letters plot one aggregate x plot for each letter"""
        fig,axe = plt.subplots(1,len(self.letters),figsize = (4*len(self.letters),4))
        for i,l in enumerate(self.letters):
            datax = self.subset(l)[0]
            for j in range(datax.shape[1]):
                x = datax.iloc[:,j]
                x = x[~np.isnan(x)]
                axe[i].plot(np.arange(0,len(x)),x,color = self.colors[j])
    
    def agg_ploty(self):
        """ For all letters plot one aggregate y plot for each letter"""
        fig,axe = plt.subplots(1,len(self.letters),figsize = (4*len(self.letters),4))
        for i,l in enumerate(self.letters):
            datay = self.subset(l)[1]
            for j in range(datay.shape[1]):
                y = datay.iloc[:,j]
                y = y[~np.isnan(y)]
                axe[i].plot(np.arange(0,len(y)),y,color = self.colors[j])
    
    def agg_plot_letters(self):
        """ For all letters plot one aggregate letter plot for each letter"""
        fig,axe = plt.subplots(1,len(self.letters),figsize = (4*len(self.letters),4))
        for i,l in enumerate(self.letters):
            datax,datay = self.subset(l)[0],self.subset(l)[1]
            for j in range(datax.shape[1]):
                x = datax.iloc[:,j]
                x = x[~np.isnan(x)]
                y = datay.iloc[:,j]
                y = y[~np.isnan(y)]
                axe[i].plot(x,y,color = self.colors[j])
    
    def combined_plot(self):
        """ For all letters plot one aggregate x plot for each letter"""
        fig,axe = plt.subplots(1,3,figsize = (12,4))
        for i,c in enumerate(self.x.columns):
            axe[0].plot(np.arange(len(self.x[c])),np.array(self.x[c]),color = self.colors[i//10]) #plot all x
        for i,c in enumerate(self.y.columns):
            axe[1].plot(np.arange(len(self.y[c])),np.array(self.y[c]),color = self.colors[i//10]) #plot all y
        for i,c in enumerate(self.x.columns):
            axe[2].plot(np.array(self.x[c]),np.array(self.y[c]),color = self.colors[i//10]) #plot all
            
    def bSpline(self,i,col,order = 3,k = 10):
        """ Assume optimum knots 10 """
        x,y = self.x[col],self.y[col]
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        ticks = np.arange(0,len(x))
        
        k_t = np.linspace(0,len(x),k,dtype = np.int32)
        
        tx,cx,kx = interpolate.splrep(ticks,x,t = k_t[1:-1])
        ty,cy,ky = interpolate.splrep(ticks,y,t = k_t[1:-1])
        
        #total_abs_error_x = np.sum(np.abs(x - interpolate.splev(ticks,(tx,cx,kx))))
        #total_abs_error_y = np.sum(np.abs(x - interpolate.splev(ticks,(ty,cy,ky))))
        
        self.x_smooth[col][:len(x)] = interpolate.splev(ticks,(tx,cx,kx))
        self.y_smooth[col][:len(y)] = interpolate.splev(ticks,(ty,cy,ky))
        self.coefficients.iloc[i,:] = [tx,cx,kx,ty,cy,ky] 


if __name__ == '__main__':
    
    #import fda libraries
    #import skfda
    
    #Read in data for recognition of letters and plot data
    root = r'C:\Users\Admin\Documents\Python Scripts\FDA'
    os.chdir('Data')
    xdata_file,ydata_file = 'xdata_abc.csv','ydata_abc.csv'
    data = letterData(xdata_file,ydata_file)   
    
    """
    #elbow method - shows 10 is a good choice of total amount of knots.
    evals = 25
    error = np.zeros(evals)
    for k in range(3,3+evals):
        error[k-3] = bSpline(x, k = k)[-1]
        
    plt.figure()
    plt.plot(np.arange(0,len(error)),error)
    
    #bspline for full letter
    """
    
    #functional data analysis
    basis = data.x_smooth
    #fpca = FPCA(n_components = 2)
    #fpca.fit(basis)