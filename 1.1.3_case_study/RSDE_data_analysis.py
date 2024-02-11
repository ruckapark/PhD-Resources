# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:57:39 2024

Code for analysis of RSDE data in 2021 INERIS report

Example used in Section 1.1.3 State of the Art - PhD George Ruck

@author: ruckapark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path, index_col=None)
        self.concentration_columns = [col for col in self.data.columns if 'concentration' in col.lower()]
        self.EFF_conc_cols = [col for col in self.concentration_columns if 'EFF' in col]
        self.INF_conc_cols = [col for col in self.concentration_columns if 'INF' in col]
        self.group_columns = [col for col in self.data.columns if col not in self.concentration_columns]
        self.replace_nan_lq()
        self.INF,self.EFF = self.separate_dataframes()
        
    def replace_nan_lq(self):
        ql_column = [col for col in self.data.columns if col.startswith('QL')][0]
        for col in self.concentration_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors = 'coerce')
            
            #use mask to replace nans with data from ql
            nan_mask = self.data[col].isna()
            self.data[col].loc[nan_mask] = self.data[ql_column].loc[nan_mask].values

    def separate_dataframes(self):
        inf_columns = self.group_columns + [col for col in self.data.columns if 'INF' in col]
        eff_columns = self.group_columns + [col for col in self.data.columns if 'EFF' in col]
        INF_df = self.data[inf_columns].copy()
        EFF_df = self.data[eff_columns].copy()
        return INF_df,EFF_df
    
    def df_substance_transpose(self,INF = True,col_ind = None):
        
        """
        Extract certain columns from DataFrame by index, and transpose Df.
        Columns selected by index; i.e. [2,5] selects second and fifth columns
        Original index (chemicals) becomes columns names
        Columns ([2,5] for example) become index
        """
        
        if INF:
            df = self.INF.copy()
            conc_cols = self.INF_conc_cols
        else:
            df = self.EFF.copy()
            conc_cols = self.EFF_conc_cols
            
        if col_ind == None:
            cols = [col for col in df.columns if col not in self.group_columns]
        else:
            cols = df.columns[col_ind].values.tolist()
            
        #scale data by last column
        df[conc_cols] = np.transpose(df[conc_cols].values.T/df[conc_cols].values.T[-1])
        data = df[cols].values.T
        df_ = pd.DataFrame(data,columns = df['Name'],index = cols)
            
        return df_
    
def plot_groupedbarplot(df):
    """ for influent or effluent"""
    custom_palette = ['#8c564b','#9467bd','#e377c2']
    dummy_palette = ['#6f4539','#775e9b','#ffffff','#a67d70','#b07fd1','#ff8db8']
    sns.set(style='whitegrid')
    df_seaborn = df.reset_index().melt(id_vars='index', var_name='Chemical', value_name='Value')
    
    fig = plt.figure(figsize=(16, 8))
    axe = fig.add_axes([0.1,0.1,0.8,0.8])
    sns.barplot(x='Chemical', y='Value', hue='index', ax = axe, data=df_seaborn, palette=dummy_palette)
    axe.set_title('Percentile of weighted average concentrations (WAC) relative to median WAC at the influent (INF) and effluent (EFF)',fontsize = 17)
    axe.set_xlabel('Chemical',fontsize = 16)
    axe.set_ylabel('Percentile WAC to median ratio',fontsize = 16)
    axe.set_yscale('log')
    axe.set_xticklabels(axe.get_xticklabels(),rotation=30,fontsize = 15)
    plt.show()

if __name__ == '__main__':
    
    # Usage (data in current directory)
    file_path = 'RSDE_table1314data.csv'
    RSDE = DataProcessor(file_path)
    
    #convert data to plotted form
    df_INF = RSDE.df_substance_transpose(col_ind = [2,3])
    df_EFF = RSDE.df_substance_transpose(INF = False,col_ind = [2,3])
    
    #dummy separation between influent and effluent
    df_dummy = pd.DataFrame(data = np.zeros((1,df_INF.shape[1])),columns = df_INF.columns,index = [' '])
    df_INF_EFF = pd.concat([df_INF,df_dummy,df_EFF])
    plot_groupedbarplot(df_INF_EFF)