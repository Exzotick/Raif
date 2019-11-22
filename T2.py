# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:28:30 2019

@author: spitc
"""
import pandas as pd
import numpy as np
from scipy.stats import t


#1
df = pd.read_csv('/usr/local/data/transactions.txt', 
                 header=None,
                 names=['ind', 'client_id', 'sum', 'segment'],
                 index_col='ind',
                 chunksize=1e13
                 )
R_clients = set()
AF_clients = set()
R_sum = 0
R_n = 0
AF_sum = 0
AF_n = 0

for chunk in df:
    R_clients.update(set(chunk.loc[chunk['segment'] == 'R', 'client_id']))
    AF_clients.update(set(chunk.loc[chunk['segment'] == 'AF', 'client_id']))
    R_sum += chunk.loc[chunk['segment'] == 'R', 'sum'].sum()
    R_n += chunk[chunk['segment'] == 'R'].shape[0]
    AF_sum += chunk.loc[chunk['segment'] == 'AF', 'sum'].sum()
    AF_n += chunk[chunk['segment'] == 'AF'].shape[0]
print('Количество клиентов в сегменте R:', len(R_clients))
print('Количество клиентов в сегменте AF:', len(AF_clients))

#2
R_mean = R_sum/R_n    
AF_mean = AF_sum/AF_n
print('Средний объем транзакции в сегменте R: ', R_mean)
print('Средний объем транзакции в сегменте AF: ', AF_mean)

#3
df = pd.read_csv('/usr/local/data/transactions.txt', 
                 header=None,
                 names=['ind', 'client_id', 'sum', 'segment'],
                 index_col='ind',
                 chunksize=1e13
                 )
R_var = 0    
AF_var = 0
for chunk in df:
    R_var += (np.power(R_mean - chunk.loc[chunk['segment']=='R', 'sum'], 2).sum())/R_n
    AF_var += (np.power(AF_mean - chunk.loc[chunk['segment']=='AF', 'sum'], 2)).sum()/AF_n

alpha = 0.1
print('90% доверительный интервал для среднего объема транзакции в сегменте R:\n',
        (R_mean - t.ppf(1-alpha/2, df=R_n-1)*np.sqrt(R_var/R_n), 
         R_mean + t.ppf(1-alpha/2, df=R_n-1)*np.sqrt(R_var/R_n))
     )
print('90% доверительный интервал для среднего объема транзакции в сегменте AF:\n',
        (AF_mean - t.ppf(1-alpha/2, df=AF_n-1)*np.sqrt(AF_var/AF_n), 
         AF_mean + t.ppf(1-alpha/2, df=AF_n-1)*np.sqrt(AF_var/AF_n)
        )
     )
        
#4
comb_var = R_var/R_n + AF_var/AF_n
t_stat = (R_mean - AF_mean) / np.sqrt(comb_var)
deg = int(
            (R_var/R_n + AF_var/AF_n)**2 / (((R_var/R_n)**2) / 
             (R_n-1) + ((AF_var/AF_n)**2)/(AF_n-1))
        )
p_val = 2 * t.sf(np.abs(t_stat), deg)
if p_val < alpha:
    print('Гипотеза о равенстве средних отвергается')
else:
    print('Гипотеза о равенстве средних принимается')