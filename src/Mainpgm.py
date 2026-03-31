# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 22:06:31 2024

@author: spoon
"""

import pandas as pd

df = pd.read_csv('diabetes1.csv')

# General
print(df.head(25))
print(df.tail())
print(df.sample(6))
print(df.shape)

# #Statistics
print(df.describe())
print(df.info())
print(df.corr())
print(df.memory_usage())

# Selection of Data
print(df[["BMI","Age"]])

# Data Preprocessing 
print(df.isnull())
print(df.dropna())
print(df.fillna(100))
