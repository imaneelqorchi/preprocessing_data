# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Spyder Editor

This is a temporary script file.
"""

#importing libraries
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


    
 
# function that import data and replace missing values with nan/ author: Imane EL QORCHI
def import_data(df_link):
    missing_values = ["n/a", "na", "--"]
    if "csv" in df_link:
        df = pd.read_csv(df_link, na_values = missing_values)
    else:
        df = pd.read_csv(df_link, na_values = missing_values, header = None)
        df = pd.DataFrame(df)
     
    df = df.replace('\t?', np.nan)
    df = df.replace('?', np.nan)
    return df
  
    

# function to get a list of categorical columns if there's any
def get_categorical(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    categorical_cols = list(set(cols) - set(num_cols))
    return categorical_cols


# function that drop column if there's too much missing values/ author: Imane EL QORCHI
def drop_col(df):
    cols = df.columns
    perc_nan = (df.isna().sum()/len(df))*100
    # if for a feature the percentage of missing values is higher than 50 we drop the column
    for i in range(len(perc_nan)):
        if perc_nan[i] > 50:
            drop_feat = cols[i]
            df = df.drop(drop_feat, 1)
            
    return df


# this function fix types into columns :transform digit string to real numerical values and delete tab from values / author: Imane EL QORCHI 
def fix_type(df):
    df = df.applymap(lambda x: x.replace("\t","") if type(x) == str else x)
    df = df.applymap(lambda x: x.replace(" ","") if type(x) == str else x)
    df = df.applymap(lambda x: float(x) if isinstance(x,str) and x.replace(".","").isdigit() else x) 
    return df


# how to handle missing numerical values/ author: Imane EL QORCHI
def handle_digit(df):
    means = round(df.mean(),3)
    df = df.fillna(means)
    return df


#the most frequent category/ author: Imane EL QORCHI
def handle_category(df):
    categorical_cols = get_categorical(df)
    if(len(categorical_cols)!=0):
        for cat in categorical_cols:
            df[cat] = df[cat].transform(lambda x: x.fillna(x.mode()[0]))
    return df



#handle missing values for categorical columns/ author: Imane EL QORCHI
#transform categorical columns to hot vectors
def hot_encode(df):
    categorical_cols = get_categorical(df)
    if(len(categorical_cols)!=0 ):
        lb_make = LabelEncoder()
        for cat in categorical_cols:
            df[cat+"_code"] = lb_make.fit_transform(df[cat])
            df = df.drop(cat,axis=1)
    return df


# final function to pre process the data/ author: Imane EL QORCHI
def pre_processing(df_link):
    df = import_data(df_link)
    df = drop_col(df)
    df = fix_type(df)
    df = handle_digit(df)
    df = handle_category(df)
    df = hot_encode(df)
    return df
