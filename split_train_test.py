#!/usr/bin/env python
# -*- coding: utf-8 -*-


import warnings
import argparse
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import pandas as pd


def split(finput, test_rate):
    dataset = pd.read_csv(finput)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    column_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_rate, random_state = 10)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv("train_" + finput, index=False)
    test.to_csv("test_" + finput, index=False)
    return


#############################################################################    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='csv format file, e.g., dataset.csv')
    parser.add_argument('-r', '--test_rate', help='test rate, e.g., 0.2, 0.3')
    args = parser.parse_args()
    finput = str(args.input)
    if args.test_rate is None:
        test_rate = 0.2
    else:
        test_rate = float(args.test_rate)
    split(finput, test_rate)
#############################################################################] 
