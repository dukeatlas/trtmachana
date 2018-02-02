#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file contains functions for preprocessing data
"""

import logging
import numpy as np
import uproot
import pandas as pd
import sklearn.utils as sku
from sklearn.model_selection import train_test_split
import collections

_logger = logging.getLogger(__name__)

def create_df(file_name, tree_name, branch_list, max_entries=-1, shuffle=False):
    """Creates a pandas dataframe from a list of branches

    Given the file name, the tree name, and the branch list, we can
    create a pandas.DataFrame which houses our ROOT data.

    Parameters
    ----------
    file_name : str
        ROOT TFile name
    tree_name : str
        ROOT TTree name which is saved in the TFile
    branch_list : list
        List of branches to store in the pandas.DataFrame (the columns)
    max_entries : int
        Maximum number of rows in the DataFrame
    shuffle : bool
        if True, shuffle the data before (uses sklearn.utils.shuffle)

    Returns
    -------
    pandas.DataFrame
        Returns the constructed pandas DataFrame

    """
    tree   = uproot.open(file_name)[tree_name]
    nparrs = collections.OrderedDict()
    for bn in branch_list:
        nparrs[bn] = tree.array(bn)
        if bn == 'p':
            nparrs[bn] /= 1000.0
    df = pd.DataFrame.from_dict(nparrs)
    if shuffle:
        column_names = df.columns.values.tolist()
        shuffled = sku.shuffle(df.as_matrix())
        df = pd.DataFrame(shuffled,columns=column_names)
    if max_entries > 0:
        return df[:max_entries]
    return df

def create_df_tts_scale(sig_file_name, sig_tree_name, bkg_file_name, bkg_tree_name,
                        branch_list, sig_n=-1, bkg_n=-1, shuffle=False, test_size=0.4,
                        scale_style='default'):
    """Creates signal and background dataframes and training and testing samples and scale.

    Given the signal and background files/trees and set of branches,
    create dataframes as well as a set of testing and training
    matrices/vectors. This a wrapper around using
    trtbrain.create_df along with scikit-learn's
    train_test_split. It also will, by default, scale all variables
    using the "default" method which scales all training variables to
    the range [0,1].

    Parameters
    ----------
    sig_file_name : str
        name of the file containing the signal tree
    sig_tree_name : str
        name of the tree for signal (must be in in the signal file)
    bkg_file_name : str
        name of the file containing the background tree
    bkg_tree_name : str
        name of the tree for the background (must be in the background file)
    branch_list: list
        list of branches which create the columns of the matrix (the "features")
    sig_n : int
        the maximum number of signal events to use
    bkg_n : int
        the maximum number of background events to use
    shuffle : bool
        shuffle the rows
    test_size : float
         the fraction of entries to use for testing
    scale_style : str or dict
         if str, all feature will be scaled with that scaling method
         if dict, keys must be row numbers (feature index), value must be a method
         available methods:
         'default' : scale to range [0,1],
         'symmetric' : scale  to range  [-1,1]
         'standardize' : subtract mean, divide by std dev
         'leave' : do nothing

    Returns
    -------
    df_sig : pandas.DataFrame
        the signal data frame (unscaled)
    df_bkg : pandas.DataFrame
        the background data frame (unscaled)
    X_train : numpy.ndarray
        a matrix (n_samples x n_features) of a mixture of signal and background
        events for training
    X_test : numpy.ndarray
        a matrix (n_samples x n_features) of a mixure of signal and background
        events for testing
    y_train : numpy.ndarray (n_samples x 1) of a mixture of signal (1) and background (0)
        events for training
    y_test : numpy.ndarray (n_samples x 1) of a mixture of signal (1) and background (0)
        events for testing

    """

    def make_sig_bkg(sig_file_name, sig_tree_name, bkg_file_name, bkg_tree_name,
                     branch_list, sig_n=-1, bkg_n=-1, shuffle=False):
        sig_df = create_df(sig_file_name,sig_tree_name,branch_list,max_entries=sig_n,shuffle=shuffle)
        bkg_df = create_df(bkg_file_name,bkg_tree_name,branch_list,max_entries=bkg_n,shuffle=shuffle)
        sig_mtx = sig_df.as_matrix()
        bkg_mtx = bkg_df.as_matrix()
        X = np.concatenate((sig_mtx,bkg_mtx))
        y = np.concatenate((np.ones(sig_mtx.shape[0]),
                            np.zeros(bkg_mtx.shape[0])))
        return (sig_df, bkg_df, X, y)

    df_sig, df_bkg, X, y = make_sig_bkg(sig_file_name,sig_tree_name,
                                        bkg_file_name,bkg_tree_name,
                                        branch_list,sig_n,bkg_n,shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)

    def scale_to_range(train,test,column,a=0,b=1):
        maximum = train[:,column].max()
        minimum = train[:,column].min()
        mmdiff = maximum - minimum
        train[:,column] = (b-a)*(train[:,column] - minimum)/(mmdiff) + a
        test[:,column]  = (b-a)*(test[:,column] - minimum)/(mmdiff) + a
        return dict(minimum=minimum,maximum=maximum,a=a,b=b)

    def standardization(train,test,column):
        mean, std = train[:,column].mean(), train[:,column].std()
        train[:,column] = (train[:,column] - mean)/std
        return dict(mean=mean,std=std)

    if isinstance(scale_style,str):
        for i in range(X_train.shape[1]):
            if scale_style == 'default':
                _ = scale_to_range(X_train,X_test,i)
            elif scale_style == 'symmetric':
                _ = scale_to_range(X_train,X_test,i,-1,1)
            elif scale_style == 'standardize':
                _ = standardization(X_train,X_test,i)
            elif scale_style == 'leave':
                continue

    if isinstance(scale_style,dict):
        for k, v in scale_style.items():
            if v == 'leave':
                continue
            elif v == 'default':
                _ = scale_to_range(X_train,X_test,k)
            elif v == 'symmetric':
                _ = scale_to_range(X_train,X_test,k,-1,1)
            elif v == 'standardize':
                _ = standardization(X_train,X_test,k)
            else:
                exit('bad scale style: '+v)

    return (df_sig, df_bkg, X_train, X_test, y_train, y_test)
