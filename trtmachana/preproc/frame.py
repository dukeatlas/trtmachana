#!/usr/bin/env python

import numpy as np
import uproot
import sklearn.utils as sku
import pandas as pd

def create_df(file_name, tree_name, branch_list, max_entries=-1, shuffle=False, pmaxcut=100.0):
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
        nparrs[bn] = tree.array(bn)[c_arr]
    df = pd.DataFrame.from_dict(nparrs)
    if shuffle:
        column_names = df.columns.values.tolist()
        shuffled = sku.shuffle(df.as_matrix())
        df = pd.DataFrame(shuffled,columns=column_names)
    if max_entries > 0:
        return df[:max_entries]
    return df
