from __future__ import print_function
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from sklearn.metrics.cluster import normalized_mutual_info_score
# better but runs out of memory
# from sklearn.metrics.cluster import adjusted_mutual_info_score 
import sklearn.utils as sku
from sklearn.utils import shuffle
import scipy.interpolate as spi
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import collections
from collections import OrderedDict
from datetime import datetime
import pickle
import os
import re


# Setup p, m variables that need to be divided to get to GeV units
pm_vars = ['p', 'pT', 'lep_pT']

########################################################
# print out our times nicely
def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)

########################################################
# Define a function to create the output dir
# If it already exists don't crash, otherwise raise an exception
# Adapted from A-B-B's response to http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
# Note in python 3.4+ 'os.makedirs(output_path, exist_ok=True)' would handle all of this...
def make_path(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise Exception('Problem creating output dir %s !!!\nA file with the same name probably already exists, please fix the conflict and run again.' % output_path)

########################################################
# 
def train_or_load(fname, default_to_load):
    if os.path.isfile(fname):

        if default_to_load:
            print("Model found on disk, loading by default")
            train_or_load = 'y'
        else:
            train_or_load = raw_input('Model found on disk, load and continue (y)? If (n) will re-train: ')
            assert isinstance(train_or_load, str);

            if train_or_load == 'Y' or train_or_load == 'y':
                train_or_load = 'y'
                print('\nLoading model')
            else:
	             train_or_load = 'n'
	             print('\nRe-training model')

    else:
        print('Model NOT found on disk, training')
        train_or_load = 'n'

    return train_or_load

########################################################
# 
def create_df(file_name, tree_name, branch_list, max_entries=-1, shuffle=False):
    tree   = uproot.open(file_name)[tree_name]
    nparrs = collections.OrderedDict()
    for bn in branch_list:
        nparrs[bn] = tree.array(bn)
        if bn in pm_vars:
            nparrs[bn] /= 1000.0
    df = pd.DataFrame.from_dict(nparrs)
    
    # TODO WARNING MC is weird, cut to a max pT of 200 GeV by hand!!!
    df = df[(df.pT <= 200)]
    
    if shuffle:
        column_names = df.columns.values.tolist()
        shuffled = sku.shuffle(df.as_matrix())
        df = pd.DataFrame(shuffled,columns=column_names)
    if max_entries > 0:
        return df[:max_entries]
    return df

########################################################
# 
def create_df_tts_scale(sig_file_name, sig_tree_name, bkg_file_name, bkg_tree_name,
                        branch_list, sig_n=-1, bkg_n=-1, shuffle=False, test_size=0.2,
                        scale_style='default'):

    def make_sig_bkg(sig_file_name, sig_tree_name, bkg_file_name, bkg_tree_name,
                     branch_list, sig_n=-1, bkg_n=-1, shuffle=False):
        df_sig = create_df(sig_file_name,sig_tree_name,branch_list,max_entries=sig_n,shuffle=shuffle)
        df_bkg = create_df(bkg_file_name,bkg_tree_name,branch_list,max_entries=bkg_n,shuffle=shuffle)
        sig_mtx = df_sig.as_matrix()
        bkg_mtx = df_bkg.as_matrix()
        X = np.concatenate((sig_mtx,bkg_mtx))
        y = np.concatenate((np.ones(sig_mtx.shape[0]),
                            np.zeros(bkg_mtx.shape[0])))
        return (df_sig, df_bkg, X, y)

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


########################################################
# 
def create_fixed_test_shuffled_train_and_scale(sig_file_name, sig_tree_name, bkg_file_name, bkg_tree_name,
                        branch_list, sig_n=-1, bkg_n=-1, test_size=0.2,
                        scale_style='default', rnd_seed=7):

    df_sig_all_rows = create_df(sig_file_name,sig_tree_name,branch_list,max_entries=sig_n,shuffle=False)
    df_bkg_all_rows = create_df(bkg_file_name,bkg_tree_name,branch_list,max_entries=bkg_n,shuffle=False)

    min_m_sig_bkg_all_rows = min(len(df_sig_all_rows.index), len(df_bkg_all_rows.index)) 
    test_m = int(test_size*min_m_sig_bkg_all_rows)

    df_sig_test  = df_sig_all_rows.iloc[:test_m,:] # Take first test_m rows, unshuffled so they are always the same run after run
    df_sig_train = df_sig_all_rows.iloc[test_m:,:] # Take remaining rows to the end, unshuffled so they are always the same run after run

    df_bkg_test  = df_bkg_all_rows.iloc[:test_m,:] # Take first test_m rows, unshuffled so they are always the same run after run
    df_bkg_train = df_bkg_all_rows.iloc[test_m:,:] # Take remaining rows to the end, unshuffled so they are always the same run after run

    def dfs_to_mtxs(df_sig, df_bkg, do_shuffle=True):
        sig_mtx = df_sig.as_matrix()
        bkg_mtx = df_bkg.as_matrix()
        X = np.concatenate((sig_mtx, bkg_mtx))
        y = np.concatenate((np.ones(sig_mtx.shape[0]), np.zeros(bkg_mtx.shape[0])))

        if do_shuffle:
            X, y = shuffle(X, y, random_state=rnd_seed)

        return X, y

    X_train, y_train = dfs_to_mtxs(df_sig_train, df_bkg_train, do_shuffle=True)
    X_test, y_test = dfs_to_mtxs(df_sig_test, df_bkg_test, do_shuffle=True)

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
            else:
                exit('bad scale style: '+scale_style)

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

    return (df_sig_all_rows, df_bkg_all_rows, X_train, X_test, y_train, y_test)

########################################################
# 
class eprob_roc_generateor(object):
    def __init__(self, sighist, bkghist, primary_axis='x',interpolate=False,
                 xbinrange=(1,1), ybinrange=(1,1), zbinrange=(1,1), npbinning=np.linspace(0.0,1.0,100)):

        sigPtConstruct = []
        bkgPtConstruct = []

        if isinstance(sighist,np.ndarray) and isinstance(bkghist,np.ndarray):
            binning = npbinning
            sigHist, sigEdges = np.histogram(sighist,bins=binning)
            bkgHist, bkgEdges = np.histogram(bkghist,bins=binning)
            self.sigInteg, self.bkgInteg = float(np.sum(sigHist)), float(np.sum(bkgHist))
            for i in range(len(sigHist)):
                x = float(np.sum(sigHist[i+1:]))/self.sigInteg
                y = float(np.sum(bkgHist[i+1:]))/self.bkgInteg
                sigPtConstruct.append(x)
                bkgPtConstruct.append(y)

        self.sigPoints = np.array(sigPtConstruct,copy=True,dtype='d')
        self.bkgPoints = np.array(bkgPtConstruct,copy=True,dtype='d')
        self.bmax, self.bmin = self.bkgPoints.max(), self.bkgPoints.min()
        self.smax, self.smin = self.sigPoints.max(), self.sigPoints.min()

        if interpolate:
            self.interpolation = spi.interp1d(self.sigPoints,self.bkgPoints,fill_value='extrapolate')

    def tpr(self):
        return self.sigPoints

    def fpr(self):
        return self.bkgPoints

    # on.plot(self.sigPoints,self.bkgPoints,*args,**kwargs)

########################################################
# 
def slice_and_plot_all_input_vars(cut_var, nname, bins, input_variables, X_train, y_train, m_path):
    var_names = list(input_variables)
    cut_index = var_names.index(cut_var)

    for i in range(len(bins)-1):
        lbin = bins[i]
        rbin = bins[i+1]

        selection = np.where( (lbin <= X_train[:,cut_index]) & (X_train[:,cut_index] < rbin))

        name = '{1} < {0} < {2}'.format(nname, lbin, rbin)
        fname = 'all_input_vars_sliced_{0}_{1}_{2}'.format(cut_var, lbin, rbin)

        plot_all_input_vars(input_variables, X_train[selection], y_train[selection], m_path, name, fname, True)

########################################################
# 
def plot_all_input_vars(input_variables, X_train, y_train, m_path, name='', fname='all_input_vars', fix_xlim=False):
    var_names = list(input_variables)
    nvars = len(var_names)
    
    nwidth = int(np.floor(np.sqrt(nvars)))
    nheight = int(np.ceil(np.sqrt(nvars)))

    fig = plt.figure('all_input_vars')

    vsize = 10 # inches
    aspect_ratio = 1.0
    fig.set_size_inches(aspect_ratio*vsize, vsize)
    
    gs = gridspec.GridSpec(nheight, nwidth)
    gridspec_list = [[i,j] for i in range(nheight) for j in range(nwidth)]

    for nvar, var in enumerate(var_names):
        ax = plt.subplot(gs[gridspec_list[nvar][0], gridspec_list[nvar][1]])

        ax.hist([X_train[:,nvar][y_train>0.5],
                 X_train[:,nvar][y_train<0.5]],
                label=['Electrons','Muons'],
                bins=30, histtype='step', normed=True)
        
        ax.set_xlabel(input_variables[var][0])
  
        if fix_xlim: 
            scale_style = input_variables[var][1]
            if scale_style == 'default':
                ax.set_xlim(0.,1.)
            elif scale_style == 'symmetric':
                ax.set_xlim(-1.,1.)
            elif scale_style != 'standardize' and scale_style != 'leave':
                print("Unrecognized scale_style {0}, leaving axis auto!".format(scale_style))
 
    # will only get the handles and lables for the last ax, but that is what we want actually
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper right',)

    plt.figtext(0.5, 0.05, name, ha='center', va='center', size=18)
     
    plt.tight_layout()
    make_path(m_path)
    fig.savefig(m_path+'/'+fname+'.pdf')
    plt.show()
    fig.clf()

########################################################
# 
def plot_scale_example(fname,tname,m_path,vname,nname,a=0,b=1):
    arr = uproot.open(fname)[tname].array(vname,np.float32)# ,dtype=np.float32) # dtype= deprecated?
    if vname in pm_vars:
        arr /= 1000.0
        
        # TODO WARNING MC is weird, cut to a max pT of 200 GeV by hand!!!
        selection_array = arr < 200
        arr = arr[selection_array]

    ma = arr.max()
    mi = arr.min()
    arr_scaled = (b-a)*(arr-mi)/(ma-mi)+a

    make_path(m_path)

    fig, ax = plt.subplots()
    ax.hist(arr,bins=50,histtype='step',normed=True)
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('Raw '+nname)
    fig.savefig(m_path+'/unscaled_'+vname+'.pdf')

    fig, ax = plt.subplots()
    ax.hist(arr_scaled,bins=50,histtype='step',normed=True)
    ax.set_ylabel('Arb. Units')
    ax.set_xlabel('Feature scaled '+nname)
    fig.savefig(m_path+'/scaled_'+vname+'.pdf')

########################################################
# 
def plot_acc_loss_vs_epoch(history_dict, name, nname, m_path, val_or_test = 'Test', do_acc = True, do_loss = False):
    expected_keys = ['acc', 'loss', 'val_acc', 'val_loss', 'acc_std', 'loss_std', 'val_acc_std', 'val_loss_std']
    keys = history_dict.keys()
    
    if not (set(keys) <= set(expected_keys)):
        print("WARNING Unknown keys in history_dict!\nAll keys:")
        print(keys)

    fig, ax = plt.subplots()
    
    if do_acc and 'acc' in keys:
        ax.plot(history_dict['acc'],
               lw=2, c='black', ls='-',
               label='Train Acc')

        if 'acc_std' in keys:
            ax.errorbar(range(len(history_dict['acc'])), history_dict['acc'], yerr=history_dict['acc_std'],
                        fmt='none', capsize=4, elinewidth=1.5, markeredgewidth=1.5, c='black')
 
    if do_loss and 'loss' in keys:
        ax.plot(history_dict['loss'],
               lw=2, c='black', ls='-',
               label='Train Loss')

        if 'loss_std' in keys:
            ax.errorbar(range(len(history_dict['loss'])), history_dict['loss'], yerr=history_dict['loss_std'],
                        fmt='none', capsize=4, elinewidth=1.5, markeredgewidth=1.5, c='black')
           
    if do_acc and 'val_acc' in keys:
        ax.plot(history_dict['val_acc'],
               lw=2, c='blue', ls='--',
               label=val_or_test+' Acc')

        if 'val_acc_std' in keys:
            ax.errorbar(range(len(history_dict['val_acc'])), history_dict['val_acc'], yerr=history_dict['val_acc_std'],
                        fmt='none', capsize=4, elinewidth=1.5, markeredgewidth=1.5, c='blue')


    if do_loss and 'val_loss' in keys:
        ax.plot(history_dict['val_loss'],
               lw=2, c='magenta', ls='--',
               label=val_or_test+' Loss')

        if 'val_loss_std' in keys:
            ax.errorbar(range(len(history_dict['val_loss'])), history_dict['val_loss'], yerr=history_dict['val_loss_std'],
                        fmt='none', capsize=4, elinewidth=1.5, markeredgewidth=1.5, c='magenta')



    fname = ''

    acc_str = ''
    if do_acc:
        acc_str = ' Accuracy'
        fname = 'accuracy'

    loss_str = ''
    if do_loss:
        if acc_str != '':
            loss_str = " and"
            fname += '_'
        loss_str += ' Loss'
        fname += 'loss'

    ax.set_title(name+acc_str+loss_str)
    ax.set_ylabel(acc_str+loss_str)
    ax.set_xlabel('Epoch')
    plt.legend()
    make_path(m_path)
    fig.savefig(m_path+'/'+fname+'_'+nname+'.pdf')

########################################################
# 
def plot_classifier_1D_output(el, mu, name, nname, m_path
                             # , title=''
                             ):
    fig, ax = plt.subplots()
    ax.hist(el,bins=50,histtype='step',normed=True,label='Electrons')
    ax.hist(mu,bins=50,histtype='step',normed=True,label='Muons')
    # ax.hist([el,mu],bins=50,histtype='step',normed=True,label=['Electrons','Muons'])
    ax.set_xlabel(name+' output')
    ax.set_ylabel('Arb. Units')
    ax.legend(loc='upper left')
#    ax.text(title)
    make_path(m_path)
    fig.savefig(m_path+'/classifier_1D_output_'+nname+'.pdf')

########################################################
# 
def plot_roc(model_lists, m_path):
    fig, ax = plt.subplots()

    fname = ''
    for model in model_lists:
        # tpr, fpr, name, nname, color, linestyle
        
        tpr = model[0]
        fpr = model[1]

        fname += '_'+model[3]
        
        ax.plot(tpr, fpr,
                lw=2, c=model[4], ls=model[5],
                label=('%s ROC, Area: %.3f' % (model[2], auc(tpr,fpr)))
               )
    
    ax.grid()
    ax.legend(loc='upper left')
    ax.set_xlim([.4,1])
    ax.set_xlabel('True positive')
    ax.set_ylabel('False positive')
    make_path(m_path)
    fig.savefig(m_path+'/roc'+fname+'.pdf')

########################################################
# 
def mutual_info_plot(var_names_dict, df, name, nname, m_path):

    # setup
    cols = [col for col in df]
    col_names = [var_names_dict[col] for col in cols]
    ncols = len(cols)

    # compute mi's
    norm_mi = np.zeros((len(cols),len(cols)))
   
    for i,col1 in enumerate(cols):
        for j,col2 in enumerate(cols[:i]):
            raw_matrix = df.as_matrix([col1,col2])
            norm_mi[i][j] = normalized_mutual_info_score(raw_matrix[:,0], raw_matrix[:,1])

    # mask upper right duplicates
    mask = np.triu(np.ones(norm_mi.shape, dtype=int))
    norm_mi_masked = np.ma.masked_array(norm_mi, mask=mask)

    # now plot
    figsize = 10
    digit_size = 12.5
    if ncols > 15:
        figsize = 15
        digit_size = 11

    fig = plt.figure(figsize=(figsize, figsize))
    ax = fig.add_subplot(111)

    norm = mpl.colors.Normalize(vmin=0.,vmax=1.)
    cmap='viridis'

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.1 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    img = ax.imshow(norm_mi_masked, cmap=cmap, norm=norm)
    cb = plt.colorbar(img, cmap=cmap, norm=norm, cax=cax)

    # annotate
    for (j,i),value in np.ndenumerate(norm_mi):
        if(i<j):
            # https://stackoverflow.com/questions/11010683/how-to-have-negative-zero-always-formatted-as-positive-zero-in-a-python-string/36604981#36604981
            value_str = re.sub(r"^-(0\.?00*)$", r"\1", "%.2f" % value)
            ax.text(i,j,value_str, ha='center', va='center', color='fuchsia', size=digit_size)

    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(ncols))

    ax.set_xticklabels(col_names, rotation='vertical')
    ax.set_yticklabels(col_names)

    plt.figtext(0.5, 0.89, name, ha='center', va='center', size=18)

    plt.figtext(0.96, 0.8, "(Dependent)", rotation='vertical', ha='center', va='center', size=16)
    plt.figtext(0.96, 0.22, "(Independent)", rotation='vertical', ha='center', va='center', size=16)
    plt.figtext(0.96, 0.5, "NMI", rotation='vertical', ha='center', va='center', size=18)

    make_path(m_path)
    fig.savefig(m_path+'/mutual_information_'+nname+'.pdf')

########################################################
# 
def process_kfold_hist_elements(accs, losses, val_accs, val_losses, plots_path, name = 'NN (kfold)', nname = 'nn_kfold'):

    accs_mtx = np.array(accs)
    losses_mtx = np.array(losses)
    val_accs_mtx = np.array(val_accs)
    val_losses_mtx = np.array(val_losses)

    def print_final_mean_std(mtx, name):
        mean = np.mean(mtx[:,-1])
        std = np.std(mtx[:,-1])
        print("{}: {:.3f} +/- {:.3f}".format(name, mean, std))
        
    print_final_mean_std(accs_mtx,"Acc")
    print_final_mean_std(losses_mtx,"Loss")
    print_final_mean_std(val_accs_mtx,"Validation Acc")
    print_final_mean_std(val_losses_mtx,"Validation Loss")
    
    def dicts_of_col_mean_std(mtx,name):
        means = np.zeros(mtx.shape[1])
        stds = np.zeros(mtx.shape[1])
    
        for i in range(mtx.shape[1]):
            means[i] = np.mean(mtx[:,i])
            stds[i] = np.std(mtx[:,i])

        return {name:list(means), name+'_std':list(stds)}

    acc_dict = dicts_of_col_mean_std(accs_mtx,'acc')
    loss_dict = dicts_of_col_mean_std(losses_mtx,'loss')
    val_acc_dict = dicts_of_col_mean_std(val_accs_mtx,'val_acc')
    val_loss_dict = dicts_of_col_mean_std(val_losses_mtx,'val_loss')
    
    hist_dict_model_kfold_mean_std = acc_dict.copy()
    hist_dict_model_kfold_mean_std.update(loss_dict)
    hist_dict_model_kfold_mean_std.update(val_acc_dict)
    hist_dict_model_kfold_mean_std.update(val_loss_dict)

    # print(hist_dict_model_kfold_mean_std)
           
    plot_acc_loss_vs_epoch(hist_dict_model_kfold_mean_std, name+' Mean', nname+'_mean', plots_path, 'Validation', True, False)
    plot_acc_loss_vs_epoch(hist_dict_model_kfold_mean_std, name+' Mean', nname+'_mean', plots_path, 'Validation', False, True)
