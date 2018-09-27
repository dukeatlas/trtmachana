
# coding: utf-8

# ## Setup

# In[ ]:


get_ipython().magic(u'run code.py')
get_ipython().magic(u'matplotlib inline')
time_all_start = datetime.now()


# In[ ]:


# file and tree names

# MC
sig_file_name = 'data/all_el.root'
bkg_file_name = 'data/all_mu.root'
sig_tree  = 'electron_mc'
bkg_tree  = 'muon_mc'

# data
# sig_tree  = 'electron_tags'
# bkg_tree  = 'muons'


# In[ ]:


# other settings

fit_verbose = 1

# max_epochs = 100
max_epochs = 20
max_epochs_model_default = 50

default_to_load = True

rnd_seed = 7

plot_mi = True


# ## Setup variables to train on

# In[ ]:


input_variables = OrderedDict([
    ('p',['$p$','default']),
    ('pT',['$p_{\mathrm{T}}$','default']),
    ('eta',['$\eta$','symmetric']),
    ('nTRThitsMan',['nTRT','default']),
    ('nTRTouts',['nTRT outs','default']),
    ('fHTMB',['Fraction HTMB','default']),
    ('fAr',['Fraction Ar','default']),
    ('trkOcc',['Track Occ.','default']),
    ('sumToTsumL',['$\sum\mathrm{ToT} / \sum L$','default']),
    ('PHF',['PHF','default']),
    ('sumL',['sumL','default']),
    ('eProbHT',['eProbHT','default']),
#    ('NhitsdEdx',['NhitsdEdx','default']), # TODO invalid values
#    ('phi',['$\phi$','default']), # Didn't add anything, don't expect it to physically though
])

# all the hit vars, arrays of length 40 for use in RNN LSTM

###########
# uninteresting vars - info is already included elsewhere
# nTRThits, nArhits, nXehits, nHThitsMan, nPrechitsMan, sumToT


# In[ ]:


var_names_dict = {k:v[0] for (k,v) in input_variables.items()}


# In[ ]:


var_comb_dir = ''
for i,v in enumerate(input_variables.keys()):
    if i != 0: var_comb_dir += '_'
    var_comb_dir += v
plots_path = 'plots/'+var_comb_dir
models_path = 'models/'+var_comb_dir
make_path(plots_path)
make_path(models_path)


# In[ ]:


# df_sig, df_bkg, X_train, X_test, y_train, y_test = create_df_tts_scale(
#    sig_file_name, sig_tree, bkg_file_name, bkg_tree,
#    list(input_variables),
#    test_size=0.2,
#    # test_size=0.333333,
#    # sig_n=50000,
#    # bkg_n=50000,
#    shuffle=True,
#    scale_style={i:v[1] for i,(_,v) in enumerate(input_variables.items())}
# )

# TODO WARNING df_'s are not weighted!

df_sig, df_bkg, X_train, X_test, y_train, y_test = create_fixed_test_shuffled_train_and_scale(
    sig_file_name, sig_tree, bkg_file_name, bkg_tree,
    list(input_variables),
    test_size=0.2,
    # test_size=0.333333,
    # sig_n=50000,
    # bkg_n=50000,
    scale_style={i:v[1] for i,(_,v) in enumerate(input_variables.items())},
    rnd_seed = rnd_seed
)


# In[ ]:


input_ndimensions = X_train.shape[1]
leptons_m = y_train.shape[0]

print("Training on m = %.2g leptons\nTesting on %.2g leptons (50/50 sig/bkg)\nNumber of input variables n = %d" % (leptons_m, y_test.shape[0], input_ndimensions))


# In[ ]:


if False:
    print(df_sig.head(3))
    print(X_train.shape)
    print(X_train[:2])


# ## Find weights to normalize pT, eta distributions

# In[ ]:


w_train, w_test = weight_pT_eta_uniform(input_variables, X_train, y_train, X_test, y_test)


# In[ ]:


val_data=(X_test, y_test, w_test) # Always use the same test data for non-kfold runs


# In[ ]:


plot_pT_eta(input_variables, X_train, y_train, plots_path, name='$p_{\mathrm{T}}$ vs $\eta$', fname='pT_eta_hist', w = None)
plot_pT_eta(input_variables, X_train, y_train, plots_path, name='$p_{\mathrm{T}}$ vs $\eta$, Weighted', fname='pT_eta_hist_weighted', w = w_train)


# ## Create eProbabilityHT curves

# In[ ]:


# TODO this technically has the issue with the high pT events, but shouldn't actually be visible on plots
sig_eprob = uproot.open(sig_file_name)[sig_tree].array('eProbHT')
bkg_eprob = uproot.open(bkg_file_name)[bkg_tree].array('eProbHT')

m_eprob = min(sig_eprob.shape[0], bkg_eprob.shape[0])
sig_eprob = sig_eprob[:m_eprob]
bkg_eprob = bkg_eprob[:m_eprob]
print('Using %.2g sig el, %.2g bkg for eProbHT' % (sig_eprob.shape[0], bkg_eprob.shape[0]))

roc_eprob_obj = eprob_roc_generateor(sig_eprob, bkg_eprob)

roc_eprob = [roc_eprob_obj.tpr(), roc_eprob_obj.fpr(), 'eProbHT', 'eprob', 'black', '-']


# ## Plot input variables

# In[ ]:


plot_all_input_vars(input_variables, X_train, y_train, plots_path, 'Unweighted')
plot_all_input_vars(input_variables, X_train, y_train, plots_path, 'Weighted', 'all_input_vars_weighted', False, w_train)


# In[ ]:


# plot_scale_example(sig_file_name,sig_tree,plots_path,'p','$p$ [GeV]'
# plot_scale_example(sig_file_name,sig_tree,plots_path,'pT','$p_{\mathrm{T}}$ [GeV]')


# # BDT (sklearn)

# In[ ]:


bdt_m = min(100000, y_train.shape[0])
# bdt_m = None # all


# In[ ]:


fname_bdt1 = 'bdt1'
train_load_bdt1 = train_or_load(models_path+'/'+fname_bdt1+'.pkl', default_to_load)


# In[ ]:


if train_load_bdt1 == 'n':
    
    # create   
    bdt1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                              algorithm="SAMME",
                              n_estimators=200,
                             )
    
    # train
    train_start = datetime.now()

    # bdt1.fit(X_train,y_train, sample_weight=w_train);
    bdt1.fit(X_train[:bdt_m],y_train[:bdt_m], sample_weight=w_train[:bdt_m]);

    print(strfdelta(datetime.now()-train_start, "Training time: {hours} hours, {minutes} minutes, {seconds} seconds"))

    # save model to pickle
    joblib.dump(bdt1, models_path+'/'+fname_bdt1+'.pkl');
    
else:
    # load model from pickle
    bdt1 = joblib.load(models_path+'/'+fname_bdt1+'.pkl');


# In[ ]:


plot_classifier_1D_output(bdt1.decision_function(X_test[y_test>0.5]), # sig
                          bdt1.decision_function(X_test[y_test<0.5]), # bkg
                          'BDT', 'bdt', plots_path
                         )

fpr_bdt1, tpr_bdt1, thresholds_bdt1 = roc_curve(y_test, bdt1.decision_function(X_test))
roc_bdt1 = [tpr_bdt1, fpr_bdt1, 'BDT', 'bdt', 'green', '-.']

plot_roc([roc_eprob, roc_bdt1], plots_path)


# # SVM (sklearn)

# In[ ]:


svm_m = min(50000, y_train.shape[0])
# goes as n*svm_m*log(svm_m)
# ~1 hour, 10 minutes for 100000
# ~TODO minutes for 50000


# In[ ]:


fname_svm1 = 'svm1'
train_load_svm1 = train_or_load(models_path+'/'+fname_svm1+'.pkl', default_to_load)


# In[ ]:


if train_load_svm1 == 'n':
    
    # create
    svm1 = svm.SVC(#C=1.0, #kernel='rbf', #tol=0.001, #gamma='auto',
    probability=True,
    cache_size=1000, # default is 200 (MB)
    verbose=False);

    # train
    train_start = datetime.now()

    svm1.fit(X_train[:svm_m],y_train[:svm_m], sample_weight=w_train[:svm_m]);

    print(strfdelta(datetime.now()-train_start, "Training time: {hours} hours, {minutes} minutes, {seconds} seconds"))

    # save model to pickle
    joblib.dump(svm1, models_path+'/'+fname_svm1+'.pkl');
    
else:
    # load model from pickle
    svm1 = joblib.load(models_path+'/'+fname_svm1+'.pkl');


# In[ ]:


plot_classifier_1D_output(svm1.decision_function(X_test[y_test>0.5]), # sig
                          svm1.decision_function(X_test[y_test<0.5]), # bkg
                          'SVM', 'svm', plots_path
                         )

svm1_roc_start = datetime.now()
fpr_svm1, tpr_svm1, thresholds_svm1 = roc_curve(y_test, svm1.decision_function(X_test))
roc_svm1 = [tpr_svm1, fpr_svm1, 'SVM', 'svm', 'blue', ':']
print(strfdelta(datetime.now()-svm1_roc_start, "SVM ROC production time: {hours} hours, {minutes} minutes, {seconds} seconds"))

plot_roc([roc_eprob, roc_svm1], plots_path)


# # Keras / Tensorflow Networks

# In[ ]:


# fix random seed for reproducibility
np.random.seed(rnd_seed)

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import load_model
    from keras.callbacks import EarlyStopping


# ## Default

# In[ ]:


fname_model_default = 'model_default'
train_load_model_default = train_or_load(models_path+'/'+fname_model_default+'.h5', default_to_load)
train_load_model_default = 'n'


# In[ ]:


if train_load_model_default == 'n':
    
    # create
    model_default = Sequential()
    model_default.add(Dense(12, input_dim=input_ndimensions, activation='relu'))
    model_default.add(Dense(8, activation='relu'))
    model_default.add(Dense(1, activation='sigmoid'))

    model_default.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_default_callbacks = [] 
    model_default_callbacks.append(EarlyStopping(
        monitor='acc', min_delta=0.0002,
        # monitor='loss', min_delta=0.00002,                                                 
        patience=5,
        verbose=0,
        mode='auto'))
    
    # train
    train_start = datetime.now()
    hist_model_default = model_default.fit(X_train, y_train,
                                           epochs=max_epochs_model_default, batch_size=50,
                                           verbose=fit_verbose, validation_data=val_data,
                                           sample_weight=w_train,
                                           callbacks=model_default_callbacks);

    hist_dict_model_default = hist_model_default.history
    print(strfdelta(datetime.now()-train_start, "Training time: {hours} hours, {minutes} minutes, {seconds} seconds"))

    # save model to HDF5, history to pickle
    model_default.save(models_path+'/'+fname_model_default+'.h5')
   
    with open(models_path+'/'+fname_model_default+'_hist.pickle', 'wb') as handle:
        pickle.dump(hist_dict_model_default, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    # load model from HDF5, history from pickle
    model_default = load_model(models_path+'/'+fname_model_default+'.h5')
    
    with open(models_path+'/'+fname_model_default+'_hist.pickle', 'rb') as handle:
        hist_dict_model_default = pickle.load(handle)


# In[ ]:


plot_acc_loss_vs_epoch(hist_dict_model_default, 'NN (Default)', 'nn_default', plots_path, 'Test', True, False)
plot_acc_loss_vs_epoch(hist_dict_model_default, 'NN (Default)', 'nn_default', plots_path, 'Test', False, True)


# In[ ]:


print("model_default %s: %.2f%%" % (model_default.metrics_names[1], model_default.evaluate(X_test,y_test,verbose=0)[1]*100))

plot_classifier_1D_output(model_default.predict(X_test[y_test>0.5], verbose=0), # sig
                          model_default.predict(X_test[y_test<0.5], verbose=0), # bkg
                          'NN (Default)', 'nn_default', plots_path
                         )

fpr_model_default, tpr_model_default, thresholds_model_default = roc_curve(y_test, model_default.predict(X_test, verbose=0))
roc_model_default = [tpr_model_default, fpr_model_default, 'NN (Default)', 'nn_default', 'magenta', '--']

plot_roc([roc_eprob, roc_model_default], plots_path)


# ### Print and plot model_default structure

# In[ ]:


model_default.summary()


# In[ ]:


from keras.utils.vis_utils import plot_model
# pip install pydot

plot_model(model_default, to_file=plots_path+'/model_default.pdf', show_shapes=True, show_layer_names=True)


# ## Print input variable plots vs default NN ouptut

# In[ ]:


input_variables_with_model_default_output = input_variables.copy() 
input_variables_with_model_default_output['model_default_nn_output'] = ['NN (Default) Output', 'leave']
model_default_output_bins = [0.0, 0.05, 0.1, 0.15, 0.2,
                             0.4, 0.6,
                             0.8, 0.85, 0.9, 0.95, 1.0]

X_train_with_model_default_output = np.append(X_train, model_default.predict(X_train, verbose=0), axis=1)


# In[ ]:


slice_and_plot_all_input_vars('model_default_nn_output', 'NN', model_default_output_bins, 
                              input_variables_with_model_default_output,
                              X_train_with_model_default_output,
                              y_train, plots_path)


# ## Dropout (Larger network, higher learning rate, etc)

# In[ ]:


from keras.layers import Dropout


# In[ ]:


fname_model_dropout = 'model_dropout'
train_load_model_dropout = train_or_load(models_path+'/'+fname_model_dropout+'.h5', default_to_load)
train_load_model_dropout = 'n' # TODO


# In[ ]:


if train_load_model_dropout == 'n':
    
    # create
    model_dropout = Sequential()
    model_dropout.add(Dense(12, input_dim=input_ndimensions, activation='relu'))
    model_dropout.add(Dense(8, activation='relu'))
    model_dropout.add(Dense(1, activation='sigmoid'))

    model_dropout.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_dropout_callbacks = [] 
    
    model_dropout_callbacks.append(EarlyStopping(
        monitor='acc', min_delta=0.0002,
        # monitor='loss', min_delta=0.00002,                                                 
        patience=5,
        verbose=0,
        mode='auto'))
    
    # train
    train_start = datetime.now()
    hist_model_dropout = model_dropout.fit(X_train, y_train,
                                           epochs=max_epochs_model_dropout, batch_size=50,
                                           verbose=fit_verbose, validation_data=val_data,
                                           sample_weight=w_train,
                                           callbacks=model_dropout_callbacks);

    hist_dict_model_dropout = hist_model_dropout.history
    print(strfdelta(datetime.now()-train_start, "Training time: {hours} hours, {minutes} minutes, {seconds} seconds"))

    # save model to HDF5, history to pickle
    model_dropout.save(models_path+'/'+fname_model_dropout+'.h5')
   
    with open(models_path+'/'+fname_model_dropout+'_hist.pickle', 'wb') as handle:
        pickle.dump(hist_dict_model_dropout, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    # load model from HDF5, history from pickle
    model_dropout = load_model(models_path+'/'+fname_model_dropout+'.h5')
    
    with open(models_path+'/'+fname_model_dropout+'_hist.pickle', 'rb') as handle:
        hist_dict_model_dropout = pickle.load(handle)


# In[ ]:


plot_acc_loss_vs_epoch(hist_dict_model_dropout, 'NN (dropout)', 'nn_dropout', plots_path, 'Test', True, False)
plot_acc_loss_vs_epoch(hist_dict_model_dropout, 'NN (dropout)', 'nn_dropout', plots_path, 'Test', False, True)


# In[ ]:


print("model_dropout %s: %.2f%%" % (model_dropout.metrics_names[1], model_dropout.evaluate(X_test,y_test,verbose=0)[1]*100))

plot_classifier_1D_output(model_dropout.predict(X_test[y_test>0.5], verbose=0), # sig
                          model_dropout.predict(X_test[y_test<0.5], verbose=0), # bkg
                          'NN (dropout)', 'nn_dropout', plots_path
                         )

fpr_model_dropout, tpr_model_dropout, thresholds_model_dropout = roc_curve(y_test, model_dropout.predict(X_test, verbose=0))
roc_model_dropout = [tpr_model_dropout, fpr_model_dropout, 'NN (dropout)', 'nn_dropout', 'maroon', '-.']

plot_roc([roc_eprob, roc_model_dropout], plots_path)


# ## L2 - TODO

# ## Dropout and L2? - TODO

# ## Wide

# In[ ]:


fname_model_wide = 'model_wide'
train_load_model_wide = train_or_load(models_path+'/'+fname_model_wide+'.h5', default_to_load)


# In[ ]:


if train_load_model_wide == 'n':
    
    # create
    model_wide = Sequential()
    model_wide.add(Dense(24, input_dim=input_ndimensions, activation='relu'))
    model_wide.add(Dense(16, activation='relu'))
    model_wide.add(Dense(1, activation='sigmoid'))

    model_wide.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train
    train_start = datetime.now()
    hist_model_wide = model_wide.fit(X_train, y_train,
                                     epochs=max_epochs, batch_size=50,
                                     verbose=fit_verbose, validation_data=val_data,
                                     sample_weight=w_train);

    hist_dict_model_wide = hist_model_wide.history
    print(strfdelta(datetime.now()-train_start, "Training time: {hours} hours, {minutes} minutes, {seconds} seconds"))

    # save model to HDF5, history to pickle
    model_wide.save(models_path+'/'+fname_model_wide+'.h5')
   
    with open(models_path+'/'+fname_model_wide+'_hist.pickle', 'wb') as handle:
        pickle.dump(hist_dict_model_wide, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    # load model from HDF5, history from pickle
    model_wide = load_model(models_path+'/'+fname_model_wide+'.h5')
    
    with open(models_path+'/'+fname_model_wide+'_hist.pickle', 'rb') as handle:
        hist_dict_model_wide = pickle.load(handle)


# In[ ]:


plot_acc_loss_vs_epoch(hist_dict_model_wide, 'NN (wide)', 'nn_wide', plots_path, 'Test', True, False)
plot_acc_loss_vs_epoch(hist_dict_model_wide, 'NN (wide)', 'nn_wide', plots_path, 'Test', False, True)


# In[ ]:


print("model_wide %s: %.2f%%" % (model_wide.metrics_names[1], model_wide.evaluate(X_test,y_test,verbose=0)[1]*100))

plot_classifier_1D_output(model_wide.predict(X_test[y_test>0.5], verbose=0), # sig
                          model_wide.predict(X_test[y_test<0.5], verbose=0), # bkg
                          'NN (wide)', 'nn_wide', plots_path
                         )

fpr_model_wide, tpr_model_wide, thresholds_model_wide = roc_curve(y_test, model_wide.predict(X_test, verbose=0))
roc_model_wide = [tpr_model_wide, fpr_model_wide, 'NN (wide)', 'nn_wide', 'cyan', '-.']

plot_roc([roc_eprob, roc_model_wide], plots_path)


# ## Deep

# In[ ]:


fname_model_deep = 'model_deep'
train_load_model_deep = train_or_load(models_path+'/'+fname_model_deep+'.h5', default_to_load)


# In[ ]:


if train_load_model_deep == 'n':
    
    # create
    model_deep = Sequential()
    model_deep.add(Dense(12, input_dim=input_ndimensions, activation='relu'))
    model_deep.add(Dense(8, activation='relu'))
    model_deep.add(Dense(8, activation='relu'))
    model_deep.add(Dense(8, activation='relu'))
    model_deep.add(Dense(8, activation='relu'))
    model_deep.add(Dense(1, activation='sigmoid'))

    model_deep.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train
    train_start = datetime.now()
    hist_model_deep = model_deep.fit(X_train, y_train,
                                     epochs=max_epochs, batch_size=50,
                                     verbose=fit_verbose, validation_data=val_data,
                                     sample_weight=w_train);

    hist_dict_model_deep = hist_model_deep.history
    print(strfdelta(datetime.now()-train_start, "Training time: {hours} hours, {minutes} minutes, {seconds} seconds"))

    # save model to HDF5, history to pickle
    model_deep.save(models_path+'/'+fname_model_deep+'.h5')
   
    with open(models_path+'/'+fname_model_deep+'_hist.pickle', 'wb') as handle:
        pickle.dump(hist_dict_model_deep, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
else:
    # load model from HDF5, history from pickle
    model_deep = load_model(models_path+'/'+fname_model_deep+'.h5')
    
    with open(models_path+'/'+fname_model_deep+'_hist.pickle', 'rb') as handle:
        hist_dict_model_deep = pickle.load(handle)


# In[ ]:


plot_acc_loss_vs_epoch(hist_dict_model_deep, 'NN (deep)', 'nn_deep', plots_path, 'Test', True, False)
plot_acc_loss_vs_epoch(hist_dict_model_deep, 'NN (deep)', 'nn_deep', plots_path, 'Test', False, True)


# In[ ]:


print("model_deep %s: %.2f%%" % (model_deep.metrics_names[1], model_deep.evaluate(X_test,y_test,verbose=0)[1]*100))

plot_classifier_1D_output(model_deep.predict(X_test[y_test>0.5], verbose=0), # sig
                          model_deep.predict(X_test[y_test<0.5], verbose=0), # bkg
                          'NN (deep)', 'nn_deep', plots_path
                         )

fpr_model_deep, tpr_model_deep, thresholds_model_deep = roc_curve(y_test, model_deep.predict(X_test, verbose=0))
roc_model_deep = [tpr_model_deep, fpr_model_deep, 'NN (deep)', 'nn_deep', 'darkorange', '--']

plot_roc([roc_eprob, roc_model_deep], plots_path)


# ## Compare all models

# In[ ]:


all_models = []
all_models.append(roc_eprob)
all_models.append(roc_bdt1)
all_models.append(roc_svm1)
all_models.append(roc_model_default)
all_models.append(roc_model_wide)
all_models.append(roc_model_deep)

plot_roc(all_models, plots_path)

roc_model_default_clean = list(roc_model_default)
roc_model_default_clean[2] = 'NN'
roc_model_default_clean[3] += '_clean'

plot_roc([roc_eprob, roc_bdt1, roc_svm1], plots_path)
plot_roc([roc_eprob, roc_bdt1, roc_model_default_clean], plots_path)
plot_roc([roc_eprob, roc_svm1, roc_model_default_clean], plots_path)


# ## $k$-Fold of Default NN

# In[ ]:


from sklearn.model_selection import StratifiedKFold

kfold_splits = 5
kfold_max_epochs = 40
kfold_fit_verbose = 0

kfold_default_to_load = True

plots_path_kfold = plots_path+'/kfold'
models_path_kfold = models_path+'/kfold'
make_path(plots_path_kfold)
make_path(models_path_kfold)

skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True, random_state=rnd_seed)


# In[ ]:


accs=[]
losses=[]
val_accs=[]
val_losses=[]

for fold_index, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train)):
    
    fname_mode_kfold = ("fold_%d" % fold_index)
    train_load_model_kfold = train_or_load(models_path_kfold+'/'+fname_mode_kfold+'.h5', kfold_default_to_load)

    if train_load_model_kfold == 'n':

        print("Training on fold {0:d}/{1:d}".format(fold_index+1, kfold_splits))
        fold_start = datetime.now()
    
        # Generate batches from indices
        this_X_train, this_X_val = X_train[train_indices], X_train[val_indices]
        this_y_train, this_y_val = y_train[train_indices], y_train[val_indices]
        this_w_train, this_w_val = w_train[train_indices], w_train[val_indices]

        this_w_train = w_train[train_indices]
    
        # create
        model_kfold = Sequential()
        model_kfold.add(Dense(12, input_dim=input_ndimensions, activation='relu'))
        model_kfold.add(Dense(8, activation='relu'))
        model_kfold.add(Dense(1, activation='sigmoid'))

        model_kfold.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # model_kfold_callbacks = [] 
        # model_kfold_callbacks.append(EarlyStopping(
        #     monitor='acc', min_delta=0.0002,
        #     monitor='loss', min_delta=0.00002,                                                 
        #     patience=5,
        #     verbose=0,
        #     mode='auto'))

        # train
        train_start = datetime.now()
        hist_model_kfold = model_kfold.fit(this_X_train, this_y_train,
                                           epochs=kfold_max_epochs, batch_size=50,
                                           verbose=kfold_fit_verbose, validation_data=(this_X_val, this_y_val, this_w_val),
                                           sample_weight=this_w_train,
                                          # callbacks=model_kfold_callbacks
                                          );
    
        print(strfdelta(datetime.now()-fold_start, "Training time: {hours} hours, {minutes} minutes, {seconds} seconds"))

        hist_dict_model_kfold = hist_model_kfold.history
        
        # save model to HDF5, history to pickle
        model_kfold.save(models_path_kfold+'/'+fname_mode_kfold+'.h5')
   
        with open(models_path_kfold+'/'+fname_mode_kfold+'_hist.pickle', 'wb') as handle:
            pickle.dump(hist_dict_model_kfold, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    else:
        # load model from HDF5, history from pickle
        model_kfold = load_model(models_path_kfold+'/'+fname_mode_kfold+'.h5')
    
        with open(models_path_kfold+'/'+fname_mode_kfold+'_hist.pickle', 'rb') as handle:
            hist_dict_model_kfold = pickle.load(handle)
        
    # save hist to kfold lists, make plots
    accs.append(hist_dict_model_kfold['acc'])
    losses.append(hist_dict_model_kfold['loss'])
    val_accs.append(hist_dict_model_kfold['val_acc'])
    val_losses.append(hist_dict_model_kfold['val_loss'])

    kfold_name = 'NN (fold {:d}/{:d})'.format(fold_index+1, kfold_splits)
    kfold_nname = 'nn_fold_{:d}'.format(fold_index+1)

    plot_acc_loss_vs_epoch(hist_dict_model_kfold, kfold_name, kfold_nname, plots_path_kfold, 'Validation', True, False)
    plot_acc_loss_vs_epoch(hist_dict_model_kfold, kfold_name, kfold_nname, plots_path_kfold, 'Validation', False, True)
 
    print("On test (not valid) data, this kfold %s: %.2f%%" % (model_kfold.metrics_names[1], model_kfold.evaluate(X_test,y_test,verbose=0)[1]*100))

    plot_classifier_1D_output(model_kfold.predict(X_test[y_test>0.5], verbose=0), # sig
                              model_kfold.predict(X_test[y_test<0.5], verbose=0), # bkg
                              kfold_name, kfold_nname, plots_path_kfold)

    fpr_model_kfold, tpr_model_kfold, thresholds_model_kfold = roc_curve(y_test, model_kfold.predict(X_test, verbose=0))
    roc_model_kfold = [tpr_model_kfold, fpr_model_kfold, kfold_name, kfold_nname, 'magenta', '--']

    plot_roc([roc_eprob, roc_model_kfold], plots_path_kfold)

    print(strfdelta(datetime.now()-fold_start, "\nTotal fold time: {hours} hours, {minutes} minutes, {seconds} seconds\n"))


# In[ ]:


process_kfold_hist_elements(accs, losses, val_accs, val_losses, plots_path,
                            'NN ({:d}-fold)'.format(kfold_splits),
                            'nn_{:d}fold'.format(kfold_splits))


# ## Mutual Information Plots

# ### Training Variables

# In[ ]:


if plot_mi: mutual_info_plot(var_names_dict, df_sig, 'Training Vars: Signal ($e$)', 'train_var_sig', plots_path)


# In[ ]:


if plot_mi: mutual_info_plot(var_names_dict, df_bkg, 'Training Vars: Background', 'train_var_bkg', plots_path)


# In[ ]:


if plot_mi: mutual_info_plot(var_names_dict,
                             pd.concat([df_sig, df_bkg]),
                             'Training Vars: Signal ($e$) & Background', 'train_var_sig_bkg', plots_path)


# ### All Variables

# In[ ]:


all_vars=[
'p',
'pT',
'eta',
'nTRThitsMan',
'nTRTouts',
'fHTMB',
'fAr',
'trkOcc',
'sumToTsumL',
# 'lep_pT',
'phi',
'PHF',
# 'NhitsdEdx',
'sumToT',
'sumL',
'nTRThits',
'nArhits',
'nXehits',
'nHThitsMan',
'nPrechitsMan',
'eProbHT'
]

if plot_mi:
    df_sig_all_vars = create_df(sig_file_name, sig_tree, all_vars)
    df_bkg_all_vars = create_df(bkg_file_name, bkg_tree, all_vars)


# In[ ]:


if plot_mi: mutual_info_plot({var:var for var in all_vars}, df_sig_all_vars, 'All Vars: Signal ($e$)', 'all_var_sig', 'plots')


# In[ ]:


if plot_mi: mutual_info_plot({var:var for var in all_vars}, df_bkg_all_vars, 'All Vars: Background', 'all_var_bkg', 'plots')


# In[ ]:


if plot_mi: mutual_info_plot({var:var for var in all_vars},
                             pd.concat([df_sig_all_vars, df_bkg_all_vars]),
                             'All Vars: Signal ($e$) & Background', 'all_var_sig_bkg', 'plots')


# In[ ]:


print("Total elapsed time: %s" % (strfdelta(datetime.now()-time_all_start, "{hours} hours, {minutes} minutes, {seconds} seconds")))


# ### Make Reference ReLU and Sigmoid diagrams

# In[ ]:


x = np.linspace(-10,10,100)

fig, ax = plt.subplots()
y = np.array([max(xi,0) for xi in x])
ax.plot(x, y, lw=2, c='black', ls='-', label='ReLU')
ax.set_xlabel('$x$')
ax.set_xticks([-10,-5,0,5,10])
plt.figtext(0.3, 0.8, '$R(x) = \max(0,x)$', ha='center', va='center', size=16)
plt.title('ReLU');
fig.savefig('plots/relu.pdf')

fig, ax = plt.subplots()
y = np.array([1./(1.+np.exp(-xi)) for xi in x])
ax.plot(x, y, lw=2, c='black', ls='-', label='Sigmoid')
ax.set_xlabel('$x$')
ax.set_xticks([-10,-5,0,5,10])
ax.set_yticks([0,1])
plt.figtext(0.3, 0.8, r'$S(x) = \frac{1}{1+e^{-x}}$', ha='center', va='center', size=16)
plt.title('Sigmoid');
fig.savefig('plots/sigmoid.pdf')

