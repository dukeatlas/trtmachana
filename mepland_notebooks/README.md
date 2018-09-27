# ATLAS TRT Particle ID Machine Learning R&D Studies  
Matthew Epland  
matthew.epland@duke.edu  

## Summary
R&D studies of particle identification in the ATLAS Transition Radiation Tracker (TRT) where conducted utilizing machine learning techniques, with the goal of separating electron tracks from muons. Developed with fellow Duke graduate students Doug Davis and Sourav Sen, and continued by Davis and others within the TRT group. Support Vector Machines (SVM) and Boosted Decision Trees (BDT) from the sklearn library were tested, as well as Neural Networks (NN) in constructed in Keras / Tensorflow.  

## Cloning the Repository
ssh  
```bash
git clone git@github.com:dukeatlas/trtmachana.git
```

https  
```bash
git clone https://github.com/dukeatlas/trtmachana.git
```
## Installing Dependencies
It is recommended to work in a `virtualenv` to avoid clashes with other installed software. A useful extension for this purpose is [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/). Follow the instructions in the documentation to install and initialize wrapper before continuing.  

```bash
mkvirtualenv newenv
pip install -r requirements.txt

```
## Running
```bash
cd mepland_notebooks
jupyter lab notebook.ipynb
```
