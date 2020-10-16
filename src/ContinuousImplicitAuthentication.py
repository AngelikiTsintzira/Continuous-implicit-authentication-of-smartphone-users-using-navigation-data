#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Angeliki Agathi Tsintzira
# Github      : https://github.com/AngelikiTsintzira
# Linkedin    : https://www.linkedin.com/in/angeliki-agathi-tsintzira/
# Created Date: October 2020
# =============================================================================
# Licence GPLv3
# =============================================================================
# This file is part of Continuous implicit authentication of smartphone users using navigation data.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# =============================================================================
# Description
# =============================================================================
"""
This is a Python 3.7.4 64bit project.

The goal of this study is to propose a methodology for continuous implicit authentication of smartphones users, 
using the navigation data, in order to improve the security and ensure the privacy of sensitive personal data.

The dataset used for this project can be found here: https://zenodo.org/record/2598135#.X4H0qXgzbeo
The sensors_data.zip file includes json files with sensors measurements of 2000 smartphones users.
More information about the dataset can be found here: https://res.mdpi.com/data/data-04-00060/article_deploy/data-04-00060.pdf?filename=&attachment=1

In order to use this software, follow the steps below:
    Step 1) Create a new folder (e.g. sensors_data) 
    Step 2) For each user create a folder with its ID (e.g. 0hz8270) inside the previous folder. In the user's folder, move all of its json files.
            You don't have to use all the users. For example, user 0hz8270 has 67 json files. Move all of them inside folder 0hz8270.
    Step 3) Set the variable "path" with the absosule path of folder sensors_data.
    Step 4) Execute the project. You can do it from an IDE or from terminal.

NOTE: The process of reading json files takes a lot of time to execute. You can use a Jupyter Notebook and split the code in cells.
Create a cell for the process of loading data and another cell(s) for the data preprocess and machine learning algorithms.
With this way, the loading of the data happens only once and then you can do whatever you want with the models and the preprocess.
(as long as you DO NOT CHANGE the gyroscope, accelerometer and Gestures Dataframes). 
The code as it is, does not do any modifications to above mention Dataframes and it is safe to split it.
"""
# =============================================================================
# Imports
# =============================================================================
import os
import sys
import math
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd
import numpy as np
import ujson as json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import IsolationForest

from sklearn import metrics
from sklearn import svm
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft,fft2, fftshift
from numpy import quantile, where, random

from metrics import Metrics
# =============================================================================
# Load Dataset and Create Panda Dataframes
# =============================================================================
# Set the absolute path in which the json files are saved.
path = '/Users/angelikitsintzira/Downloads/Dissertation/sensors_data'
users = [ f.path for f in os.scandir(path) if f.is_dir() ]
info = pd.DataFrame(columns= ['accelometer_size', 'gyroscope_size', 'timestamp'])

accelerometer = pd.DataFrame(columns=['x', 'y', 'z', 'screen', 'user', 'magnitude','combine_angle', 'timestamp'])
gyroscope = pd.DataFrame(columns=['x_gyroscope', 'y_gyroscope', 'z_gyroscope', 'screen_gyroscope', 'user_gyroscope', 'magnitude_gyroscope', 'combine_angle_gyroscope', 'timestamp_gyroscope'])
screenName = 'MemoriaGame'
Gestures = pd.DataFrame(columns=['type', 'dx', 'dy', 'vx', 'vy', 'screen', 'user', 't_start', 't_stop'])

usersCleanName = []

# Read sensors data from json file and save them in Dataframes
for i in range(0, len(users)):

    json_files = [pos_json for pos_json in os.listdir(users[i]) if pos_json.endswith('.json')]

    for index, js in enumerate(json_files):
        with open(os.path.join(users[i], js)) as json_file:
            json_text = json.load(json_file)
            accSize = 0
            gyrSize = 0
            js = js.replace('.json','')
            arr = js.split('_')
            usersCleanName.append(arr[0])

            for j in json_text['accelerometer']:
                if screenName in j['screen']:
                    x = j['x']
                    y = j['y']
                    z = j['z']
                    if x == 0 and y == 0:
                        continue
                    screen = j['screen']
                    user = arr[0]
                    m = x**2 + y**2 + z**2
                    m = np.sqrt(m)
                    ca = np.sqrt(y**2 + z**2)
                    timestamp = arr[1]
                    accSize = accSize + 1
                    df = {'x': x, 'y': y, 'z' : z, 'screen' : screen, 'user': user, 'magnitude' : m, 'combine_angle': ca, 'timestamp': timestamp}
                    accelerometer = accelerometer.append(df, ignore_index=True)
                    
            for j in json_text['gyroscope']:
                if screenName in j['screen']:
                    x = j['x']
                    y = j['y']
                    z = j['z']
                    if x == 0 and y == 0:
                        continue
                    screen = j['screen']
                    user = arr[0]
                    m = x**2 + y**2 + z**2
                    m = np.sqrt(m)
                    ca = np.sqrt(y**2 + z**2)
                    timestamp = arr[1]
                    gyrSize =  gyrSize + 1
                    df = {'x_gyroscope': x, 'y_gyroscope': y, 'z_gyroscope' : z, 'screen_gyroscope' : screen, 'user_gyroscope': user, 'magnitude_gyroscope' : m, 'combine_angle_gyroscope': ca, 'timestamp_gyroscope': timestamp}
                    gyroscope = gyroscope.append(df, ignore_index=True)
                
            dframe = {'accelometer_size': accSize, 'gyroscope_size': gyrSize, 'timestamp': arr[1]}
            info = info.append(dframe, ignore_index=True)

# =============================================================================
# Pre-process 
# =============================================================================
# feauture variable gets values 0, 1, 5, 6
# where 0 is X, 1 is Y, 5 is magnitude and 6 is combined angle
feauture = 5
samples = 500
overlap = 50
print('Feature: ', feauture)

# Objects with metrics for each model
accelerometerModel = Metrics()
gyroscopeModel = Metrics()
ensembleModel = Metrics()

for user in users:

    ## Feautures 
    # Statistic feautures based on time
    Μean = []
    STD = []
    Max = []
    Min = []
    Range = []
    Percentile25 = []
    Percentile50 = []
    Percentile75 = []
    Kurtosis = []
    Skewness = []
    Entropy = []

    # Statistics feautures based on frequency
    Amplitude1 = []
    Amplitude2 = []
    Frequency2 = []
    Mean_frequency = []

    # Statistic feautures based on time
    Μean_Gyroscope = []
    STD_Gyroscope  = []
    Max_Gyroscope  = []
    Min_Gyroscope = []
    Range_Gyroscope = []
    Percentile25_Gyroscope = []
    Percentile50_Gyroscope = []
    Percentile75_Gyroscope = []
    Kurtosis_Gyroscope = []
    Skewness_Gyroscope = []
    Entropy_Gyroscope = []

    # Statistics feautures based on frequency
    Amplitude1_Gyroscope = []
    Amplitude2_Gyroscope = []
    Frequency2_Gyroscope = []
    Mean_frequency_Gyroscope = []
    
    # Output label
    list_y=[]
    list_y_Gyroscope=[]

    # Dataset dataframe
    df = pd.DataFrame() 
    df_gyroscope = pd.DataFrame() 

    train = []
    test = []
    train_gyroscope = []
    test_gyroscope  = []
    arr = user.rsplit('/', 1)
    user = arr[1]
    print('User: ', user)

    # Create accelerometer and gyroscope Dataframes where each instance in accelerometer Dataframe 
    # has an instance in gyroscope Dataframe at the same timestamp. 
    for ind in info.index: 

        if info['accelometer_size'][ind] == 0 & info['gyroscope_size'][ind] == 0:
            continue
        else:

            if info['gyroscope_size'][ind] < info['accelometer_size'][ind]:
                info['accelometer_size'][ind] = info['gyroscope_size'][ind]   
        
            if info['accelometer_size'][ind] < info['gyroscope_size'][ind]:
                info['gyroscope_size'][ind] = info['accelometer_size'][ind]

            # Indecies where each timestamp contains accelometer and gyroscope measures
            idxAccelometer = accelerometer.index[accelerometer['timestamp'] == info['timestamp'][ind]]
            idxGyroscope = gyroscope.index[gyroscope['timestamp_gyroscope'] == info['timestamp'][ind]]

            for i in range(0, info['gyroscope_size'][ind]):

                # Original user
                if accelerometer.iloc[idxAccelometer[i],4] == user: 
                    frame = {'x': accelerometer.iloc[idxAccelometer[i],0], 'y' : accelerometer.iloc[idxAccelometer[i],1], 'z' : accelerometer.iloc[idxAccelometer[i],2], 
                    'screen' : accelerometer.iloc[idxAccelometer[i],3], 'user' : accelerometer.iloc[idxAccelometer[i],4], 'magnitude' : accelerometer.iloc[idxAccelometer[i],5],
                    'combine_angle' : accelerometer.iloc[idxAccelometer[i],6], 'timestamp': accelerometer.iloc[idxAccelometer[i],7]}
                    frame2 = {  'x_gyroscope' : gyroscope.iloc[idxGyroscope[i],0], 'y_gyroscope': gyroscope.iloc[idxGyroscope[i],1], 
                    'z_gyroscope': gyroscope.iloc[idxGyroscope[i],2], 'screen_gyroscope': gyroscope.iloc[idxGyroscope[i],3], 'user_gyroscope' : gyroscope.iloc[idxGyroscope[i],4], 
                    'magnitude_gyroscope': gyroscope.iloc[idxGyroscope[i],5], 'combine_angle_gyroscope': gyroscope.iloc[idxGyroscope[i],6], 'timestamp_gyroscope': gyroscope.iloc[idxGyroscope[i],7]}
                    train.append(frame)
                    train_gyroscope.append(frame2)
                # Attackers    
                else:
                    frame = {'x': accelerometer.iloc[idxAccelometer[i],0], 'y' : accelerometer.iloc[idxAccelometer[i],1], 'z' : accelerometer.iloc[idxAccelometer[i],2], 
                    'screen' : accelerometer.iloc[idxAccelometer[i],3], 'user' : accelerometer.iloc[idxAccelometer[i],4], 'magnitude' : accelerometer.iloc[idxAccelometer[i],5],
                    'combine_angle' : accelerometer.iloc[idxAccelometer[i],6], 'timestamp': accelerometer.iloc[idxAccelometer[i],7]}
                    frame2 = {  'x_gyroscope' : gyroscope.iloc[idxGyroscope[i],0], 'y_gyroscope': gyroscope.iloc[idxGyroscope[i],1], 
                    'z_gyroscope': gyroscope.iloc[idxGyroscope[i],2], 'screen_gyroscope': gyroscope.iloc[idxGyroscope[i],3], 'user_gyroscope' : gyroscope.iloc[idxGyroscope[i],4], 
                    'magnitude_gyroscope': gyroscope.iloc[idxGyroscope[i],5], 'combine_angle_gyroscope': gyroscope.iloc[idxGyroscope[i],6], 'timestamp_gyroscope': gyroscope.iloc[idxGyroscope[i],7]}
                    test.append(frame)
                    test_gyroscope.append(frame2)
                    
    print('Accelometer Data Sizes:\n', 'Train Size One Class: ', len(train), ' Test Size Two Class: ', len(test))
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    train_gyroscope = pd.DataFrame(train_gyroscope)
    test_gyroscope = pd.DataFrame(test_gyroscope)

    # =============================================================================
    # Feature Extraction
    # =============================================================================     
    # Create train dataset for accelerometer
    for w in range(0, train.shape[0] - samples, overlap):
        end = w + samples 

        #DFT
        discreteFourier = fft(train.iloc[w:end, feauture])
        # Frequencies
        freq = np.fft.fftfreq(samples)

        # Amplitudes
        idx = (np.absolute(discreteFourier)).argsort()[-2:][::-1]
        amplitude1 = np.absolute(discreteFourier[idx[0]])
        amplitude2 = np.absolute(discreteFourier[idx[1]])
        frequency2 = freq[idx[1]]

        # Frequency features
        mean_frequency = np.mean(freq)
        Amplitude1.append(amplitude1)
        Amplitude2.append(amplitude2)
        Frequency2.append(frequency2)
        Mean_frequency.append(mean_frequency)

        # Time Based Feautures
        Μean.append(np.mean(train.iloc[w:end, feauture]))
        STD.append(np.std(train.iloc[w:end, feauture]))
        Max.append(np.max(train.iloc[w:end, feauture]))
        Min.append(np.min(train.iloc[w:end, feauture]))
        Range.append(np.ptp(train.iloc[w:end, feauture]))

        percentile = np.percentile(train.iloc[w:end, feauture], [25, 50, 75])
        Percentile25.append(percentile[0])
        Percentile50.append(percentile[1])
        Percentile75.append(percentile[2])
        Entropy.append(entropy(train.iloc[w:end, feauture], base = 2))

        Kurtosis.append(kurtosis(train.iloc[w:end, feauture]))
        Skewness.append(skew(train.iloc[w:end, feauture]))

        # Output Label
        list_y.append(1)

    # Create train dataset for gyroscope
    for w in range(0, train_gyroscope.shape[0] - samples, overlap):
        end = w + samples 

        #DFT
        discreteFourier = fft(train_gyroscope.iloc[w:end, feauture])
        # Frequencies
        freq = np.fft.fftfreq(samples)

        # Amplitudes
        idx = (np.absolute(discreteFourier)).argsort()[-2:][::-1]
        amplitude1 = np.absolute(discreteFourier[idx[0]])
        amplitude2 = np.absolute(discreteFourier[idx[1]])
        frequency2 = freq[idx[1]]

        # Frequency features
        mean_frequency = np.mean(freq)
        Amplitude1_Gyroscope.append(amplitude1)
        Amplitude2_Gyroscope.append(amplitude2)
        Frequency2_Gyroscope.append(frequency2)
        Mean_frequency_Gyroscope.append(mean_frequency)

        # Time Based Feautures
        Μean_Gyroscope.append(np.mean(train_gyroscope.iloc[w:end, feauture]))
        STD_Gyroscope.append(np.std(train_gyroscope.iloc[w:end, feauture]))
        Max_Gyroscope.append(np.max(train_gyroscope.iloc[w:end, feauture]))
        Min_Gyroscope.append(np.min(train_gyroscope.iloc[w:end, feauture]))
        Range_Gyroscope.append(np.ptp(train_gyroscope.iloc[w:end, 5]))

        percentile = np.percentile(train_gyroscope.iloc[w:end, feauture], [25, 50, 75])
        Percentile25_Gyroscope.append(percentile[0])
        Percentile50_Gyroscope.append(percentile[1])
        Percentile75_Gyroscope.append(percentile[2])
        Entropy_Gyroscope.append(entropy(train_gyroscope.iloc[w:end, feauture], base = 2))

        Kurtosis_Gyroscope.append(kurtosis(train_gyroscope.iloc[w:end, feauture]))
        Skewness_Gyroscope.append(skew(train_gyroscope.iloc[w:end, feauture]))

        # Output Label
        list_y_Gyroscope.append(1)

    # Create test dataset for accelerometer
    for w in range(0, len(test) - samples, overlap):
        end = w + samples 

        #DFT
        discreteFourier = fft(test.iloc[w:end, feauture])
        # Frequencies
        freq = np.fft.fftfreq(samples)

        # Amplitudes
        idx = (np.absolute(discreteFourier)).argsort()[-2:][::-1]
        amplitude1 = np.absolute(discreteFourier[idx[0]])
        amplitude2 = np.absolute(discreteFourier[idx[1]])
        frequency2 = freq[idx[1]]

        # Frequency features
        mean_frequency = np.mean(freq)
        Amplitude1.append(amplitude1)
        Amplitude2.append(amplitude2)
        Frequency2.append(frequency2)
        Mean_frequency.append(mean_frequency)

        # Time Based Feautures
        Μean.append(np.mean(test.iloc[w:end, feauture]))
        STD.append(np.std(test.iloc[w:end, feauture]))
        Max.append(np.max(test.iloc[w:end, feauture]))
        Min.append(np.min(test.iloc[w:end, feauture]))
        Range.append(np.ptp(test.iloc[w:end, feauture].to_numpy()))

        percentile = np.percentile(test.iloc[w:end, feauture], [25, 50, 75])
        Percentile25.append(percentile[0])
        Percentile50.append(percentile[1])
        Percentile75.append(percentile[2])
        Entropy.append(entropy(test.iloc[w:end, feauture], base=2))

        Kurtosis.append(kurtosis(test.iloc[w:end, feauture]))
        Skewness.append(skew(test.iloc[w:end, feauture]))

        # Output Label
        list_y.append(-1)

    
    # Create test dataset for gyroscope
    for w in range(0, test_gyroscope.shape[0] - samples, overlap):
        end = w + samples 

        #DFT
        discreteFourier = fft(test_gyroscope.iloc[w:end, feauture])
        # Frequencies
        freq = np.fft.fftfreq(samples)

        # Amplitudes
        idx = (np.absolute(discreteFourier)).argsort()[-2:][::-1]
        amplitude1 = np.absolute(discreteFourier[idx[0]])
        amplitude2 = np.absolute(discreteFourier[idx[1]])
        frequency2 = freq[idx[1]]

        # Frequency features
        mean_frequency = np.mean(freq)
        Amplitude1_Gyroscope.append(amplitude1)
        Amplitude2_Gyroscope.append(amplitude2)
        Frequency2_Gyroscope.append(frequency2)
        Mean_frequency_Gyroscope.append(mean_frequency)

        # Time Based Feautures
        Μean_Gyroscope.append(np.mean(test_gyroscope.iloc[w:end, feauture]))
        STD_Gyroscope.append(np.std(test_gyroscope.iloc[w:end, feauture]))
        Max_Gyroscope.append(np.max(test_gyroscope.iloc[w:end, feauture]))
        Min_Gyroscope.append(np.min(test_gyroscope.iloc[w:end, feauture]))
        Range_Gyroscope.append(np.ptp(test_gyroscope.iloc[w:end, feauture].to_numpy()))

        percentile = np.percentile(test_gyroscope.iloc[w:end, feauture], [25, 50, 75])
        Percentile25_Gyroscope.append(percentile[0])
        Percentile50_Gyroscope.append(percentile[1])
        Percentile75_Gyroscope.append(percentile[2])
        Entropy_Gyroscope.append(entropy(test_gyroscope.iloc[w:end, feauture], base = 2))

        Kurtosis_Gyroscope.append(kurtosis(test_gyroscope.iloc[w:end, feauture]))
        Skewness_Gyroscope.append(skew(test_gyroscope.iloc[w:end, feauture]))

        # Output Label
        list_y_Gyroscope.append(-1)
    
    # =============================================================================
    # Machine Learning Models
    # ============================================================================= 

    # Add features to dataframe
    df['Mean'] = Μean 
    df['Std'] = STD
    df['Skew'] = Skewness
    df['Kurtosis'] = Kurtosis
    df['Max'] = Max
    df['Min'] = Min
    #df['Range'] = Range
    df['Percentile25'] = Percentile25
    df['Percentile50'] = Percentile50
    df['Percentile75'] = Percentile75
    #df['Entropy'] = Entropy

    df['Amplitude1'] = Amplitude1
    df['Amplitude2'] = Amplitude2
    df['Frequency'] = Frequency2
    df['MeanFrequency'] = Mean_frequency

    df_gyroscope['Mean_Gyroscope'] = Μean_Gyroscope
    df_gyroscope['Std_Gyroscope'] = STD_Gyroscope
    df_gyroscope['Skew_Gyroscope'] = Skewness_Gyroscope
    df_gyroscope['Kurtosis_Gyroscope'] = Kurtosis_Gyroscope
    df_gyroscope['Max_Gyroscope'] = Max_Gyroscope
    df_gyroscope['Min_Gyroscope'] = Min_Gyroscope
    #df_gyroscope['Range_Gyroscope'] = Range_Gyroscope
    df_gyroscope['Percentile25_Gyroscope'] = Percentile25_Gyroscope
    df_gyroscope['Percentile50_Gyroscope'] = Percentile50_Gyroscope
    df_gyroscope['Percentile75_Gyroscope'] = Percentile75_Gyroscope
    #df_gyroscope['Entropy_Gyroscope'] = Entropy_Gyroscope

    df_gyroscope['Amplitude1_Gyroscope'] = Amplitude1_Gyroscope
    df_gyroscope['Amplitude2_Gyroscope'] = Amplitude2_Gyroscope
    df_gyroscope['Frequency_Gyroscope'] = Frequency2_Gyroscope
    df_gyroscope['MeanFrequency_Gyroscope'] = Mean_frequency_Gyroscope

    df['y'] = list_y
    df_gyroscope['y_gyroscope'] = list_y_Gyroscope

    train =  pd.DataFrame(df[df['y'] == 1])
    test =  pd.DataFrame(df[df['y'] == -1])
    
    train_gyroscope =  pd.DataFrame(df_gyroscope[df_gyroscope['y_gyroscope'] == 1])
    test_gyroscope =  pd.DataFrame(df_gyroscope[df_gyroscope['y_gyroscope'] == -1])

    print('After Sampling Sizes\n','Train Size One Class: ', train.shape[0], 'Test Size Two Class: ', test.shape[0])

    Far1 = []
    Far2 = []
    Far3 = []
    
    Frr1 = []
    Frr2 = []
    Frr3 = []
    
    ac1 = []
    ac2 = []
    ac3 = []
    
    f11 = []
    f12 = []
    f13 = []
    
    ROC11 = []
    ROC12 = []
    ROC13 = []
    
    fA1 = []
    fA2 = []
    fA3 = []
    
    FR1 = []
    FR2 = []
    FR3 = []
    
    TA1 = []
    TA2 = []
    TA3 = []
    
    TR1 = []
    TR2 = []
    TR3 = []
    
    testSize1 = []
    testSize2 = []
    testSize3 = []
    
    trainStarter = train
    testStarter = test
        
    trainStarter_gyroscope = train_gyroscope
    testStarter_gyroscope = test_gyroscope
    
    # 10 Executions per original user
    for fold in range(0,10):

        ######################################## MODEL 1 - Accelerometer ########################################

        # Split Dataset in a random way
        percent = int(math.ceil(0.2 * trainStarter.shape[0]))
        sampling = trainStarter.sample(n = percent)
        
        indecies = []
        for ii in range(0, percent):
            indecies.append(sampling.iloc[ii,:].name)

        test = pd.concat([testStarter, sampling])
        train = trainStarter.drop(trainStarter.index[indecies])

        X_train = train.iloc[:,0:test.shape[1]-2]
        y_train = train.iloc[:,test.shape[1]-1 ]

        X_test = test.iloc[:,0:test.shape[1]-2]
        y_test = test.iloc[:,test.shape[1]-1]

        print('After Split Sizes:')
        print('Train Size One Class: ', X_train.shape[0], 'Test Size Two Class: ', y_test.shape[0])

        # MinMaxScaler Normalized to [0,1]
        scaler = MinMaxScaler().fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)
        
        model1 = LocalOutlierFactor(n_neighbors = 3, novelty = True)
        model1.fit(X_train_norm)
        decision1 = model1.decision_function(X_train_norm)
        maxDistance = max(decision1) 
        yhat1 = model1.predict(X_test_norm)
        
        '''
        model1 = LocalOutlierFactor(n_neighbors = 3, novelty = True)
        model1.fit(X_train_norm)
        decision1 = model1.decision_function(X_train_norm)
        maxDistance = max(decision1) 
        yhat1 = model1.predict(X_test_norm)
        
        model1 = EllipticEnvelope(contamination = 0).fit(X_train_norm)
        decision1 = model1.decision_function(X_train_norm)
        maxDistance = max(decision1) 
        yhat1 = model1.predict(X_test_norm)
        
        model1 = IsolationForest(n_jobs = -1, n_estimators = 100, contamination = 0, bootstrap = False).fit(X_train_norm)
        decision1 = model1.decision_function(X_train_norm)
        maxDistance = max(decision1) 
        yhat1 = model1.predict(X_test_norm)

        # auto einai t teleutaio
        model1 = svm.OneClassSVM(gamma = 0.1, kernel="rbf", nu = 0.01, cache_size = 500)
        # fit on majority class
        model1.fit(X_train_norm)

        decision1 = model1.decision_function(X_train_norm)
        maxDistance = max(decision1) 
        #print('-------- Distance Max: ', maxDistance)
        
        # detect outliers in the test set
        yhat1 = model1.predict(X_test_norm)
        '''

        decision1 = model1.decision_function(X_test_norm)
        decision1 = decision1 /maxDistance
        #print('-------- Probability: ', decision1)

        print("******************************* First Model *******************************")
        score = f1_score(y_test, yhat1, pos_label= 1)
        #print('F1 Score: %.3f' % score)
        acc = accuracy_score(y_test, yhat1)
        #print(f'SVM accuracy is {acc}')
        cfm = confusion_matrix(y_test, yhat1, labels = [-1, 1])
        print(cfm)
        np.sum(y_test == -1)
        far = cfm[0,1]/ np.sum(y_test == -1)
        frr = cfm[1,0]/ np.sum(y_test == 1)
        print('FAR: ', far, ' FFR: ', frr)
        
        Far1.append(far)
        Frr1.append(frr)
        ac1.append(acc)
        f11.append(score)
        fA1.append(cfm[0,1])
        FR1.append(cfm[1,0])
        TA1.append(cfm[1,1])
        TR1.append(cfm[0,0])
        testSize1.append(y_test.shape[0])

        # AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.
        roc = roc_auc_score(y_test, yhat1)
        #print('ROC AUC Score: ',roc)

        ######################################## MODEL 2 - Gyroscope ########################################
        
        print('After Sampling Sizes Gyroscope \n','Train Size One Class: ', train_gyroscope.shape[0], 'Test Size Two Class: ', test_gyroscope.shape[0])

        percent = int(math.ceil(0.2 * trainStarter_gyroscope.shape[0]))
        sampling = trainStarter_gyroscope.sample(n = percent) 
        
        indecies = []
        for ii in range(0, percent):
            indecies.append(sampling.iloc[ii,:].name)
            
        test_gyroscope = pd.concat([testStarter_gyroscope, sampling])
        train_gyroscope = trainStarter_gyroscope.drop(trainStarter_gyroscope.index[indecies])

        X_train = train_gyroscope.iloc[:,0:test_gyroscope.shape[1]-2]
        y_train = train_gyroscope.iloc[:,test_gyroscope.shape[1]-1 ]

        X_test = test_gyroscope.iloc[:,0:test_gyroscope.shape[1]-2]
        y_test = test_gyroscope.iloc[:,test_gyroscope.shape[1]-1]
        
        print('After Split Sizes:')
        print('Train Size One Class: ', X_train.shape[0], 'Test Size Two Class: ', y_test.shape[0])

        # MinMaxScaler Normalized to [0,1]
        scaler = MinMaxScaler().fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)

        model2 = LocalOutlierFactor(n_neighbors = 5, novelty = True)
        model2.fit(X_train_norm)
        decision2 = model2.decision_function(X_train_norm)
        maxDistance = max(decision2) 
        yhat2 = model2.predict(X_test_norm)

        '''
        model2 = LocalOutlierFactor(n_neighbors = 5, novelty = True)
        model2.fit(X_train_norm)
        decision2 = model2.decision_function(X_train_norm)
        maxDistance = max(decision2) 
        yhat2 = model2.predict(X_test_norm)

        model2 = EllipticEnvelope(contamination = 0).fit(X_train_norm)
        decision2 = model2.decision_function(X_train_norm)
        maxDistance = max(decision2) 
        yhat2 = model2.predict(X_test_norm)   
        
        model2 = IsolationForest(n_jobs = -1, n_estimators = 100, max_features = 1, contamination = 0, bootstrap = False).fit(X_train_norm)
        decision2 = model2.decision_function(X_train_norm)
        maxDistance = max(decision2) 
        yhat2 = model2.predict(X_test_norm)
        
        # One class SVM last
        model2 = svm.OneClassSVM(gamma = 0.001, kernel="rbf", nu = 0.1, cache_size = 500)
        # fit on majority class
        model2.fit(X_train_norm)

        decision2 = model2.decision_function(X_train_norm)
        maxDistance = max(decision2)  
        #print('-------- Distance Max: ', maxDistance)

        # detect outliers in the test set
        yhat2 = model2.predict(X_test_norm)
        '''

        decision2 = model2.decision_function(X_test_norm)
        decision2 = decision2 /maxDistance
        #print('-------- Probability: ', decision2)

        print("******************************* Second Model *******************************")
        #print("User: ", user)
        # mark inliers 1, outliers -1
        score = f1_score(y_test, yhat2, pos_label= 1)
        #print('F1 Score: %.3f' % score)
        acc = accuracy_score(y_test, yhat2)
        #print(f'SVM accuracy is {acc}')
        cfm = confusion_matrix(y_test, yhat2, labels = [-1, 1])
        print(cfm)
        np.sum(y_test == -1)
        far = cfm[0,1]/ np.sum(y_test == -1)
        frr = cfm[1,0]/ np.sum(y_test == 1)
        print('FAR: ', far, ' FFR: ', frr)
    
        Far2.append(far)
        Frr2.append(frr)
        ac2.append(acc)
        f12.append(score)
        fA2.append(cfm[0,1])
        FR2.append(cfm[1,0])
        TA2.append(cfm[1,1])
        TR2.append(cfm[0,0])
        testSize2.append(y_test.shape[0])

        # AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.
        roc = roc_auc_score(y_test, yhat2)
        #print('ROC AUC Score: ',roc)

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        sumPred = 0
        predictions = []
        for i in range(0, len(decision1)):

            sumPred = (decision1[i] + decision2[i]) 
            if sumPred >= 0:
                predictions.append(1)
                if y_test.iloc[i] == 1:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                predictions.append(-1)
                if y_test.iloc[i] == -1:
                    TN = TN + 1
                else:
                    FN = FN + 1
        print('------------- Ensemle Results ---------------')
        print(TN, " ", FP)
        print(FN, " ", TP)  
        print() 

        score = f1_score(y_test, predictions, pos_label= 1)
        #print('F1 Score: %.3f' % score)
        acc = accuracy_score(y_test, predictions)
        cfm = confusion_matrix(y_test, predictions, labels = [-1, 1])

        far = FP/ np.sum(y_test == -1)
        frr = FN/ np.sum(y_test == 1)           
        print('FAR: ', far, ' FFR: ', frr)
        
        Far3.append(far)
        Frr3.append(frr)
        
        ac3.append(acc)
        f13.append(score)
        fA3.append(cfm[0,1])
        FR3.append(cfm[1,0])
        TA3.append(cfm[1,1])
        TR3.append(cfm[0,0])
        testSize3.append(y_test.shape[0])
    
        decision1 = []
        decision2 = []
        
    # Average of 10 execution per user
    
    far = 0
    frr = 0
    
    far1 = 0
    far2 = 0
    frr1 = 0
    frr2 = 0
    
    accu = 0
    fscore = 0
    
    accu1 = 0
    accu2 = 0
    fscore1 =0
    fscore2 = 0
    
    FA = 0
    FalseR = 0
    TrueA = 0
    TrueR = 0
    testS = 0
    
    FA1 = 0
    FA2 = 0
    
    FalseR1 = 0
    FalseR2 = 0
    
    TrueA1 = 0
    TrueA2 = 0
    
    TrueR1 = 0
    TrueR2 = 0
    
    testS1 = 0
    testS2 = 0
    
    for index in range(0,10):
        far = far + Far3[index]
        frr = frr + Frr3[index]
        
        far1 = far1 + Far1[index]
        far2 = far2 + Far2[index]
        frr1 = frr1 + Frr1[index]
        frr2 = frr2 + Frr2[index]
        
        accu = accu + ac3[index] 
        accu1 = accu1 + ac1[index]
        accu2 = accu2 + ac2[index]
        
        fscore = fscore + f13[index]
        fscore1 = fscore1 + f11[index]
        fscore2 = fscore1 + f12[index]
    
        FA = FA + fA3[index]
        FA1 = FA1 + fA1[index]
        FA2 = FA2 + fA2[index]
        
        FalseR = FalseR + FR3[index]
        FalseR1 =  FalseR1 + FR1[index]
        FalseR2 =  FalseR2 + FR2[index]

        TrueA = TrueA + TA3[index]
        TrueA1 = TrueA1 + TA1[index]
        TrueA2 = TrueA1 + TA2[index]

        TrueR = TrueR + TR3[index]
        TrueR1 = TrueR1 + TR1[index]
        TrueR2 = TrueR1 + TR2[index]
        
        testS = testS + testSize3[index]
        testS1 = testS1 + testSize1[index]
        testS2 = testS2 + testSize2[index]

    accelerometerModel.setAccuracy(accu1/10)
    accelerometerModel.setf1score(fscore1/10)
    accelerometerModel.setFAR(far1/10)
    accelerometerModel.setFRR(frr1/10)
    accelerometerModel.setfalseAccept(FA1/10)
    accelerometerModel.setfalseReject(FalseR1/10)
    accelerometerModel.settrueAccept(TrueA1/10)
    accelerometerModel.settrueReject(TrueR1/10)
    accelerometerModel.setsizeTest(abs(testS1/10))

    gyroscopeModel.setAccuracy(accu2/10)
    gyroscopeModel.setf1score(fscore2/10)
    gyroscopeModel.setFAR(far2/10)
    gyroscopeModel.setFRR(frr2/10)
    gyroscopeModel.setfalseAccept(FA2/10)
    gyroscopeModel.setfalseReject(FalseR2/10)
    gyroscopeModel.settrueAccept(TrueA2/10)
    gyroscopeModel.settrueReject(TrueR2/10)
    gyroscopeModel.setsizeTest(abs(testS2/10))

    ensembleModel.setAccuracy(accu/10)
    ensembleModel.setf1score(fscore/10)
    ensembleModel.setFAR(far/10)
    ensembleModel.setFRR(frr/10)
    ensembleModel.setfalseAccept(FA/10)
    ensembleModel.setfalseReject(FalseR/10)
    ensembleModel.settrueAccept(TrueA/10)
    ensembleModel.settrueReject(TrueR/10)
    ensembleModel.setsizeTest(abs(testS/10))
    
# =============================================================================
# Performance Evaluation
# ============================================================================= 
# Final System Metrics        
accuracy_average = 0
f1score_average = 0
far_average = 0
frr_average = 0
roc_average = 0
for i in range(0, len(users)):
    accuracy_average = accuracy_average + accelerometerModel.getAccuracy()[i]
    f1score_average = f1score_average +  accelerometerModel.getf1score()[i] 
    far_average = far_average +  accelerometerModel.getFAR()[i]
    frr_average = frr_average +  accelerometerModel.getFRR()[i]

print()
print('AVERAGE ONE CLASS SVM PERFORMANCE MODEL 1:')
print('Accuracy: ', accuracy_average / len(users), '\nF1 Score: ', f1score_average / len(users), '\nFAR: ', far_average / len(users), '\nFRR: ', frr_average / len(users))
sumTest = sum(accelerometerModel.getsizeTest())
sumFalseAccept = sum(accelerometerModel.getfalseAccept())
sumFalseReject = sum(accelerometerModel.getfalseReject())
sumTrueAccept = sum(accelerometerModel.gettrueAccept())
sumTrueReject = sum(accelerometerModel.gettrueReject())
print('Confusion Matrix')
print(sumTrueReject, ' ',  sumFalseAccept)
print(sumFalseReject, ' ',  sumTrueAccept)

print('**********************************************************************')
accuracy_average = 0
f1score_average = 0
far_average = 0
frr_average = 0
roc_average = 0
for i in range(0, len(users)):
    accuracy_average = accuracy_average + gyroscopeModel.getAccuracy()[i]
    f1score_average = f1score_average +  gyroscopeModel.getf1score()[i] 
    far_average = far_average +  gyroscopeModel.getFAR()[i]
    frr_average = frr_average +  gyroscopeModel.getFRR()[i]

print()
print('AVERAGE ONE CLASS SVM PERFORMANCE MODEL 2:')
print('Accuracy: ', accuracy_average / len(users), '\nF1 Score: ', f1score_average / len(users), '\nFAR: ', far_average / len(users), '\nFRR: ', frr_average / len(users))
sumTest = sum(gyroscopeModel.getsizeTest())
sumFalseAccept = sum(gyroscopeModel.getfalseAccept())
sumFalseReject = sum(gyroscopeModel.getfalseReject())
sumTrueAccept = sum(gyroscopeModel.gettrueAccept())
sumTrueReject = sum(gyroscopeModel.gettrueReject())
print('Confusion Matrix')
print(sumTrueReject, ' ',  sumFalseAccept)
print(sumFalseReject, ' ',  sumTrueAccept)

print('**********************************************************************')
accuracy_average = 0
f1score_average = 0
far_average = 0
frr_average = 0
roc_average = 0
for i in range(0, len(users)):
    accuracy_average = accuracy_average + ensembleModel.getAccuracy()[i]
    f1score_average = f1score_average +  ensembleModel.getf1score()[i] 
    far_average = far_average +  ensembleModel.getFAR()[i]
    frr_average = frr_average +  ensembleModel.getFRR()[i]

print()
print('AVERAGE ONE CLASS SVM PERFORMANCE ENSEMBLE:')
print('Accuracy: ', accuracy_average / len(users), '\nF1 Score: ', f1score_average / len(users), '\nFAR: ', far_average / len(users), '\nFRR: ', frr_average / len(users))
sumTest = sum(ensembleModel.getsizeTest())
sumFalseAccept = sum(ensembleModel.getfalseAccept())
sumFalseReject = sum(ensembleModel.getfalseReject())
sumTrueAccept = sum(ensembleModel.gettrueAccept())
sumTrueReject = sum(ensembleModel.gettrueReject())
print('Confusion Matrix')
print(sumTrueReject, ' ',  sumFalseAccept)
print(sumFalseReject, ' ',  sumTrueAccept)
