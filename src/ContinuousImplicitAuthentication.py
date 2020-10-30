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
from features import Features

# =============================================================================
# Functions
# =============================================================================

# This function extract features with the method sliding window.
def FeatureExtraction(dataset, samples, overlap, output, feautureObject):

    for w in range(0, dataset.shape[0] - samples, overlap):
        end = w + samples 

        #DFT
        discreteFourier = fft(dataset.iloc[w:end, feauture])
        # Frequencies
        freq = np.fft.fftfreq(samples)

        # Amplitudes
        idx = (np.absolute(discreteFourier)).argsort()[-2:][::-1]
        amplitude1 = np.absolute(discreteFourier[idx[0]])
        amplitude2 = np.absolute(discreteFourier[idx[1]])
        frequency2 = freq[idx[1]]

        # Frequency features
        mean_frequency = np.mean(freq)
        feautureObject.setAmplitude1(amplitude1)
        feautureObject.setAmplitude2(amplitude2)
        feautureObject.setFrequency2(frequency2)
        feautureObject.setMean_frequency(mean_frequency)

        # Time Based Feautures
        feautureObject.setΜean(np.mean(dataset.iloc[w:end, feauture]))
        feautureObject.setSTD(np.std(dataset.iloc[w:end, feauture]))
        feautureObject.setMax(np.max(dataset.iloc[w:end, feauture]))
        feautureObject.setMin(np.min(dataset.iloc[w:end, feauture]))
        feautureObject.setRange(np.ptp(dataset.iloc[w:end, feauture]))

        percentile = np.percentile(dataset.iloc[w:end, feauture], [25, 50, 75])
        feautureObject.setPercentile25(percentile[0])
        feautureObject.setPercentile50(percentile[1])
        feautureObject.setPercentile75(percentile[2])
        feautureObject.setEntropy(entropy(dataset.iloc[w:end, feauture], base = 2))

        feautureObject.setKurtosis(kurtosis(dataset.iloc[w:end, feauture]))
        feautureObject.setSkewness(skew(dataset.iloc[w:end, feauture]))

        # Output Label
        feautureObject.setY(output)

    return  feautureObject

# This function calculates average metrics per model. (FAR, FRR, Confucion Matrix, Accuracy and F1-Score)
def PerformanceMetrics(users, model, text):

    accuracy_average = 0
    f1score_average = 0
    far_average = 0
    frr_average = 0
    roc_average = 0
    for i in range(0, len(users)):
        accuracy_average = accuracy_average + model.getAccuracy()[i]
        f1score_average = f1score_average +  model.getf1score()[i] 
        far_average = far_average +  model.getFAR()[i]
        frr_average = frr_average +  model.getFRR()[i]

    print()
    print('AVERAGE ONE CLASS SVM PERFORMANCE MODEL: ' + text)
    print('Accuracy: ', accuracy_average / len(users), '\nF1 Score: ', f1score_average / len(users), '\nFAR: ', far_average / len(users), '\nFRR: ', frr_average / len(users))
    sumTest = sum(model.getsizeTest())
    sumFalseAccept = sum(model.getfalseAccept())
    sumFalseReject = sum(model.getfalseReject())
    sumTrueAccept = sum(model.gettrueAccept())
    sumTrueReject = sum(model.gettrueReject())
    print('Confusion Matrix')
    print(sumTrueReject, ' ',  sumFalseAccept)
    print(sumFalseReject, ' ',  sumTrueAccept)

# Local Outlier Factor Algorithm Execution
def LocalOutlierFactorAlgorithm(parameters, X_train, X_test):

    model = LocalOutlierFactor(n_neighbors = parameters[0], novelty = True)
    model.fit(X_train)
    decision = model.decision_function(X_train)
    maxDistance = max(decision) 
    prediction = model.predict(X_test)

    decision = model.decision_function(X_test)
    decision = decision /maxDistance

    return decision, prediction

# Elliptic Envelope Algorithm Execution
def EllipticEnvelopeAlgorithm(parameters, X_train, X_test):

    model = EllipticEnvelope(contamination = parameters[0]).fit(X_train)
    decision = model.decision_function(X_train)
    maxDistance = max(decision) 
    prediction = model.predict(X_test)

    decision = model.decision_function(X_test)
    decision = decision /maxDistance

    return decision, prediction

# Isolation Forest Algorithm Execution
def IsolationForestAlgorithm(parameters, X_train, X_test):

    model = IsolationForest(n_jobs = -1, n_estimators = parameters[0], contamination = parameters[1], bootstrap = False).fit(X_train)
    decision = model.decision_function(X_train)
    maxDistance = max(decision) 
    prediction = model.predict(X_test)

    decision = model.decision_function(X_test)
    decision = decision /maxDistance

    return decision, prediction

# One Class SVM Algorithm Execution
def OneClassSVMAlgorithm(parameters, X_train, X_test):

    model = svm.OneClassSVM(gamma = parameters[0], kernel = 'rbf', nu = parameters[1], cache_size = 500)
    model.fit(X_train)
    decision = model.decision_function(X_train)
    maxDistance = max(decision) 
    prediction = model.predict(X_test)

    decision = model.decision_function(X_test)
    decision = decision /maxDistance

    return decision, prediction

# This function executed the Machine Learning Algorithm
def AlgorithmExecution(trainData, testData, algorithm, parameters, text, model):

    # Split Dataset in a random way 
    percent = int(math.ceil(0.2 * trainData.shape[0]))
    sampling = trainData.sample(n = percent)
        
    indecies = []
    for ii in range(0, percent):
        indecies.append(sampling.iloc[ii,:].name)

    test = pd.concat([testData, sampling])
    train = trainData.drop(trainData.index[indecies])

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

    # Call the appropriate algorithm, loop throught the functions, find the desired algorithm and execute it
    functions = (LocalOutlierFactorAlgorithm, EllipticEnvelopeAlgorithm, IsolationForestAlgorithm, OneClassSVMAlgorithm)
    for func in functions:
        functionName = func.__name__
        if algorithm in functionName:
            decision, prediction = func(parameters, X_train_norm, X_test_norm)
        
    print("******************************* Model " + text + " *******************************")
    score = f1_score(y_test, prediction, pos_label = 1)
    #print('F1 Score: %.3f' % score)
    acc = accuracy_score(y_test, prediction)
    #print(f'SVM accuracy is {acc}')
    cfm = confusion_matrix(y_test, prediction, labels = [-1, 1])
    print(cfm)
    np.sum(y_test == -1)
    far = cfm[0,1]/ np.sum(y_test == -1)
    frr = cfm[1,0]/ np.sum(y_test == 1)
    print('FAR: ', far, ' FFR: ', frr)
    
    model.setFAR(far)
    model.setFRR(frr)
    model.setAccuracy(acc)
    model.setf1score(score)
    model.setfalseAccept(cfm[0,1])
    model.setfalseReject(cfm[1,0])
    model.settrueAccept(cfm[1,1])
    model.settrueReject(cfm[0,0])
    model.setsizeTest(y_test.shape[0])

    # AUC represents the probability that a random positive (green) example is positioned to the right of a random negative (red) example.
    roc = roc_auc_score(y_test, prediction)
    #print('ROC AUC Score: ',roc)

    return decision, model, y_test

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
    
    # Objects with features for each model
    accelerometerFeatures = Features()
    gyroscopeFeatures = Features()

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

    accelerometerFeatures = FeatureExtraction(train, samples, overlap, 1, accelerometerFeatures)
    gyroscopeFeatures = FeatureExtraction(train_gyroscope, samples, overlap, 1, gyroscopeFeatures)

    accelerometerFeatures = FeatureExtraction(test, samples, overlap, -1, accelerometerFeatures)
    gyroscopeFeatures = FeatureExtraction(test_gyroscope, samples, overlap, -1, gyroscopeFeatures)

    # =============================================================================
    # Machine Learning Models
    # ============================================================================= 

    # Add features to dataframe
    df['Mean'] = accelerometerFeatures.getMean() 
    df['Std'] = accelerometerFeatures.getSTD()
    df['Skew'] = accelerometerFeatures.getSkewness()
    df['Kurtosis'] = accelerometerFeatures.getKurtosis()
    df['Max'] = accelerometerFeatures.getMax()
    df['Min'] = accelerometerFeatures.getMin()
    #df['Range'] = Range
    df['Percentile25'] = accelerometerFeatures.getPercentile25()
    df['Percentile50'] = accelerometerFeatures.getPercentile50()
    df['Percentile75'] = accelerometerFeatures.getPercentile75()
    #df['Entropy'] = Entropy

    df['Amplitude1'] = accelerometerFeatures.getAmplitude1()
    df['Amplitude2'] = accelerometerFeatures.getAmplitude2()
    df['Frequency'] = accelerometerFeatures.getFrequency2()
    df['MeanFrequency'] = accelerometerFeatures.getMean_frequency()

    df_gyroscope['Mean_Gyroscope'] = gyroscopeFeatures.getMean()
    df_gyroscope['Std_Gyroscope'] = gyroscopeFeatures.getSTD()
    df_gyroscope['Skew_Gyroscope'] = gyroscopeFeatures.getSkewness()
    df_gyroscope['Kurtosis_Gyroscope'] = gyroscopeFeatures.getKurtosis()
    df_gyroscope['Max_Gyroscope'] = gyroscopeFeatures.getMax()
    df_gyroscope['Min_Gyroscope'] = gyroscopeFeatures.getMin()
    #df_gyroscope['Range_Gyroscope'] = Range_Gyroscope
    df_gyroscope['Percentile25_Gyroscope'] = gyroscopeFeatures.getPercentile25()
    df_gyroscope['Percentile50_Gyroscope'] = gyroscopeFeatures.getPercentile50()
    df_gyroscope['Percentile75_Gyroscope'] = gyroscopeFeatures.getPercentile75()
    #df_gyroscope['Entropy_Gyroscope'] = Entropy_Gyroscope

    df_gyroscope['Amplitude1_Gyroscope'] = gyroscopeFeatures.getAmplitude1()
    df_gyroscope['Amplitude2_Gyroscope'] = gyroscopeFeatures.getAmplitude2()
    df_gyroscope['Frequency_Gyroscope'] = gyroscopeFeatures.getFrequency2()
    df_gyroscope['MeanFrequency_Gyroscope'] = gyroscopeFeatures.getMean_frequency()

    df['y'] = accelerometerFeatures.getY()
    df_gyroscope['y_gyroscope'] = gyroscopeFeatures.getY()

    train =  pd.DataFrame(df[df['y'] == 1])
    test =  pd.DataFrame(df[df['y'] == -1])
    
    train_gyroscope =  pd.DataFrame(df_gyroscope[df_gyroscope['y_gyroscope'] == 1])
    test_gyroscope =  pd.DataFrame(df_gyroscope[df_gyroscope['y_gyroscope'] == -1])

    print('After Sampling Sizes\n','Train Size One Class: ', train.shape[0], 'Test Size Two Class: ', test.shape[0])
    
    trainStarter = train
    testStarter = test
        
    trainStarter_gyroscope = train_gyroscope
    testStarter_gyroscope = test_gyroscope

    accelerometerModelUser = Metrics()
    gyroscopeModelUser = Metrics()
    ensembleModelUser = Metrics()
    
    # 10 Executions per original user
    for fold in range(0,10):

        # =============================================================================
        # Best Hyper-Parameters values for Model 1 - Accelerometer
        # ============================================================================= 
        '''
        LocalOutlierFactor(n_neighbors = 3, novelty = True)
        EllipticEnvelope(contamination = 0)
        IsolationForest(n_jobs = -1, n_estimators = 100, contamination = 0, bootstrap = False)
        svm.OneClassSVM(gamma = 0.1, kernel="rbf", nu = 0.01, cache_size = 500)
        '''

        # =============================================================================
        # Best Hyper-Parameters values for Model 2 - Gyroscope
        # ============================================================================= 
        '''
        LocalOutlierFactor(n_neighbors = 5, novelty = True)
        EllipticEnvelope(contamination = 0)
        IsolationForest(n_jobs = -1, n_estimators = 100, max_features = 1, contamination = 0, bootstrap = False)
        svm.OneClassSVM(gamma = 0.001, kernel="rbf", nu = 0.1, cache_size = 500)
        '''
        decision1 = []  
        decision2 = []
        decision1, accelerometerModelUser, y_test= AlgorithmExecution(trainStarter, testStarter, "LocalOutlierFactor", [3], "Accelerometer", accelerometerModelUser)
        decision2, gyroscopeModelUser, y_test = AlgorithmExecution(trainStarter_gyroscope, testStarter_gyroscope, "LocalOutlierFactor", [5], "Gyroscope", gyroscopeModelUser)

        # =============================================================================
        # Enseble Models
        # ============================================================================= 
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

        ensembleModelUser.setfalseAccept(cfm[0,1])
        ensembleModelUser.setfalseReject(cfm[1,0])
        ensembleModelUser.settrueAccept(cfm[1,1])
        ensembleModelUser.settrueReject(cfm[0,0])
        ensembleModelUser.setsizeTest(y_test.shape[0])
        ensembleModelUser.setAccuracy(acc)
        ensembleModelUser.setFAR(far)
        ensembleModelUser.setFRR(frr)
        ensembleModelUser.setf1score(score)

    # =============================================================================
    # Average Performance per user (10-FOLD)
    # =============================================================================     
    
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

        far = far + ensembleModelUser.getFAR()[index]  
        frr = frr + ensembleModelUser.getFRR()[index]  
        accu = accu + ensembleModelUser.getAccuracy()[index]  
        fscore = fscore + ensembleModelUser.getf1score()[index]
        FA = FA + ensembleModelUser.getfalseAccept()[index]
        FalseR = FalseR + ensembleModelUser.getfalseReject()[index]
        TrueA = TrueA + ensembleModelUser.gettrueAccept()[index]
        TrueR = TrueR + ensembleModelUser.gettrueReject()[index]
        testS = testS + ensembleModelUser.getsizeTest()[index]

        far1 = far1 + accelerometerModelUser.getFAR()[index]
        frr1 = frr1 + accelerometerModelUser.getFRR()[index]
        accu1 = accu1 + accelerometerModelUser.getAccuracy()[index]
        fscore1 = fscore1 + accelerometerModelUser.getf1score()[index]
        FA1 = FA1 + accelerometerModelUser.getfalseAccept()[index]
        FalseR1 =  FalseR1 + accelerometerModelUser.getfalseReject()[index]
        TrueA1 = TrueA1 + accelerometerModelUser.gettrueAccept()[index]
        TrueR1 = TrueR1 + accelerometerModelUser.gettrueReject()[index]
        testS1 = testS1 + accelerometerModelUser.getsizeTest()[index]

        far2 = far2 + gyroscopeModelUser.getFAR()[index]
        frr2 = frr2 + gyroscopeModelUser.getFRR()[index]
        accu2 = accu2 + gyroscopeModelUser.getAccuracy()[index]
        fscore2 = fscore1 + gyroscopeModelUser.getf1score()[index]
        FA2 = FA2 +  gyroscopeModelUser.getfalseAccept()[index]
        FalseR2 =  FalseR2 + gyroscopeModelUser.getfalseReject()[index]
        TrueA2 = TrueA1 +  gyroscopeModelUser.gettrueAccept()[index]
        TrueR2 = TrueR1 +  gyroscopeModelUser.gettrueReject()[index]
        testS2 = testS2 +  gyroscopeModelUser.getsizeTest()[index]

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
PerformanceMetrics(users, accelerometerModel, "ACCELEROMETER")
PerformanceMetrics(users, gyroscopeModel, "GYROSCOPE")
PerformanceMetrics(users, ensembleModel, "ENSEMLBE")
