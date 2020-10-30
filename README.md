# Continuous implicit authentication of smartphone users using navigation data

This project proposes a methodology for continuous implicit authentication of smartphone users, using the navigation data, in order to improve security and ensure the privacy of sensitive personal data. I used Novelty Detection Machine Learning Algorithms with advanced preprocessing methods on noisy data from sensors. The results show that the proposed approach provides an additional level of security and privacy and can ensure that 99% of unauthorised users will be denied access to the device and the users' personal data.

Privacy and security are 2 interrelated concepts. Privacy refers to the right that the user has regarding the control of his/her personal information and how they are used. Security refers to the way personal data is protected from any unauthorized third-party access or malicious attacks. Smartphones contain a wealth of personal data such as photos, chats, medical data, bank details, personal passwords and information related to a person's close circle. It is of vital importance to protect the above information from third parties. The solution to this problem is achieved through continuous implicit authentication. The system processes the user’s behavior it collects from the sensors, as a background process. If the behavior does not belong to the owner of the device, the device is locked. This behavior protects the data and the device. Each user's behavior is unique. Subsequently, the device remains locked and personal data is protected until the correct behavior is recognized.

**This project is my Diploma Thesis for my Master's Degree in Artificial Intelligence and Data Science. Τhe complete description will be uploaded soon (but in Greek, only Abstract and Title available in English)**

# Description of the repository

The repository consists of the following 4 files.

- ContinuousImplicitAuthentication.py: This file includes the methodology of applying Novelty Detection Algorithms to recognise users bahaviour through accelerometer and gyroscope data 
- Sensors_and_GesturesContinuousImplicitAuthentication.py: This file includes the methodology of applying Novelty Detection Algorithms to recognise users bahaviour through accelerometer data,gyroscope data and gestures data
- metrics.py: Class that contains the calculated metrics for each model
- features.py: Class that contains the input features from each machine learning model
- dataHandler.py: Handles gestures data from MongoDB.
- MongoDBHandler.py: Used for Gestures Data. Connection to MongoDB.

# Prerequisites

## Dataset 

The dataset used for this project can be found here: https://zenodo.org/record/2598135#.X4H0qXgzbeo
The sensors_data.zip file includes JSON files with sensors measurements of 2000 smartphone users.
The gestures_devices_users_games_data.zip includes bson files (MongoDB) with gestures measurements of the same 2000 smartphone users.
More information about the dataset can be found here: https://res.mdpi.com/data/data-04-00060/article_deploy/data-04-00060.pdf?filename=&attachment=1

## Installation
In order to use this software, follow the steps below:
* **Step 1)** Create a new folder (e.g. sensors_data) 
* **Step 2)** For each user create a folder with its ID (e.g. 0hz8270) inside the previous folder. In the user's folder, move all of its JSON files.
        For example, user 0hz8270 has 67 JSON files. Move all of them inside folder 0hz8270. You don't have to use all the users. 
* **Step 3)** Set the variable "path" with the absolute path of folder sensors_data.
* **Step 4)** For Gestures data you will need to use MongoDB. Download and Install MongoDB.
* **Step 5)** Download gestures_devices_users_games_data.zip from https://zenodo.org/record/2598135#.X4H0qXgzbeo
* **Step 6)** Follow the instructions on page 12 https://res.mdpi.com/data/data-04-00060/article_deploy/data-04-00060.pdf?filename=&attachment=1
            in order to import the data set.
* **Step 7)** Start MongoDB.

## Execution

* Download Python Python 3.7.4 64bit
* Set the variables:
  - path: The absolute path of the folder created before (sensors_data)
  - screenName: The name of the game. BrainRun Dataset contains 5 games (ReactonGame, SpeedyGame, MathisisGame, FocusGame, MemoriaGame)
  - For Gestures data, in the file (Sensors_and_GesturesContinuousImplicitAuthentication.py) you should set the server URL and the database name (line 162)
* Execute the project. You can do it from an IDE or from a terminal.

```bash
python3 ContinuousImplicitAuthentication.py
```

# Optional
The process of reading JSON files takes a lot of time to execute. You can use a **Jupyter Notebook** and split the code into cells. Create a cell for the process of loading data and another cell(s) for the data preprocess and machine learning algorithms. In this way, the loading of the data happens only once and then you can do whatever you want with the models and the preprocess. (as long as you DO NOT CHANGE the gyroscope, accelerometer and Gestures Dataframes). The code as it is, does not do any modifications to above mention Dataframes and it is safe to split it.

# Methodology

Within the context of this study, the accelerometer and gyroscope sensors were selected to model the way a user interacts with its smartphone. The measurements were collected in an uncontrolled environment from an application downloaded from the Store. Two machine learning models were trained, one for each sensor and then, the results were combined to produce the final system’s performance. The performance of the final system exceeded the performance of the literature.

## Dataset Description

The dataset used is called BrainRun and is a set of biometric data for use in continuous implicit user authentication. A mobile application called BrainRun for Android and iOS mobile phones was developed for data collection. The application includes a list of brain training games through which data is collected from the sensors. BrainRun includes five different game-types, which are called “Focus”, “Mathisis”, “Memoria”, “Reacton”, and “Speedy”, and their main goal is the collection of users’ information. Each game-type is specifically designed to collect different kind of hand gestures, such as taps, horizontal swipes, vertical swipes, swipes and taps combined, etc. In addition, sensors measurements are collected from accelerometer, gyroscope kai magnetometer sensor.

## Data Preprocessing

The data received from the motion sensors are of the form (x, y, z, screen), where x, y, z is the position of the mobile according to the 3 axes and screen is the game that was recorded e.g. SpeedyGame. The accelerometer detects changes in the orientation of the mobile phone (rate of change of speed) in relation to the x, y and z axes, while the gyroscope helps the accelerometer to understand how the mobile phone is oriented, adding an extra level of accuracy.

The dataset, for each user, consists of an unknown number of JSON files. Each JSON file is also a timestamp, during which the data was collected. In addition to the values x, y, z, their derivatives are also calculated, such as the magnitude and the combined angle.

![Combine Angle and Magnitude Dimensions](https://github.com/AngelikiTsintzira/Continuous-implicit-authentication-of-smartphone-users-using-navigation-data/blob/main/figures/Magnitude.png?raw=true)

![Combine Angle and Magnitude Dimensions](https://github.com/AngelikiTsintzira/Continuous-implicit-authentication-of-smartphone-users-using-navigation-data/blob/main/figures/CombinedAngle.png?raw=true)

To select the appropriate variable, experiments were performed with each variable (x, y, z, combined angle, magnitude) and at the end, the performance of the algorithms was compared. From the experiments, it emerged that the variable **magnitude** was the one with which, the models achieved the best performance.

## Data Exploration

During the exploration, visualization and processing techniques were used to better understand and observe the data, which led to more accurate conclusions. Additionally, before the Novelty Detection algorithms were used, the simple SVM algorithm was used. The problem with SVM is that it requires data from both classes (device owner and attacker). This leads to the case of imbalanced data where one class has a large percentage of more data than the other. This problem was addressed by using Imputation Methods to generate new data according to some distribution. In the real world, only the data of the device owner is available, so there are no attackers data for training.

An example of 14 users for dimensions Y and Z are shown below. It is clear that there is a distinction between users' behaviour.

![Combine Angle and Magnitude Dimensions](https://github.com/AngelikiTsintzira/Continuous-implicit-authentication-of-smartphone-users-using-navigation-data/blob/main/figures/Y_axis.png?raw=true)

![Combine Angle and Magnitude Dimensions](https://github.com/AngelikiTsintzira/Continuous-implicit-authentication-of-smartphone-users-using-navigation-data/blob/main/figures/Z_axis.png?raw=true)

## Segmentation - Sliding Window

The data collected by the sensors is related to time. The sensors collect many measurements per second. Multiple measurements per second result in error correction, as if a measurement is not taken or if there is an extreme value due to an error, it can be corrected by simply checking the previous and next values. Sudden abrupt changes between the previous and next values indicate an error. In order to compress the data but also to eliminate these errors, a sliding window partitioning technique was chosen. The logic of the methodology followed is to create sections with data of equal size from which the features will be extracted. These sections will have some overlap with each other.

The values ​​to be selected are the window size and the overlap. There is no specific methodology for this. Most of the times, they are selected based on the sampling frequency of the sensors, while others are empirical. The BrainRun dataset has measurements from many different devices, which means that there is no specific sampling frequency during the measurements. For the selection of the above parameters, the literature was used in combination with an empirical study testing various values.

## Feature Extraction

Feature extraction is the feature selection process, which is considered important, in the sense that they provide valuable knowledge for drawing conclusions and correlations. Choosing the right features is a key factor in the performance of a model. The need to export key signal characteristics to enable advanced processing algorithms to discover useful information has led to the development of a wide range of algorithmic approaches. These approaches are based on converting input signals to and from different areas of representation. 

For this project, the final features that were chosen are shown below. From the time domain, I choose 9 features and from the frequency domain, I choose 4. I used correlation for dimensional reduction.

| Feature       | Description   | Domain        |
| ------------- | ------------- | ------------- |
| Mean  | Mean value  | Time  |
| STD  | Standard deviation  | Time  |
| Max  | Maximum value  | Time  |
| Min  | Minimum value  | Time  |
| Percentile25  | 25% quartiles  | Time  |
| Percentile50  | 50% quartiles  | Time  |
| Percentile75  | 75% quartiles  | Time  |
| Kurtosis  | Width of peak  | Time  |
| Skewness  | Orientation of peak | Time  |
| P1  | Amplitude of the first highest peak | Frequency  |
| F1  | Frequency of the second highest peak | Frequency  |
| P2  | Amplitude of the second highest peak  | Frequency  |
| Mean Frequency  | Mean value of frequencies  | Frequency  |

Once the feature vector is finished, I used normalization to transform data into [0,1]. Normalization is a rescaling of the data from the original range so that all values are within the new range of 0 and 1. Normalization is very important when we use machine learning algorithms based on distance. The scale of the data matters and data that are on a higher scale than others may contribute to the result more due to their larger value. So we normalize the data to bring all the variables to the same range.

## Training Algorithms

I used 4 Novelty Detection Algorithms:
- Local Outlier Factor 
- One Class SVM
- Isolation Forest
- Elliptic Envelope - Robust Covariance

For Hyperparameter Tuning, the Cross-Validation technique was applied in combination with the Grid Search technique. The purpose of cross-validation is to determine the learning parameters that generalize well. The right choice of hyper-parameters is extremely important to achieve a good performance of a model. The process of finding the right hyper-parameters is a process of Try and Error, as there is no mathematical way to show the appropriate value for each problem or there is no mathematical formula, which through the training data, can extract the appropriate prices.

The Hyperparameter Tuning procedure is shown below.

![Combine Angle and Magnitude Dimensions](https://github.com/AngelikiTsintzira/Continuous-implicit-authentication-of-smartphone-users-using-navigation-data/blob/main/figures/HyperParameterTuning.png?raw=true)

The following steps were applied:
- For each user
- - 5 repetitions
- - 1. Split dataset into 3 sub-sets, training, validation and test with a random way
- - 2. 10 Fold
- - 3. Test in the training set
- -  Choose the best hyper-parameters
- - Test on the validation set
- Test in the test set
- Final values of hyper-parameters

## Ensemble Learning

Ensemble learning improves the overall accuracy of the system by combining 2 or more models. The combination of models is achieved either by combining 2 or more models of different algorithms or by combining models of the same algorithms but with some differences.

Ensemble Learning is shown below.

![Combine Angle and Magnitude Dimensions](https://github.com/AngelikiTsintzira/Continuous-implicit-authentication-of-smartphone-users-using-navigation-data/blob/main/figures/EnsembleModels.png?raw=true)

Initially, the data of the 2 sensors (gyroscope and accelerometer) must be identical in time, ie each recording of the accelerometer must correspond to one recording of the gyroscope at the same time. Unidentified samples were discarded. The result of the above processing is 2 different datasets, which consist of the same set of data with measurements from the 2 sensors (one contains measurements from the accelerometer, while the other from the gyroscope) at the same time. 

The next step is the training of the 2 models. Each is trained separately with the corresponding dataset. The results are combined using the decision_function function. The decision_function function predicts the confidence (probability) scores for each sample. The confidence score for a sample is the distance of that sample from the separation surface. When its value for a sample is positive, it means that this sample belongs to the class of the real user, ie the positive class, while when its value for a sample is negative, it means that this sample is an extreme value, therefore it is classified as a malicious or unauthorized user. During the training process, the distance of each sample from the separation surface is calculated and the maximum distance for each model is found. 

Then, after completing the prediction process, divide the distance of each sample by the maximum distance calculated in the training and what emerges is a good estimate of the probability that the investigated sample belongs to the class of the actual user or to the class of the malicious user. Once the execution of the 2 models is completed, the process of combining the results of the models follows. During the combination, for each sample, the above estimate from the 2 models is added and if the sum value is positive the sample is categorized as a real user otherwise as a malicious user.

## Evaluation Metrics

Performance metrics are measures that quantify the performance of a machine learning model. The problem with this categorization is the classification of an initially unknown sample into one of two possible categories, that of the original user and that of the attacher one. As a result, there are 2 classes. The value 1 symbolizes the original user of the mobile phone, while the value -1 symbolizes the malicious user. Algorithms are trained in only one class, that of the original user. Therefore, any unknown sample that matches the distribution of the actual user data takes the value 1, otherwise the value -1. 

The primary goal of the continuous user authentication problem is to minimize **FAR**, the percentage of unknown users or intruders who are incorrectly classified as the original user and thus gain access to the device with unknown consequences. At the same time, it is also very important to minimize **FRR** as much as possible. The FRR measures the percentage of cases in which the original user is not recognized by the model and is therefore prohibited from using the device until it is switched on again. In addition, the **Confusion Matrix**, **Accuracy** and **F1-Score** metrics were used.

# Conclusions

- Motion sensor measurements (accelerometer, gyroscope) provide valuable information related to user behavior, capable of user identification only with navigation data.
- Proper data processing was considered vital. The correct selection of the sampling and overlap window significantly improved the performance of the algorithms. The appropriate selection of features, which are the independent variables, improved the performance of the models as they were the features, based on which the algorithms learned the distribution of the authorized user's behavior. The combination of data in the domain of time and frequency gave the best results.
- The behavior and performance fluctuations of the algorithms were kept constant in all experiments. This leads to the conclusion that the models that were trained, the algorithms that were selected, and the data processing that followed, led to strong and robust models that can be used in the real world.
- One Class SVM and Local Outlier Factor algorithms efficiently solve the problem of continuous implicit authentication. Their performance surpassed that of the literature.
- For applications that require a low acceptance rate of malicious or unauthorized users (<0.7%), the Local Outlier Factor algorithm is more appropriate.
- For applications that require a low actual user rejection rate (<5.7%) combined with a low malicious user acceptance rate (1.1%), the One Class SVM algorithm is considered appropriate.
- The One Class SVM and Local Outlier Factor algorithms achieved the best percentages of metric FAR compared to all literature studies, even those using the same dataset. The percentages of metric FRR were particularly low (4-6%) and comparable to the literature, but in some cases slightly higher.
- This study was applied to uncontrolled environmental data. Most of the literature studies were done in a controlled environment.
- **The results show that the proposed approach provides an additional level of security and privacy and can ensure that 99% of unauthorised users will be denied access to the device and the users' personal data.**