# Continuous implicit authentication of smartphone users using navigation data

This project, proposes a methodology for continuous implicit authentication of smartphones users, using the navigation data, in order to improve the security and ensure the privacy of sensitive personal data. I used Novelty Detection Machine Learning Algorithms with advanced preprocessing methods on noisy data from sensors. The results show that the proposed approach provides an additional level of security and privacy and can ensure that 99% of unauthorised users will be denied access to the device and the users personal data.

Privacy and security are 2 interrelated concepts. Privacy refers to the right that the user has regarding the control of his/her personal information and how they are used. Security refers to the way personal data is protected from any unauthorized third-party access or malicious attacks. Smartphones contain a wealth of personal data such as photos, chats, medical data, bank details, personal passwords and information related to person's close circle. It is of vital importance to protect the above information from third parties. The solution to this problem is achieved through continuous implicit authentication. The system processes the user’s behavior it collects from the sensors, as a background process. If the behavior does not belong to the owner of the device, the device is locked. This behavior protects the data and the device. Each user's behavior is unique. Subsequently, the device remains locked and personal data is protected until the correct behavior is recognized.

**This project is my Diploma Thesis for my Masters Degree in Artificial Intelligence and Data Science. Τhe complete description will be uploaded soon (but in Greek, only Abstract and Title available in English)**

# Description of the repository

The repository consists of the following 2 files.

- ContinuousImplicitAuthentication.py: This file includes the methodology of applying Novelty Detection Algorithms to revognise users bahaviour through accelerometer and gyroscope data 
- Sensors_and_GesturesContinuousImplicitAuthentication.py: This file includes the methodology of applying Novelty Detection Algorithms to revognise users bahaviour through accelerometer data,gyroscope data and gestures data

# Prerequisites

## Dataset 

The dataset used for this project can be found here: https://zenodo.org/record/2598135#.X4H0qXgzbeo
The sensors_data.zip file includes json files with sensors measurements of 2000 smartphones users.
The gestures_devices_users_games_data.zip includes bson files (MongoDB) with gestures measurements of the same 2000 smartphones users.
More information about the dataset can be found here: https://res.mdpi.com/data/data-04-00060/article_deploy/data-04-00060.pdf?filename=&attachment=1

## Installation
In order to use this software, follow the steps below:
* **Step 1)** Create a new folder (e.g. sensors_data) 
* **Step 2)** For each user create a folder with its ID (e.g. 0hz8270) inside the previous folder. In the user's folder, move all of its json files.
        You don't have to use all the users. For example, user 0hz8270 has 67 json files. Move all of them inside folder 0hz8270.
* **Step 3)** Set the variable "path" with the absosule path of folder sensors_data.
* **Step 4)** For Gestures data you will need to use MongoDB. Download and Install MongoDB.
* **Step 5)** Download gestures_devices_users_games_data.zip from https://zenodo.org/record/2598135#.X4H0qXgzbeo
* **Step 6)** Follow the instractions on page 12 https://res.mdpi.com/data/data-04-00060/article_deploy/data-04-00060.pdf?filename=&attachment=1
            in order to import the data set.
* **Step 7)** Start MongoDB.

## Execution

* Download Python Python 3.7.4 64bit
* Set the variables:
  - path: The absolute path of the folder created before (sensors_data)
  - For Gestures data, in file (Sensors_and_GesturesContinuousImplicitAuthentication.py) you should set the server URL and the database name (line 162)
* Execute the project. You can do it from an IDE or from terminal.
```bash
python3 ContinuousImplicitAuthentication.py
```

# Optional
The process of reading json files takes a lot of time to execute. You can use a **Jupyter Notebook** and split the code in cells. Create a cell for the process of loading data and another cell(s) for the data preprocess and machine learning algorithms. With this way, the loading of the data happens only once and then you can do whatever you want with the models and the preprocess. (as long as you DO NOT CHANGE the gyroscope, accelerometer and Gestures Dataframes).  The code as it is, does not do any modifications to above mention Dataframes and it is safe to split it.

# Methodology

Within the context of this study, the accelerometer and gyroscope sensors were selected to model the way a user interacts with its smartphone. The measurements were collected in uncontrolled environment from an application downloaded from the Store. Two machine learning models were trained, one for each sensor and then, the results were combined to produce the final system’s performance. The performance of the final system exceeded the performance of the literature.