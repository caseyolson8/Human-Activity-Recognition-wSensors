<html>
<head>
<style>
h1 {text-align: center;}
p {text-align: center;}
div {text-align: center;}
</style>
</head>
<body>

<h1>SPHERE/AARP Dataset: Human Activity Recognition using multimodal sensor data</h1>

</body>
</html>

<<<<<<< HEAD
=======
# SPHERE/AARP Dataset: Human Activity Recognition using multimodal sensor data

>>>>>>> 4c8c38569919980c19e096669f0afe547d9c39d0

<p align="center">
  <img src="reports//figures/wearable_accelerometer.jpg">
</p>

## The Problem: Aging, Mobility, and Disease
Obesity, depression, cardiovascular and musculoskeletal diseases are often associated with aging & are exarcerbated by reduced mobility. Expenses from a larger aged population are placing an ever increasing financial strain on health care system across the globe.  Tracking the daily activities of a patient at home and throughout the day would aide healthcare professionals in early diagnoses, detecting lifestyle change and thus aiding the ability of patients to live in good health at home.  Sensors and data science methods have the potential to quantify daily living activities unobtrusively and without invasion of privacy.

## Project Goal:
Using multimodal sensor data (provided via a competition by DrivenData.org) train a model to accurately predict activities from 20 predefined classes.

## Description of Raw Data:
This pre-labeled dataset is from a closed competition at [drivendata](https://www.drivendata.org/competitions/42/senior-data-science-safe-aging-with-sphere/), sponsored by the [AARP](https://www.aarp.org/aarp-foundation/) (American Association for Retired Persons), and managed by [IRC-SPHERE](https://www.irc-sphere.ac.uk/).  All data is presented as a time series monitoring numerous subjects (separately) as they perform tasks in a standardized home-like environment.  Primary data is from a wearable 3D accelerometer which transmits samples at 20Hz via bluetooth to any of 3 receivers placed in the 'home', provided in its raw form.  Secondary synchronized contextual data comes from 1) infrared proximity sensors placed in rooms around the home to detect the location of the subject and 2) 3 RGB-D cameras in 3 rooms in the home, which have been pre-processed to provide only the coordinates of a 3D bounding box of the subject when located within the camera frame (bounding calculated using [OpenNI_library](https://en.wikipedia.org/wiki/OpenNI)).  
Each data point in the time-series have been annotated with one of 20 typical human-activities (e.g. walking, sitting, ascending_stairs) separately by 10 trained observers to create a probabalistic target (frequency of observation) for each time point.

For simplicity in Capstone_2 analysis has been limited to accelerometer data, filtered to 3 classes of activity: ```[stand, walk, sit]```.  The target has been simplified to the most likely class at any given time.


## Featurization:
![alt text](reports//figures/accel_examples.png)
Time series were split by the labelled activity into isolated activity blocks and aggregate features derived over 1 second windows, with a 0.5 s overlap.  
The following features were derived from each window and represent the columns on our feature matrix:
1. Mean
2. Standard deviation
3. Root mean square
4. Zero crossing rate - count of the signal crossing the average value
5. Absolute difference - sum of abs difference in sample and mean
6. Spectral energy - total energy in signal via fast fourier transform (FFT)
7. FFT coefficients - via FFT, represents magnitude of signal at frequencies:
    - 0, 2, 4, 6, 8 Hz  

Aggregates were calculated for each axis separately (x,y,z) as well as the L2 norm of the x,y,z vector to give a total of 44 features for each window.


## Logistic Regression model:
As a first attempt a hard classifier was used to predict the class of the most likely action for each window.  The LogisticRegression class from sklearn was implemented with a 'one vs all' approach for multiple classes. Class specific weights were applied to the cost function to counteract the imbalance in classes on the training dataset;  
-  ***w_class = 1 - (n_class/n_total)***.  

The feature space was normalized (mean: 0, std: 1).  As this model was intended to highlight useful features it was trained on the entire dataset.  
  Accuracy on the training set was 0.64, predicted number of occurences in each class: 
<p align="center">

|Class     |     Predicted Freq.   |  True Freq. |
|---------:|:-------------:   |  :--:            |
| Stand :  |         8896     |   9868           |
| Sit :    |         5405     |   4460           |
| Walk :   |         2300     |   2273           |

![alt text](reports/figures/pred_actual_classes.png)
</p>
<n></n>   
<n></n>  

### **Metrics to distinguish class Sit**:

|Metric    |       odds for |||Metric    |   odds against |  
|---------:|:---------------|--|--|--------:| :---------------|
| RMS_L2 :   |         10.14 |||ABS_L2 :    |        0.24 |
| RMS_X :    |         4.51 ||| Mean_L2 :   |        0.37 |
| RMS_Y :    |         2.79 ||| FFT5_L2_0 : |        0.37 |
| Sp_Energy_Z : |      2.52 ||| Sp_Energy_X : |      0.53 |
| FFT5_X_1 : |         1.80 ||| Std_X :    |         0.54 |
| RMS_Z :    |         1.73 ||| FFT5_Z_0  :|         0.56 |    
<n></n>   
<n></n>  


### **Metrics to distinguish class Stand**:  
  
| Metric          |   odds for |||Metric          |   odds against |
|--------:    |:----------- |-|-|-------:|:--------------|
| RMS_L2 :      |       7.42  ||| RMS_Y    : |          0.30 |
| Sp_Energy_Y : |       2.71  ||| FFT5_L2_0  :|         0.41 |
| FFT5_Y_0  :   |       1.91  ||| Mean_L2   : |         0.41 |
| Sp_Energy_Z  :|       1.84  ||| Sp_Energy_L2 : |      0.50 |
| Sp_Energy_X  :|       1.69  ||| RMS_Z     :|          0.62 |
| Std_Y      :  |       1.60  ||| FFT5_X_0  :|          0.74 |  
<n></n>   
<n></n>  


### **Metrics to distinguish class Walk**:  
| Metric          |  odds for ||| Metric  |   odds against |
|---------: | :-----------|--|--|--------: | :----------------|
| Mean_L2   : |      6.69 ||| RMS_L2          :|        0.01 |
| FFT5_L2_0  :|      6.69 ||| RMS_X         :  |        0.14  |
| ABS_L2     :|      3.89 ||| Sp_Energy_Z   :  |        0.21  |
| FFT5_X_0  :|       2.39 ||| Sp_Energy_Y   :  |        0.29  |
| Sp_Energy_L2  :|   2.21 ||| ABS_Z         :  |        0.59  |
| Std_X    : |       1.78 ||| FFT5_X_1    :    |        0.69  |


## Lasso Regularization
Logistic Regression was performed with lasso regularization with regularization weights spanning from 0.01 to 100 to understand which parameters may have little influence.  The difference in performance on the training set between a model with none and with strong regularization was 0.64 and 0.63 accuracy.
<center>

![alt text](reports/figures/Lasso_Regularization.png)

</center>

This is a messy graph but you can see many features with large coefficients when no regularization willl go to zero fairly quickly.

## Future Work
- Clear up the pipeline to qualify features on usefulness
- Add data from video modalities to decision process:
    - video would clearly highlight the difference between sitting/standing and walking for example
- Create more features
- Train more models, which perform the best?


--------



### Resources


(http://github.com/OpenNI/OpenNI) - automatic detection of humans via RGB-D  
(https://www.asus.com/us/3D-Sensor/Xtion_PRO_LIVE/) - the rgb-d camera  
(https://github.com/IRC-SPHERE/sphere-challenge) - SPHERE github  




<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
