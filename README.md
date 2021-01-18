
# SPHERE/AARP Dataset: Human Activity Recognition using multimodal sensor data


<p align="center">
  <img src="reports//figures/wearable_accelerometer.jpg">
</p>

## The Problem: Aging, Mobility, and Disease
Obesity, depression, cardiovascular and musculoskeletal diseases are often associated with aging & are exarcerbated by reduced mobility. Expenses from a larger aged population are placing an ever increasing financial strain on health care system across the globe.  Tracking the daily activities of a patient at home and throughout the day would aide healthcare professionals in early diagnoses, detecting lifestyle change and thus aiding the ability of patients to live in good health at home.  Sensors and data science methods have the potential to quantify daily living activities unobtrusively and without invasion of privacy.

## Project Goal:
Using multimodal sensor data (public dataset by [IRC-SPHERE](https://www.irc-sphere.ac.uk/) train a model to accurately predict activities from 20 predefined classes.

## Description of Raw Data:
This pre-labeled dataset is from a closed competition at [drivendata](https://www.drivendata.org/competitions/42/senior-data-science-safe-aging-with-sphere/), sponsored by the [AARP](https://www.aarp.org/aarp-foundation/) (American Association for Retired Persons), and managed by [IRC-SPHERE](https://www.irc-sphere.ac.uk/).  All data is presented as a time series monitoring numerous subjects (separately) as they perform tasks in a standardized home-like environment.  Primary data is from a wearable 3D accelerometer which transmits samples at 20Hz via bluetooth to any of 3 receivers placed in the 'home', provided in its raw form.  Secondary synchronized contextual data comes from 1) infrared proximity sensors placed in rooms around the home to detect the location of the subject and 2) 3 RGB-D cameras in 3 rooms in the home, which have been pre-processed to provide only the coordinates of a 3D bounding box of the subject when located within the camera frame (bounding calculated using [OpenNI_library](https://en.wikipedia.org/wiki/OpenNI)).  
Each data point in the time-series have been annotated with one of 20 typical human-activities (e.g. walking, sitting, ascending_stairs) separately by 10 trained observers to create a probabalistic target (frequency of observation) for each time point.




### References

Niall Twomey, Tom Diethe, Meelis Kull, Hao Song, Massimo Camplani, Sion Hannuna, Xenofon Fafoutis, Ni Zhu, Pete Woznowski, Peter Flach, and Ian Craddock. The SPHERE Challenge: Activity Recognition with Multimodal Sensor Data. 2016.

(http://github.com/OpenNI/OpenNI) - automatic detection of humans via RGB-D  
(https://www.asus.com/us/3D-Sensor/Xtion_PRO_LIVE/) - the rgb-d camera  
(https://github.com/IRC-SPHERE/sphere-challenge) - SPHERE github  




<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
