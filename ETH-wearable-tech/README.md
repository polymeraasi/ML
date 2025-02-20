# ETH - Wearable Technology

Group exercises from wearable technology course at ETH Zurich.

Authors: Axel Hedman, Pinja Koivisto & Maja Wahlin

## lab3.ipynb - EEG & IMU Data Processing & Analysis
Data: EEG_IMU folder which contains IMU and EEG signals corresponding to two trials of data recording, chanloc.mat which stored the location of 8 EEG electrodes. The participant was instructed to perform two types of activities: writing name and bringing their hand to mouth, mimicking eating. The EEG data was stored in the BioRadioData variable (8 channels). The participant was wearing a 3-axes accelerometer (IMU) on the hand recording acceleration of hand motion during each activity.

Task: Filtering and cleaning the EEG signals, Common Spatial Pattern (CSP) for feature extraction from EEG signals. LDA & SVM for classifying (writing vs. eating). 

## lab4.ipynb - ECG and Physiological Signal Processing & Analysis
Data: anxietyData.mat files which contains ECG, respiratory and electrodermal activity signals corresponding to 55 volunteers in a base and anxiolytic state. 

Task: Filtering and cleaning the ECG signal, detection of the QRS complex in the ECG with Pan & Tompkins, PCA for feature reduction. LR, RF & SVM for classification (to differentiate between non-anxiety and anxiety states). 

