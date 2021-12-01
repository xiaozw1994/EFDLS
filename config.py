import os 
import numpy as np
################# Root Files
data_files = '/home/josen/deep learning/Caps Time Series/dataset/UCRArchive_2018'
###### 128 UCR datasets
sub_dir_name = ['DodgerLoopWeekend', 'CricketX', 'FiftyWords', 'FaceFour', 'GestureMidAirD1', 'Wine', 'MixedShapesSmallTrain', 
'InsectEPGSmallTrain', 'Rock', 'MiddlePhalanxOutlineAgeGroup', 'StarLightCurves', 'ChlorineConcentration', 'CBF', 
'InsectEPGRegularTrain', 'TwoLeadECG', 'ECGFiveDays', 'Chinatown', 'DodgerLoopGame', 'ToeSegmentation2', 
'ElectricDevices', 'Trace', 'Haptics', 'Symbols', 'Lightning2', 'MixedShapesRegularTrain', 'LargeKitchenAppliances',
 'ShapeletSim', 'HouseTwenty', 'Mallat', 'OliveOil', 'HandOutlines', 'Strawberry', 'MoteStrain', 'GunPoint', 'EOGHorizontalSignal', 
 'BirdChicken', 'PigCVP', 'OSULeaf', 'GesturePebbleZ1', 'FreezerSmallTrain', 'SmallKitchenAppliances', 'ECG5000', 
 'Fungi', 'UMD', 'GesturePebbleZ2', 'UWaveGestureLibraryZ', 'ShapesAll', 'Plane', 'Lightning7', 'DistalPhalanxTW', 
 'SyntheticControl', 'Fish', 'GunPointOldVersusYoung', 'ECG200', 'InsectWingbeatSound', 'DistalPhalanxOutlineCorrect', 
 'GestureMidAirD3', 'Beef', 'ToeSegmentation1', 'PigArtPressure', 'Phoneme', 'RefrigerationDevices', 'SmoothSubspace', 
 'FordB', 'ArrowHead', 'MedicalImages', 'SemgHandGenderCh2', 'Adiac', 'SemgHandSubjectCh2', 'PowerCons', 
 'UWaveGestureLibraryAll', 'DiatomSizeReduction', 'WordSynonyms', 'GestureMidAirD2', 'SemgHandMovementCh2', 'Herring', 
 'SonyAIBORobotSurface1', 'PickupGestureWiimoteZ', 'EthanolLevel', 'PhalangesOutlinesCorrect', 'SonyAIBORobotSurface2', 
 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'Ham', 'PigAirwayPressure', 'PLAID', 'GunPointAgeSpan', 'ProximalPhalanxOutlineCorrect',
  'ProximalPhalanxTW', 'TwoPatterns', 'AllGestureWiimoteZ', 'Coffee', 'NonInvasiveFetalECGThorax2', 'DodgerLoopDay', 'InlineSkate', 
  'Earthquakes', 'Car', 'Crop', 'DistalPhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'SwedishLeaf', 
  'NonInvasiveFetalECGThorax1', 'Wafer', 'FaceAll', 'MelbournePedestrian', 'Computers', 'ShakeGestureWiimoteZ', 
  'WormsTwoClass', 'EOGVerticalSignal', 'GunPointMaleVersusFemale', 'Worms', 'FreezerRegularTrain', 'ScreenType', 
  'CinCECGTorso', 'FordA', 'UWaveGestureLibraryX', 'BME', 'ProximalPhalanxOutlineAgeGroup', 'BeetleFly', 'ACSF1', 'Meat', 
  'FacesUCR', 'CricketZ', 'Yoga', 'MiddlePhalanxTW', 'CricketY', 'ItalyPowerDemand', 'UWaveGestureLibraryY']

long_dir_name = ['FordA','FordB','ShapeletSim','BeetleFly','BirdChicken','Earthquakes','Herring','ShapesAll',
'OliveOil','Car','InsectEPGRegularTrain','InsectEPGSmallTrain','Lightning2','Computers','LargeKitchenAppliances','RefrigerationDevices',
'ScreenType,''SmallKitchenAppliances','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2','Worms','WormsTwoClass',
'UWaveGestureLibraryAll','Mallat','Phoneme','StarLightCurves','MixedShapesRegularTrain','MixedShapesSmallTrain',
'Haptics','EOGHorizontalSignal','EOGVerticalSignal','ACSF1','SemgHandGenderCh2','SemgHandMovementCh2','SemgHandSubjectCh2',
'CinCECGTorso','EthanolLevel','InlineSkate','HouseTwenty','PigAirwayPressure','PigArtPressure','PigCVP','HandOutlines','Rock']

vary_dir_name = ['AllGestureWiimoteX','AllGestureWiimoteY','AllGestureWiimoteZ','GestureMidAirD1','GestureMidAirD2','GestureMidAirD3',
'GesturePebbleZ1','GesturePebbleZ2','PickupGestureWiimoteZ','PLAID','ShakeGestureWiimoteZ']

longer_vary_dir_name = ['Mallat','Phoneme','StarLightCurves','MixedShapesRegularTrain','MixedShapesSmallTrain',
'Haptics','EOGHorizontalSignal','EOGVerticalSignal','ACSF1','SemgHandGenderCh2','SemgHandMovementCh2','SemgHandSubjectCh2',
'CinCECGTorso','EthanolLevel','InlineSkate','HouseTwenty','PigAirwayPressure','PigArtPressure','PigCVP','HandOutlines','Rock','AllGestureWiimoteX','AllGestureWiimoteY','AllGestureWiimoteZ','GestureMidAirD1','GestureMidAirD2','GestureMidAirD3',
'GesturePebbleZ1','GesturePebbleZ2','PickupGestureWiimoteZ','PLAID','ShakeGestureWiimoteZ']

each_elen_dir_name = [
'Chinatown','MelbournePedestrian','SonyAIBORobotSurface2','SonyAIBORobotSurface1'
,'DistalPhalanxOutlineAgeGroup','DistalPhalanxOutlineCorrect','DistalPhalanxTW','TwoLeadECG','MoteStrain','ECG200','CBF',
  'DodgerLoopDay','DodgerLoopGame','DodgerLoopWeekend','CricketX','CricketY','CricketZ','Ham','Meat','Fish','Beef',
'FaceFour','OliveOil','Car','Lightning2','Computers',
  'Mallat','Phoneme','StarLightCurves','MixedShapesRegularTrain','MixedShapesSmallTrain','ACSF1','SemgHandGenderCh2',
  'AllGestureWiimoteX','AllGestureWiimoteY','AllGestureWiimoteZ','GestureMidAirD1','GestureMidAirD2','GestureMidAirD3',
'GesturePebbleZ1','GesturePebbleZ2','PickupGestureWiimoteZ','PLAID','ShakeGestureWiimoteZ']


New_dir = "./data/"
#_Xtrain.npy
#_Ytrain.npy

expand = 1
sub_expand = 1
loc = 0.05
scale = 0.05
locslist = [0.05,0.06,0.07,0.08,0.09,0.10,0.20] 
scalelist = [0.05,0.06,0.07,0.08,0.09,0.10,0.20] 
locslist = [i*expand for i in locslist]
scalelist = [i*sub_expand for i in scalelist]
per_class = [1,2,5,10,20,100,200]