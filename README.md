# An Efficient Federated Distillation Learning System for Time Series Classification
## Submit to ACM KDD 2022
## Preparation

### Dataset Preparation
In our experiments, we use the UCR time series achive to evaluate our model, which it includes 128 vary datasets and its details are shown at http://www.timeseriesclassification.com . 
### Install require tools
```
Details are shown in Require.txt.
```
### Modify the adress of UCR Archive
```
        Open the configuration file: concifg.py
        Amend: data_files = '/home/josen/deep learning/Caps Time Series/dataset/UCRArchive_2018'
        python3 prepare.py  
```
## Training
```
      python3 train.py           ##our proposed method
      Results: result.txt
      python3 trainFedKD.py      ### FedKD
```
