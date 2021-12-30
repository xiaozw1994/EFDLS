# An Efficient Federated Distillation Learning System for Time Series Classification
## Submit to IEEE Transactions on Knowledge and Data Engineering 
## Preprint at Arxiv 

### Dataset Preparation
In our experiments, we use the UCR time series achive to evaluate our model, which it includes 128 vary datasets and its details are shown at http://www.timeseriesclassification.com . 

### Download
```
git clone https://github.com/xiaozw1994/EFDLS.git
```

### Install require tools
```
Details are shown in require.txt. #
      #without environment
        pip3 install -r require.txt
        mkdir data
        mkdir basicFCN
        mkdir FedTemp1
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
