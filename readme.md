
## code package and its contents
1. dataset folder - including all the selected datasets

2. src folder - including the 
    training script, 
    utils
    different models here

3. trained model weights
    model_name_time_trained_epochs




# Data

1. Data should be place inside `data` folder

Example:  
```
./    
  source/    
  data/   
    BUSI/   
    PASCAL/   
  results/    
```

```
curl -L -o medical-image-segmentation-datasets-hi-gmisnet.zip\
  https://www.kaggle.com/api/v1/datasets/download/tushartalukder/medical-image-segmentation-datasets-hi-gmisnet
```  
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar  

tar -xvf VOCtrainval_11-May-2012.tar
``` 
ls

# Reference
- Unet: https://arxiv.org/abs/1505.04597
- Unet++: https://arxiv.org/abs/1807.10165
