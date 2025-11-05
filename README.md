# patternrecognition-group1
## Some generalities
> uv is needed to run this project
 
### To run a script
1. Go to the root of the repo
2. ```uv run pathtothescript```
3. Wait

## Data folder structure
```
data
|- Fashion-MNIST
|   |- All files of the datasets
|- MNIST
|   |- All files of the datasets
```

## Implementation
### SVM
#### Results
![SVM c-value vs accuracy](./SVM/SVM_RBF_Ctest.png)

### MLP
#### Results
![Training curve](./MLP/training-curve.png)
![Confusion matrix](./MLP/confusion-matrix.png)

### CNN
#### Results
![training accuracy](./CNN/exploration/runs/cnn_baseline/acc.png)
![training loss](./CNN/exploration/runs/cnn_baseline/loss.png)
![confusion matrix](./CNN/exploration/runs/cnn_baseline/confusion_matrix.png)