Machine Learning Project
========================================================
We preprocessed the data by removing NAs and unuseful predictors. This data is partitioned into training set and cross validation set. Around 20% of training set is randomly selected in order to reduce time taken to fit the model. 


```r
# load data
setwd("G:/DataScience//Coursera//data")
trainRawData <- read.csv("pml-training.csv", na.strings = c("NA", ""))
testingRawData <- read.csv("pml-testing.csv", na.strings = c("NA", ""))
# discard NAs
NAs <- apply(trainRawData, 2, function(x) {
    sum(is.na(x))
})
validData <- trainRawData[, which(NAs == 0)]

testNAs <- apply(testingRawData, 2, function(x) {
    sum(is.na(x))
})
validTestData <- testingRawData[, which(testNAs == 0)]

library(e1071)
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
trainIndex <- createDataPartition(y = validData$classe, p = 0.75, list = FALSE)
trainData <- validData[trainIndex, ]
crossValidateData <- validData[-trainIndex, ]

# discards unuseful predictors
removeIndex <- grep("timestamp|X|user_name|new_window", names(trainData))
trainData <- trainData[, -removeIndex]

# select random observations
trainInds <- sample(nrow(trainData), 3000)
trainD <- trainData[trainInds, ]
```

Random Forest model is fitted and evaluated using cross validation data set.

```r
modFit <- train(classe ~ ., data = trainD, method = "rf", prox = TRUE)
modFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 2.23%
## Confusion matrix:
##     A   B   C   D   E class.error
## A 870   0   0   0   0     0.00000
## B  17 560  14   0   0     0.05245
## C   0  10 534   1   0     0.02018
## D   1   1  13 451   2     0.03632
## E   0   2   3   3 518     0.01521
```

```r
predictions <- predict(modFit, newdata = crossValidateData)
confusionMatrix(predictions, crossValidateData$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395   30    0    0    0
##          B    0  907    9    0    4
##          C    0   12  845   13    2
##          D    0    0    1  790    8
##          E    0    0    0    1  887
## 
## Overall Statistics
##                                        
##                Accuracy : 0.984        
##                  95% CI : (0.98, 0.987)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.979        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.956    0.988    0.983    0.984
## Specificity             0.991    0.997    0.993    0.998    1.000
## Pos Pred Value          0.979    0.986    0.969    0.989    0.999
## Neg Pred Value          1.000    0.989    0.998    0.997    0.997
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.185    0.172    0.161    0.181
## Detection Prevalence    0.291    0.188    0.178    0.163    0.181
## Balanced Accuracy       0.996    0.976    0.991    0.990    0.992
```

Using confusion matrix, we find that accuracy is 97.59%. Other statistics also look good to finalize the fitted model. Prediction algorithm is applied on the 20 test cases

```r
newPredictions <- predict(modFit, newdata = validTestData)
```


Predicted values are

```r
newPredictions
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


