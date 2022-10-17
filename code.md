Coursera Practical Machine Learning Course Project
================
Elizabeth Lundeen
2022-10-14

# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

## Data

The training data for this project are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:
<http://groupware.les.inf.puc-rio.br/har>.

## Project Aim

The goal of your project is to predict the manner in which they did the
exercise. This is the “classe” variable in the training set. You may use
any of the other variables to predict with. You should create a report
describing how you built your model, how you used cross validation, what
you think the expected out of sample error is, and why you made the
choices you did. You will also use your prediction model to predict 20
different test cases.

## Load the Data

``` r
library(caret)
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
library(rpart)
library(data.table)
library(rattle)
```

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.5.1 Copyright (c) 2006-2021 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

``` r
library(randomForest)
```

    ## randomForest 4.7-1.1

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:rattle':
    ## 
    ##     importance

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
####Load the data
training <- read.csv("C:/Users/yxj4/OneDrive - CDC/+My_Documents/CDC/1 DDT/Data Modernization/Coursera/Hopkins Data Science Specialization/8 Practical Machine Learning/3 Course Project/pml-training.csv")
#19622 obs. 160 variables

testing <- read.csv("C:/Users/yxj4/OneDrive - CDC/+My_Documents/CDC/1 DDT/Data Modernization/Coursera/Hopkins Data Science Specialization/8 Practical Machine Learning/3 Course Project/pml-testing.csv")
#20 obs. of  160 variables
```

# Step 1: Slice Training Data into a Smaller Training Set and a Validation Set (For Out-of-Sample Error Estimation)

``` r
set.seed(37501)
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
train <- training[inTrain, ]
validate <- training[-inTrain, ]
```

# Step 2: Data Preparation and Feature Engineering

## Remove features that are not useful for this classification problem, such as the participant’s name, the time stamp, and X (subject ID).

``` r
#Remove the subject's name and time stamp as these aren't useful predictors
train2 <- subset(train, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
validate2 <- subset(validate, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
```

## Deal with the problem of missing data.

Most machine learning algorithms are not able to handle missing data. I
cannot impute missing predictor variables (features) using k nearest
neighbors because the data contain a mixture of continuous and
categorical data (k nearest neighbors can only handle continuous data).
I tried using the R package missForest, which uses Random Forest
techniques to impute missing data (because it can handle mixed type data
frames). However, this did not work because missForest requires that all
columns are either factor or numeric. I converted all of the character
variables to factor, but missForest cannot handle categorical predictors
with more than 53 categories. Therefore, I ended up deciding just to
eliminate variables/features that had 95% missing data or more.

``` r
dim(train2)
```

    ## [1] 13737   155

``` r
dim(validate2)
```

    ## [1] 5885  155

``` r
fewmissing <- colSums(is.na(train2))/nrow(train2) < 0.95
train3 <- train2[,fewmissing]
validate3 <- validate2[,fewmissing]
dim(train3)
```

    ## [1] 13737    88

``` r
dim(validate3)
```

    ## [1] 5885   88

## Near Zero Variance Variables

Remove near zero variance variables because they will not contribute
much to the prediction.

``` r
nzv <- nearZeroVar(train3)
train4 <- train3[, -nzv]
validate4 <- validate3[, -nzv]
```

``` r
rm(train,train2,train3,training,validate,validate2,validate3,fewmissing,nzv,inTrain)
```

## Classe should be a factor variable.

Convert *classe* to a factor variable in both the training and
validation data sets.

``` r
train4$classe <- as.factor(train4$classe)
validate4$classe <- as.factor(validate4$classe)
```

# Step 3: Model Building

From the lecture materials, we know that random forest and boosting are
the machine learning techniques that provide the greatest accuracy for
prediction/classification problems. Therefore, I will first use a random
forest model and assess performance and accuracy in prediction. I am
instructing train to use 3-fold cross validation to select optimal
tuning parameters.

``` r
modFit <- train(classe ~ ., data=train4, method="rf", prox=TRUE, trControl = trainControl(method="cv",number=3))
print(modFit)
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    53 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 9158, 9158, 9158 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9913373  0.9890414
    ##   27    0.9951227  0.9938301
    ##   53    0.9918468  0.9896870
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 27.

``` r
#Print the final model to examine chosen tuning parameters/features
modFit$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 27
    ## 
    ##         OOB estimate of  error rate: 0.22%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 3905    1    0    0    0 0.0002560164
    ## B    4 2652    2    0    0 0.0022573363
    ## C    0    5 2391    0    0 0.0020868114
    ## D    0    0   12 2239    1 0.0057726465
    ## E    0    1    0    4 2520 0.0019801980

``` r
plot(modFit)
```

![](code_files/figure-gfm/unnamed-chunk-8-1.png)<!-- --> This output
shows that the random forest model decided to use 500 trees and tried 27
variables at each split. Accuracy was 0.9951227 (out-of-bag estimate of
error rate: 0.22%).

# Step 4: Model Evaluation and Selection

The next step is to use the fitted model to predict *classe* in the
validation data set, and produce a confusion matrix to compare the
predicted *classe* versus the actual *classe*.

``` r
predictions <- predict(modFit, newdata=validate4)
cm <- confusionMatrix(predictions,validate4$classe)
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    3    0    0    0
    ##          B    0 1135    2    0    0
    ##          C    0    1 1024    7    0
    ##          D    0    0    0  957    2
    ##          E    1    0    0    0 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9973          
    ##                  95% CI : (0.9956, 0.9984)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9966          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9965   0.9981   0.9927   0.9982
    ## Specificity            0.9993   0.9996   0.9984   0.9996   0.9998
    ## Pos Pred Value         0.9982   0.9982   0.9922   0.9979   0.9991
    ## Neg Pred Value         0.9998   0.9992   0.9996   0.9986   0.9996
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1929   0.1740   0.1626   0.1835
    ## Detection Prevalence   0.2848   0.1932   0.1754   0.1630   0.1837
    ## Balanced Accuracy      0.9993   0.9980   0.9982   0.9962   0.9990

``` r
accuracy <- postResample(predictions, validate4$classe)
accuracy
```

    ##  Accuracy     Kappa 
    ## 0.9972812 0.9965609

``` r
oose <- 1 - as.numeric(confusionMatrix(validate4$classe, predictions)$overall[1])
oose
```

    ## [1] 0.002718777

The accuracy is 99.73%, meaning the out-of-sample error is 0.27%. This
is a very high rate of accuracy, so I will most likely select random
forest as the classification algorithm of choice to predict on the test
set.

# Step 5: Determine if a Higher Degree of Accuracy is Obtained with Boosting

``` r
modFit2 <- train(classe ~ ., method="gbm",data=train4,verbose=FALSE, trControl = trainControl(method="cv",number=3))
print(modFit2)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 13737 samples
    ##    53 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 9159, 9158, 9157 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  Accuracy   Kappa    
    ##   1                   50      0.7580982  0.6930712
    ##   1                  100      0.8306038  0.7855066
    ##   1                  150      0.8696953  0.8350206
    ##   2                   50      0.8827985  0.8515757
    ##   2                  100      0.9400890  0.9241916
    ##   2                  150      0.9628013  0.9529404
    ##   3                   50      0.9316452  0.9134712
    ##   3                  100      0.9705903  0.9627892
    ##   3                  150      0.9856594  0.9818602
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## 
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were n.trees = 150, interaction.depth =
    ##  3, shrinkage = 0.1 and n.minobsinnode = 10.

``` r
#Print the final model to examine chosen tuning parameters/features
modFit2$finalModel
```

    ## A gradient boosted model with multinomial loss function.
    ## 150 iterations were performed.
    ## There were 53 predictors of which 53 had non-zero influence.

``` r
plot(modFit2)
```

![](code_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
predictions2 <- predict(modFit2, newdata=validate4)
cm2 <- confusionMatrix(predictions2,validate4$classe)
cm2
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1670    3    0    0    0
    ##          B    4 1123   12    9    2
    ##          C    0   12 1009   15    0
    ##          D    0    0    4  938   10
    ##          E    0    1    1    2 1070
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.9873        
    ##                  95% CI : (0.9841, 0.99)
    ##     No Information Rate : 0.2845        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.9839        
    ##                                         
    ##  Mcnemar's Test P-Value : NA            
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9976   0.9860   0.9834   0.9730   0.9889
    ## Specificity            0.9993   0.9943   0.9944   0.9972   0.9992
    ## Pos Pred Value         0.9982   0.9765   0.9739   0.9853   0.9963
    ## Neg Pred Value         0.9991   0.9966   0.9965   0.9947   0.9975
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2838   0.1908   0.1715   0.1594   0.1818
    ## Detection Prevalence   0.2843   0.1954   0.1760   0.1618   0.1825
    ## Balanced Accuracy      0.9984   0.9901   0.9889   0.9851   0.9940

``` r
accuracy2 <- postResample(predictions2, validate4$classe)
accuracy2
```

    ##  Accuracy     Kappa 
    ## 0.9872557 0.9838797

``` r
oose2 <- 1 - as.numeric(confusionMatrix(validate4$classe, predictions2)$overall[1])
oose2
```

    ## [1] 0.01274427

The accuracy is 98.73%, meaning the out-of-sample error is 1.27%.

# Step 6: Chose the Final Model

Boosting provided a lower accuracy and higher error rate than what was
obtained through random forest models, so I will choose random forest as
the final model.

# Step 7: Apply the Final Model to Create Predictions for the Test Set (n=20)

``` r
Results <- predict(modFit, newdata=testing)
Results
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
