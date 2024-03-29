# cebd1160 final project: Breast cancer dataset

| Name | Date |
|:-------|:---------------|
|Paola Rimac | Jun 15, 2019 |

-----

### Resources

- Python script for your analysis: `classification_analysis.py`
- Results figure/saved file: `figures/`
- Dockerfile for your experiment: `Dockerfile`
- runtime-instructions in a file named `RUNME.md`

-----

## Research Question

From the 30 atributes of breast cancer dataset, which attributes have more correlation and which not and how accuracy the logistic regression model can predict the diagnosis.

### Abstract

Derived from UCI Machine Learning Repository, a brest cancer dataset is freely available in python's sklearn library and it represents the features computed 
from digitized image of a fine needle aspirate (FNA) of a breast mass.
Using these dataset, the understanding of this characteristics and its relationships could impact in the prediction of malignant or bening cancer. 
To achieve this I have used Logistic Regression Model to fit a function that can predict the discrete class of new input. 

### Introduction

###### Identify the problem

Breast cancer is the most common cancer among canadian women and it is the second leading cause of death from cancer according to Canadian Cancer Society. 
Some statistics from 2017 are: 
- 26,300 women were diagnosed with breast cancer and that was 25% of all new cancer in woman in 2017. 
- 5,000 women died from breast cancer which represents 13% of all cancer deaths in women. 
- An average of 72 canadian women were diagnosed from breast cancer every day.
- An average 14 canadian woman died from breast cancer every day.

The rate of incidence of breast cancer in Canada rose in the early of 90's but decreased in the early of 2000's and one of the reason was the use of mamography. 
In addtion the death rate has been declining since the mid of 80's and this is the reflect of the improvement of screening and imaging.(*) 

###### Description of the dataset
The dataset is from the "Breast Cancer Wisconsin (Diagnostic) Database" freely available in python's sklearn library (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

    Number of Samples: 569
    Number of Features: 30 numeric attributes/features
    Number of Classes: 2

Ten real-valued features are computed for each cell nucleus. 
The mean, standard error and 'worst' or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. 
For instance, the radius measurements are for the 'mean radius', 'standard error of the radius', and 'worst radius'. 

Describe feature statistics

           mean radius  mean texture   mean perimeter  mean area     mean smoothness  mean compactness
    count   569.000000    569.000000      569.000000   569.000000       569.000000        569.000000
    mean     14.127292     19.289649       91.969033   654.889104         0.096360          0.104341
    std       3.524049      4.301036       24.298981   351.914129         0.014064          0.052813
    min       6.981000      9.710000       43.790000   143.500000         0.052630          0.019380
    25%      11.700000     16.170000       75.170000   420.300000         0.086370          0.064920
    50%      13.370000     18.840000       86.240000   551.100000         0.095870          0.092630
    75%      15.780000     21.800000      104.100000   782.700000         0.105300          0.130400
    max      28.110000     39.280000      188.500000  2501.000000         0.163400          0.345400


###### Visualization of the data

First let's take a look of the proportion of classes 0 for Bening and 1 for Malignant in the following chart and graph.
   
    Number of cells class 0 - Benign:       357
    Number of cells class 1 - Malignant :   212
    % of cells class 0 - Benign:            62.74 %
    % of cells class 1 - Malignant:         37.26 %
![Countplot](./figures/countplot.png)
                       
We can see that from the 569 observations, 357 or 62.7% are labeled as Benign and 212 or 37.2% are labeled as Malignant.
   
Now as the dataset has 30 features, a good way to check correlations between all the columns is by visualizing the correlation matrix as a heatmap
![Heatmap](./figures/heatmap-all.png) 

### Methods

As the breast cancer dataset has 30 attributes, the first question that appears is: are all of them necessary to fit in a model? How could we prevent the overfiting?
To answer this question, we can apply the PCA that is essentially a method to reduces the dimensions to a new group of variables called principal components. 
But before to use the PCA, we need to scale the dataset in order to have each feature a unit variance. Then the Logistic Regression model will be fitted to obtain the function.

- Scale the dataset with StandardScaler
- PCA reduce feature
- Logistic Regression Model
- Accuracy of the model

`Scale the dataset`

        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
 
`PCA - reduce feature`

        pca = PCA(n_components=3)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        
        print('shape of X_pca', X_pca.shape)
        expl = pca.explained_variance_ratio_
        print(expl)
        print('sum of 2 components: ', sum(expl[0:2]))
        print('sum of 3 components: ', sum(expl[0:3]))
        
        shape of X_pca (569, 3) 
        [0.44272026 0.18971182 0.09393163]
        sum of 2 components:  0.6324320765155929
        sum of 3 components:  0.7263637090898976

![PCA-plot](./figures/pca-plot-n-4.png)
![PCA-scatter](./figures/pca-scatter-n-3.png)

`Logistic Regression Model`

First we need to split the data and we use the ratio of 0.35. In our case the train subdataset has 369 records and the Test subdataset has 200 records.
        
        X_train.shape : (369, 30), y_train.shape : (369,)
        X_test.shape : (200, 30), y_test.shape : (200,)

Then the train dataset is fitted to the Logistic Regression Model which presents the following information:
        
        Intercept per class: [0.24865834]
             
        Coeficients per class: [[-0.29977415 -0.52960362 -0.32794625 -0.43076019  0.04923918  0.35786804
               -0.85080214 -0.75914363 -0.22669181  0.24262574 -1.3151279   0.30292613
               -0.80117423 -0.89942787  0.37723735  0.9021331   0.20562958 -0.20817417
                0.20599237  0.60046117 -0.83368415 -0.95678081 -0.72914745 -0.86124086
               -0.88889321 -0.05175214 -0.75131017 -0.89889266 -0.4511914  -0.47978468]]

### Results
After applying the model to Test dataset, we can evaluate the accuracy of our predictions by checking out the Classification report and the confusion matrix.
       
       Classification report
       
              precision    recall  f1-score   support

           0       0.96      0.95      0.95        76
           1       0.97      0.98      0.97       124

    accuracy                           0.96       200
    macro avg      0.96      0.96      0.96       200
    weighted avg   0.96      0.96      0.96       200


        Confusion Matrix: 
            [[ 72   4]
            [  3 121]] 

        True Negative: 72
        False Positive: 4
        False Negative: 3
        True Positive: 121
    
        Correct Predictions 96.5 %
        
        Overal f1-score : 0.9627649671533818

Our model have accurately labeled 96.5% of the test data. We could try to increase the accuracy even higher by using a different algorithm other than the 
logistic regression for example Super Vector Machines, Nave Bayes, Decision Trees, Neural Networds among others; or try our model with different set of variables. 

### Discussion

The Logistic Regression Model is one of the simple algorithm to predict a categorical variable. And in this case, it gives us the accuracy of 96.5% that is very important.
As part of the academic studies of how to apply, understand and interpret a machine learning algorithm, the logistic regression model is a good way to start.
Nowadays there are many machine learning algorithms, each one with its pros and cons. I would like to modify some parameters of PCA, the size of the training dataset and or 
other parameter of the model in order to learn more how is the impact in the results and the accuracy, but that would be in a near future. 


### References
https://scikit-learn.org/stable/datasets/index.html#breast-cancer-wisconsin-diagnostic-dataset
https://www.cancer.ca/en/cancer-information/cancer-type/breast/statistics/?region=on
https://towardsdatascience.com/dive-into-pca-principal-component-analysis-with-python-43ded13ead21
https://rstudio-pubs-static.s3.amazonaws.com/344010_1f4d6691092d4544bfbddb092e7223d2.html
https://www.kaggle.com/leemun1/predicting-breast-cancer-logistic-regression