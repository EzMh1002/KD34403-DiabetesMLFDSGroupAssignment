# KD34403 - Diabetes MLFDS Group 13 Assignment 
## Machine Learning Fundamental Data Science Group Assignment
                                                                                  
### --------Group Member---------                                                      
1. Lim Ming Herng                                                                 
2. Lim Chern Hang                                                                 
3. Brandon Evanoel Rayner                                                         
4. Andy Siah Chun Xing                                                         
5. Isma Daniel Chung Soon Fan                                                     
__________________________________________________________________________________
# TO COMPILE AND RUN ALL THE PYTHON FROM MILESTONE 1-5, EXECUTE DiabetesFullRun.py 
Dataset used: Pima Indians Diabetes Database; 768 instances; 8 numeric attributes
#### Metadata shows no missing values
---------------------------------------------
## Milestone 1: Data Pipeline - Lim Ming Herng
**Data Cleaning**
1. Replacing 'hidden' missing zeros with NaN for attributes such as 'plas', 'pres', 'skin', 'insu', 'mass'.
2. Used Median imputation to fill gaps in case of missing values is found in dataset.
3. Performed label encoding for clarity.
4. Generate feature correlation heatmap to find relation between the attributes to check for possible dimensionality reduction.
5. Process an .arff format file and output a cleaned dataset in .csv format.

## Milestone 2: Architecture Logic - Andy Siah Chun Xing
1. Identify Problem and Task type
2. Chosen Random Forest (RF) for its suitability on tabular data, the scale of datasets, non-linear decision boundary of the task and capability of outputting feature importance natively.
3. Describe the architecture and pipeline of the task, and outlining the key hyperparameters that can be used.

## Milestone 3: Training - Brandon Evanoel Rayner
1. Load prepared cleaned dataset
2. Uses stratified splits (70% Training, 15% Validation, 15% Testing)
3. Apply standard scaling
4. Train RF with multiple configuration (n_estimator = 10, 25, 50, 75, 100, 150, 200)
5. The trained RF model is evaluated using multiple metrics 
(eg: Training accuracy, Training recall, Validation accuracy, Validation recall, Precision, Recall, F1-Score, ROC AUC)
6. Selects model with best Recall.
   
## Milestone 4: Optimization - Isma Daniel Chung Soon Fan
1. Using different stratified splits ratio (60% Training, 20% Validation, 20% Testing).
2. Setting up different RF parameter configuration combinations (n_estimator, max_depth, min_samples_split, min_samples_leaf, max_features).
3. Setting up different sampling and imbalance handling technique.
4. Optimizing threshold by testing between 0.3 to 0.65 to find the one with best F1-Score
5. Using for loop to run different RF combinations of parameters, sampling and imbalance handling technique, and different threshold to find the model that produces best results
6. Utilizing 5-Fold Stratified Cross Validation to compare it with manual splits, using the aforementioned loop to find the model with combination of parameters that has best results.
7. Produce final Result that summarise and shows the overall best model along with the parameters used.
7. Generate Visualization to help compare the metrics of optimized best model with the initial training from Milestone 3.


## Milestone 5: Final Evaluation - Lim Chern Hang
1. Evaluating the final selected model with different standard metrics
2. Comparison between final model and other models used
3. Visualization of the best model
