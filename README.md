# KD34403 - Diabetes MLFDS Group 13 Assignment 
## Machine Learning Fundamental Data Science Group Assignment
                                                                                  
### --------Group Member---------                                                      
1. Lim Ming Herng                                                                 
2. Lim Chern Hang                                                                 
3. Brandon Evanoel Rayner                                                         
4. Andy Siah Chun Xing                                                            
5. Isma Daniel Chung Soon Fan                                                     
__________________________________________________________________________________

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
   
