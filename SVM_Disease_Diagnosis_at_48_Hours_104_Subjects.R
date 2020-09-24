# Libraries 
library(class) # For various classification functions
library(caret) # For various machine learning functions
library(dplyr) # For efficient access of dataframes 
library(ggplot2) # For Plotting

# Read the labeled gene expression data
Data_104_Subjects_0_48_hr <- read.csv("Data_104_Subjects_0_48_Hours.csv", 
                                      header = TRUE, sep = ",")

# Display the data
Data_104_Subjects_0_48_hr[c(1:7),c(1:7)] # show first 7 rows

# Display the dimensions (rows columns)
(dim(Data_104_Subjects_0_48_hr))

## Dividing data set into train (78%) and test (22%) using createDataPartition function of caret package
set.seed(1234)
index_Train <- createDataPartition(y = Data_104_Subjects_0_48_hr$Label, p = 0.78, list = 
                                      FALSE)
g_train_data <- Data_104_Subjects_0_48_hr[index_Train, ]
g_test_data <- Data_104_Subjects_0_48_hr[-index_Train, ]

# Display the dimensions (rows columns)
(dim(g_train_data))
(dim(g_test_data))

# Converting class labels into categorical variable
g_train_data[["Label"]] = factor(g_train_data[["Label"]])


#########################################################################################
################################### Linear SVM  #########################################
#########################################################################################
set.seed(1234)
cross_validation_10_fold <- trainControl(method = "repeatedcv", # apply repeated CV
                                         number = 10, # 10 fold cv 
                                         repeats = 3)  # 3 repititions of CV
                                         
set.seed(1234)

# Assigning values to the parameter C
grid <- expand.grid(C = c(2^-5, 2^-3, 2^-1, 1, 2^1, 3, 5, 2^3, 2^5,
                          2^7, 2^9, 2^11, 2^13, 2^15))

set.seed(1234)
(svm_train_linear_tuned <- train(Label~., # Class labels of training data
                                data = g_train_data,  # Training Data
                                method = "svmLinear", # Train using linear kernel
                                tuneGrid = grid, # Passing grid for tuning
                                trControl = cross_validation_10_fold)) # Cross validation setting

# Plot tuned linear svm (Trained Model)
plot(svm_train_linear_tuned)

# Save the trained model for future use
save(svm_train_linear_tuned, file = "svm_train_linear_tuned_104_sub_0_48_hr_22.rda")

# Important Genes
(varImp(svm_train_linear_tuned))

# Save the results of trained model in a file
svm_train_linear_tuned_104_sub_0_48_hr_22_VarIMP_1 <- varImp(svm_train_linear_tuned)
write.table(svm_train_linear_tuned_104_sub_0_48_hr_22_VarIMP_1$importance, file = "svm_train_linear_tuned_104_sub_0_48_hr_22_VarIMP_1.tsv", sep = "\t")

# Display the trained model
print(svm_train_linear_tuned)

# Training set prediction
Train_Predict_svm_train_linear_tuned <- predict(svm_train_linear_tuned)
write.table(Train_Predict_svm_train_linear_tuned, file = "Train_Predict_svm_train_linear_tuned.tsv", sep = "\t")

# Predicting Test Set 
# Passing test data without labels (without fist column which contains labels)
(testPrediction_tuned_L_SVM <- predict(svm_train_linear_tuned, newdata = g_test_data[,2:12024]))

# Test data set
(g_test_data$Label)

# Display confusion matrix
(confusionMatrix(testPrediction_tuned_L_SVM, g_test_data$Label))


#########################################################################################
################################# Non-Linear SVM  #######################################
#########################################################################################

# Assigning values to the parameters sigma and C
gridRBF <- expand.grid(sigma = c(2^-25, 2^-23, 2^-21, 2^-19, 2^-17,2^-15, 
                                 2^-13, 2^-11, 2^-9, 2^-7, 2^-5, 2^-3, 
                                 2^-1, 1, 2^1, 3, 5, 2^3),
                           C = c(2^-5, 2^-3, 2^-1, 1, 2^1, 3, 5, 2^3, 2^5,
                                 2^7, 2^9, 2^11, 2^13, 2^15))

set.seed(1234)
(svm_train_RBF_tuned <- train(Label~., # Class labels of training data
                                data = g_train_data,  # Training Data
                                method = "svmRadial", # Use RBF Kernel
                                #preProcess = "range", # range between 0 to 1
                                tuneGrid = gridRBF, # Passing grid parameter 
                                trControl = cross_validation_10_fold)) # Passing cross validation values

# Plot tuned linear svm (Trained Model)
plot(svm_train_RBF_tuned)


# Save the trained model for future use
save(svm_train_RBF_tuned, file = "svm_train_RBF_tuned_104_sub_0_48_hr_22.rda")

# Important Genes
(varImp(svm_train_RBF_tuned))

# Save the results of trained model in a file
svm_train_RBF_tuned_104_sub_0_48_hr_22_VarIMP_1 <- varImp(svm_train_RBF_tuned)
write.table(svm_train_RBF_tuned_104_sub_0_48_hr_22_VarIMP_1$importance, file = "svm_train_RBF_tuned_104_sub_0_48_hr_22_VarIMP_1.tsv", sep = "\t")

# Display the trained model
print(svm_train_RBF_tuned)

# Training set prediction
Train_Predict_svm_train_RBF_tuned <- predict(svm_train_RBF_tuned)
write.table(Train_Predict_svm_train_RBF_tuned, file = "Train_Predict_svm_train_RBF_tuned.tsv", sep = "\t")

# Predicting Test Set 
# Passing test data without labels (without fist column which contains labels)
(testPrediction_tuned_RBF_SVM <- predict(svm_train_RBF_tuned, newdata = g_test_data[,2:12024]))

# Test data set
(g_test_data$Label)

# Display confusion matrix
(confusionMatrix(testPrediction_tuned_RBF_SVM, g_test_data$Label))
