# Libraries 
library(class) # For various classification functions
library(caret) # For various machine learning functions
library(dplyr) # For efficient access of dataframes 
library(ggplot2) # For Plotting

# Read the labeled gene expression data
Data_64_Subjects_0_48_Hours <- read.csv("Data_64_Subjects_0_48_Hours.csv", 
                                      header = TRUE, sep = ",")

# Showing inicial 15 rows and 15 column
Data_64_Subjects_0_48_Hours[c(1:7),c(1:7)]   

# Print number of rows
nrow(Data_64_Subjects_0_48_Hours)

# Dividing data set into train (78%) and test (22%) using createDataPartition function of caret package
set.seed(1234)
index_Train <- createDataPartition(y = Data_64_Subjects_0_48_Hours$Label, p = 0.78, list = 
                                      FALSE)
g_train_data <- Data_64_Subjects_0_48_Hours[index_Train, ]
g_test_data <- Data_64_Subjects_0_48_Hours[-index_Train, ]

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

# Predicting Test Set 
# Passing test data without labels (without fist column which contains labels)
(testPrediction_tuned_RBF_SVM <- predict(svm_train_RBF_tuned, newdata = g_test_data[,2:12024]))

# Test data set
(g_test_data$Label)

# Display confusion matrix
(confusionMatrix(testPrediction_tuned_RBF_SVM, g_test_data$Label))
