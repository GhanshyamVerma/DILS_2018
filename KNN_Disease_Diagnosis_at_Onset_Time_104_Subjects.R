# Libraries 
library(class) # For various classification functions
library(caret) # For various machine learning functions
library(dplyr) # For efficient access of dataframes 
library(ggplot2) # For plots

# Read the labeled gene expression data
All_104_Subjects_0_Onset_Avg_Onset <- read.csv("Data_104_Subjects_0_n_Onset_Avg_Onset_Time.csv", 
                                      header = TRUE, sep = ",")

# Display the data
All_104_Subjects_0_Onset_Avg_Onset[c(1:7),c(1:7)] # show first 7 rows

# Display the dimensions (rows columns)
(dim(All_104_Subjects_0_Onset_Avg_Onset))

## Dividing data set into train (78%) and test (22%) using createDataPartition function of caret package
set.seed(1234)
index_Train <- createDataPartition(y = All_104_Subjects_0_Onset_Avg_Onset$Label, p = 0.78, list = 
                                      FALSE)
g_train_data <- All_104_Subjects_0_Onset_Avg_Onset[index_Train, ]
g_test_data <- All_104_Subjects_0_Onset_Avg_Onset[-index_Train, ]

(dim(g_train_data))
(dim(g_test_data))

# Converting class labels into categorical variable
g_train_data[["Label"]] = factor(g_train_data[["Label"]])

set.seed(1234)

# 10-fold cross validation, repeating 3 times
cross_validation_10_fold <- trainControl(method = "repeatedcv", # apply repeated CV
                                         number = 10, # 10 fold cv 
                                         repeats = 3, # 3 repititions of CV
                                         search = "grid")  
                                         
set.seed(1234)
metric <- "Accuracy"
grid <- expand.grid(k = c(1:50))

# training the k nearest neighbour classifier
(KNN_train <- train(Label~., # Class labels of training data
                   data = g_train_data,  # Training Data
                   method = "knn", # Train using KNN
                   metric = metric,
                   tuneGrid = grid,
                   trControl = cross_validation_10_fold))


# plot the trained KNN classifier
plot(KNN_train, main = "Accuracy of classifier at different values of k")

# Predicting Test Set 
# Passing test data without labels (without fist column which contains labels)
(testPrediction <- predict(KNN_train, newdata = g_test_data[,2:12024]))

# Test data set
(g_test_data$Label)

# Display confusion matrix
(confusionMatrix(testPrediction, g_test_data$Label))
