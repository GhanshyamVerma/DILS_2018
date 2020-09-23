#Libraries 
library(class)
library(caret)
library(dplyr) # For efficient access of dataframes 
library(pROC) # For Plotting the ROC curves
library(ggplot2)

# Read the labeled gene expression data
All_104_Subjects_0_48_Hr <- read.csv("Data_104_Subjects_0_48_Hours.csv", 
                                      header = TRUE, sep = ",")


# Display the data
All_104_Subjects_0_48_Hr[c(1:7),c(1:7)] # show first 7 rows


# Display the dimensions (rows columns)
(dim(All_104_Subjects_0_48_Hr))


## Dividing data set into train (78%) and test (22%) using createDataPartition function of caret package
set.seed(1234)
index_Train <- createDataPartition(y = All_104_Subjects_0_48_Hr$Label, p = 0.78, list = 
                                      FALSE)
g_train_data <- All_104_Subjects_0_48_Hr[index_Train, ]
g_test_data <- All_104_Subjects_0_48_Hr[-index_Train, ]


(dim(g_train_data))
(dim(g_test_data))

# Converting class labels into categorical variable
g_train_data[["Label"]] = factor(g_train_data[["Label"]])


# Enable Parallel Processing
library(doSNOW)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoSNOW(cl)
pt<-proc.time()
set.seed(1234)

#10-fold cross validation, repeating 3 times
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

# Stop Parallel Processing
proc.time()-pt
stopCluster(cl)

# plot the trained KNN classifier
plot(KNN_train, main = "Accuracy of classifier at different values of k")

# Predicting Test Set 
# Passing test data without labels (without fist column which contains labels)
(testPrediction <- predict(KNN_train, newdata = g_test_data[,2:12024]))

# Test data set
(g_test_data$Label)

# Display confusion matrix
(confusionMatrix(testPrediction, g_test_data$Label))
