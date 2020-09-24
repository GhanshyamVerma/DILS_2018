
# Libraries 
library(class) # For various classification functions
library(caret) # For various machine learning functions
library(randomForest) # For Random Forest

# Read the labeled gene expression data
Data_64_Subjects_0_Onset <- read.csv("Data_64_Subjects_0_Onset_Time.csv", 
                                      header = TRUE, sep = ",")


# Display the data
Data_64_Subjects_0_Onset[c(1:7),c(1:7)] # show first 7 rows

# Display the dimensions (rows columns)
(dim(Data_64_Subjects_0_Onset))
 
## Dividing data set into train (78%) and test (22%) using createDataPartition function of caret package
set.seed(1234)
index_Train <- createDataPartition(y = Data_64_Subjects_0_Onset$Label, 
                                   p = 0.78, list = FALSE, times = 1)
g_train_data <- Data_64_Subjects_0_Onset[index_Train, ]
g_test_data <- Data_64_Subjects_0_Onset[-index_Train, ]

# Display the dimensions (rows columns)
(dim(g_train_data))
(dim(g_test_data))

# Converting class labels into categorical variable
g_train_data[["Label"]] = factor(g_train_data[["Label"]])
g_test_data[["Label"]] = factor(g_test_data[["Label"]])


# Enable Parallel Processing
library(doSNOW)
library(doParallel)
cl <- makeCluster(detectCores())
registerDoSNOW(cl)
pt<-proc.time()

set.seed(1234)

## Training the model
# Defining trainControl parameters
cross_validation_10_fold <- trainControl(method = "repeatedcv", # apply repeated CV
                                         number = 10, # 10 fold cv 
                                         repeats = 3, # 3 repititions of CV
                                         search="grid")  
# defining evaluation metric
metric <- "Accuracy"


# ntree: parameter that allows number of trees to grow
# The mtry parameter setting: Number of variables selected as candidates at each split.
# Square root of number of features
mtry <- floor(sqrt(ncol(g_train_data)))

# Passing parameter into tunegrid
tunegrid <- expand.grid(.mtry=mtry)
set.seed(1234)

# training the model
RF_train1 <- train(Label~.,  # Class labels of training data
                   data = g_train_data, # Training Data
                   method = "rf", # Train using Random Forest
                   metric= metric, # Passing "Accuracy" as evaluation matric 
                   tuneGrid=tunegrid, # Passing tunegrid for tuning parameters
                   # Number of trees
                   ntree = 10001,
                   # Passing training control parameters
                   trControl = cross_validation_10_fold)

# Save the trained model for future use
save(RF_train1, file = "RF_train1_64_sub_0_Onset_Time.rda")

# Important Genes
(varImp(RF_train1))

# Save the results of trained model in a file
RF_train_64_sub_0_Onset_VarIMP_1 <- varImp(RF_train1)
write.table(RF_train_64_sub_0_Onset_VarIMP_1$importance, file = "RF_train_64_sub_0_Onset_VarIMP_1.tsv", sep = "\t")

# Display the trained model
print(RF_train1)

# Training set prediction
Train_Predict1 <- predict(RF_train1)
write.table(Train_Predict1, file = "Train_Predict1.tsv", sep = "\t")

# Predicting Test Set 
# Passing test data without labels (without fist column which contains labels)
(testPrediction <- predict(RF_train1, newdata = g_test_data[,2:12024]))

# Predict the class probability
Class_probability1 <- predict(RF_train1, type = "prob")
write.table(Class_probability1, file = "Class_probability1.tsv", sep = "\t")

# Test data set
(g_test_data$Label)

# Display confusion matrix
(confusionMatrix(testPrediction, g_test_data$Label))
