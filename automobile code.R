#set working directory
setwd("C:/Users/Lenovo/Desktop/phd ml")

## Loading the required libraries 
library(ipred)
library(ROSE)
library(ada)
library(rpart.plot)
library(rpart)
library(randomForest)
library(C50)
library(factoextra)
library(xgboost)
library(glmnet)
library(mice)
library(dplyr)
library(ROCR)
library(DMwR)
library(car)
library(MASS)
library(vegan)
library(dummies)
library(infotheo)
library(caTools)
library(caret)
library(e1071)
library(corrplot)
library(dplyr)
library(purrr)


# clear working space
rm(list=ls(all=T))
# read train data
machine_train <- read.csv("Train.csv")
machine_test <- read.csv("Test.csv")

#see the structure and summary of data
# * Observing the structure will reveal what are the data types of attributes
# * It can be helpful to understand any data type changes are required

str(machine_train)
summary(machine_train)

#Drop the duplicate records.
machine_train <- machine_train[!duplicated(machine_train),] # no duplicate values
machine_test <- machine_test[!duplicated(machine_test),]

#check missing values
sum(is.na(machine_train))
dim(machine_train)
# percentage of missing values
sum(is.na(machine_train))/(3156*22)

#missing values by each attribute
colSums(is.na(machine_train))
dims <-dim(machine_train)
dims[1]
# percentage wise
count <-apply(machine_train, 2, function(x) sum(is.na(x))/dims[1])
count

#convert into appropriate datatypes

ggplot(machine_train,aes(x = Number.of.Cylinders, fill= y)) +
  geom_bar(width = 0.5) +
  xlab("Number.of.Cylinders") + ylab("y")  +
  ggtitle("Number of cylinders")+
  theme(axis.text.x=element_text(angle=45,size=12),
        text=element_text(size=14))

# Three levels are there we need to convert this into factor variable
machine_train$Number.of.Cylinders <- as.factor(machine_train$Number.of.Cylinders)
machine_test$Number.of.Cylinders <- as.factor(machine_test$Number.of.Cylinders)

####### add additional information to actual train data

Train_AdditionalData <- read.csv("Train_AdditionalData.csv")
Test_AdditionalData <- read.csv("Test_AdditionalData.csv")

#Create 2 new columns TestA1 and TestB1
Train_AdditionalData$TestA1 <- 1
Train_AdditionalData$TestB1  <- 1
Test_AdditionalData$TestA1 <- 1
Test_AdditionalData$TestB1  <- 1

## Merge train and train additional data 
library(dplyr)
machine_train = left_join(machine_train, Train_AdditionalData[,c("TestA", "TestA1")], by = c("ID"="TestA"))
machine_train = left_join(machine_train, Train_AdditionalData[,c("TestB", "TestB1")], by = c("ID"="TestB"))
machine_test = left_join(machine_test, Test_AdditionalData[,c("TestA", "TestA1")], by = c("ID"="TestA"))
machine_test = left_join(machine_test, Test_AdditionalData[,c("TestB", "TestB1")], by = c("ID"="TestB"))

#Null values in the columns TestA1 and TestB1 are made 0.
machine_train["TestA1"][is.na(machine_train["TestA1"])] <- 0
machine_train["TestB1"][is.na(machine_train["TestB1"])] <- 0
machine_test["TestA1"][is.na(machine_test["TestA1"])] <- 0
machine_test["TestB1"][is.na(machine_test["TestB1"])] <- 0

# Merge the two columns TestA1 and TestB1. As if an engine has passed both TestA and TestB, then it may be potential indicator of final test performance
machine_train$TestAB <- NA
machine_test$TestAB <- NA
machine_train$TestAB <- ifelse(machine_train$TestA1 == 1 & machine_train$TestB1 == 1,"1","0")
machine_test$TestAB <- ifelse(machine_test$TestA1 == 1 & machine_test$TestB1 == 1,"1","0")

# Distribution of test results with target
ggplot(machine_train,aes(x = TestAB, fill= y)) +
  geom_bar(width = 0.5) +
  xlab("Tests") + ylab("y")  +
  ggtitle(" Additional Test results")+
  theme(axis.text.x=element_text(angle=45,size=12),
        text=element_text(size=14))

table(machine_train$TestAB)
table(machine_test$TestAB)

#convert test variable into factor type
machine_train$TestAB <- as.factor(machine_train$TestAB)
machine_test$TestAB <- as.factor(machine_test$TestAB)

#remove TestA1 and TestB1
machine_train$TestA1 <- NULL
machine_train$TestB1 <- NULL
machine_test$TestA1 <- NULL
machine_test$TestB1 <- NULL
Test_AdditionalData <- NULL
Train_AdditionalData <- NULL

#remove unique id columns
machine_train$ID <- NULL
machine_test$ID <- NULL

###Impute the null values with central imputation
Train <- centralImputation(machine_train)
Test <- centralImputation(machine_test)
sum(is.na(Train))

###split train data into train and validation split
set.seed(786)
train_rows<-createDataPartition(Train$y,p = 0.7,list = F)
Train1<-Train[train_rows,]
validation<-Train[-train_rows,]

################# Model Building #####################
#1.logistic model
log_reg <- glm(y ~ ., data = Train1, family = "binomial")
summary(log_reg) # model is not significant 

Step1 <- stepAIC(log_reg, direction="both")

#build another model with significant variables
log_reg1 <- glm(y ~ Lubrication + Fuel.Type + cam.arrangement + Cylinder.deactivation + 
                  displacement + Max..Torque + Peak.Power + Liner.Design. + 
                  TestAB,data = Train1, family = "binomial")
summary(log_reg1) # intercept and 3 attributes are not significant lets check for collinearity
vif(log_reg1) #no collinearity

#cutoff value for output probabilities
prod_train<-predict(log_reg1,type = "response")
pred <- prediction(prod_train, Train1$y)

perf <- performance(pred,measure="tpr", x.measure="fpr")
perf_auc <- performance(pred, measure="auc")
auc <- perf_auc@y.values[[1]]

print(auc) #92.83

cutoffs <- data.frame(cut= perf@alpha.values[[1]], fpr= perf@x.values[[1]], 
                      tpr=perf@y.values[[1]])

cutoffs <- cutoffs[order(cutoffs$tpr, decreasing=TRUE),]

plot(perf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

#Predictions
predictTrain = predict(log_reg1, type="response", newdata=Train1)

pred_class_train <- ifelse(predictTrain > 0.4, "yes", "no")
table(Train1$y,pred_class_train)

# Confusion matrix with threshold of 0.4
conf_matrix_train <- table(Train1$y, predictTrain > 0.4)
accuracy <- sum(diag(conf_matrix_train))/sum(conf_matrix_train)
print(accuracy)  #87.55

specificity <- conf_matrix_train[1, 1]/sum(conf_matrix_train[1, ])
print(specificity) #84.8
sensitivity <- conf_matrix_train[2, 2]/sum(conf_matrix_train[2, ])
print(sensitivity) #90.08

# Predictions on the test set
predictTest = predict(log_reg1, type="response", newdata=validation)
pred_class_test <- ifelse(predictTest > 0.4, "yes", "no")
table(validation$y,pred_class_test)

# Confusion matrix with threshold of 0.4
conf_matrix_test <- table(validation$y, predictTest > 0.5)
accuracy <- sum(diag(conf_matrix_test))/sum(conf_matrix_test)
print(accuracy) # 84.8 percent

specificity <- conf_matrix_test[1, 1]/sum(conf_matrix_test[1, ])
print(specificity) #84.9
sensitivity <- conf_matrix_test[2,2]/sum(conf_matrix_test[2, ])
print(sensitivity) #84.8

### To avoid overfitting K-fold cross validation for logistic 
#logistic model with cross validation

ctrl <- trainControl(method = "repeatedcv", number = 5, savePredictions = TRUE,repeats = 4)

log_reg2_cv <- train(y ~ .,  data=Train, method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 5)

pred_train = predict(log_reg2_cv, newdata=Train1)
confusionMatrix(data=pred_train, Train1$y) # acc=87.51 sensitivity = 86.73 specificity = 88.25
pred_test = predict(log_reg2_cv, newdata=validation)
confusionMatrix(data = pred_test, validation$y) # acc = 84.78 sen = 84.72 spec = 84.84

test <- read.csv("Test.csv")

log_reg2_pred <- predict(log_reg2_cv,Test)
submission <- data.frame(test$ID,log_reg2_pred)
colnames(submission)<- c("ID","y")
write.csv(submission,"submission5.csv")

## for logistic grader score is 87%  

### LASSO Regression
# 1. Let us build a simple Lasso  regression
# 2. Lets do a cross validation with Lasso  regression
###create dummies for factor varibales using a new function called "dummyVars"
###dummyVars is a simplified function directly converts into dummy variable.
#create dummies for factor variables 
library(caret)
dummies <- dummyVars(y~.,data=Train)
x.train = predict(dummies, newdata = Train1)
y.train = Train1$y
x.validation = predict(dummies, newdata = validation)
y.validation = validation$y

fit.lasso <- glmnet(x.train, y.train, family="binomial", alpha=1)
fit.lasso.cv <- cv.glmnet(x.train, y.train, type.measure = "auc", alpha=1, 
                          family="binomial",nfolds=10,parallel=TRUE)
plot(fit.lasso.cv)
coef(fit.lasso.cv,s = fit.lasso.cv$lambda.min)
pred.lasso.cv.train <- predict(fit.lasso.cv,x.train,s = fit.lasso.cv$lambda.min,type="response")
pred.lasso.cv.validation <- predict(fit.lasso.cv,x.validation,s = fit.lasso.cv$lambda.min,type="response")

library(ROCR)
pred <- prediction(pred.lasso.cv.train, Train1$y)
perf <- performance(pred, measure="tpr", x.measure="fpr")
plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.1))

pred_train <- ifelse(pred.lasso.cv.train> 0.4, "pass", "fail")
pred_validation <- ifelse(pred.lasso.cv.validation> 0.4, "pass", "fail")
pred_test <- ifelse(pred.lasso.cv.test> 0.4, "pass", "fail")

# Build Confusion matrix
confusionMatrix(pred_train,y.train) #acc = 87.6 sen = 86.73 spec = 88.42
confusionMatrix(pred_validation,y.validation) #acc = 84.88 sen = 84.72 spec = 85.04

############################ Naive Bayes ###########################
model_nb<-naiveBayes(Train1$y ~ . ,Train1)
#Response of the model
model_nb

#Predict the  Status on the validation data 
pred_T <- predict(model_nb,Train1)
pred <- predict(model_nb,validation)
table(pred)

#Confusion Matrix
library(caret)
confusionMatrix(pred, validation$y) # acc = 82, sen = 77, spec = 87
confusionMatrix(pred_T,Train1$y) # 85.11, sen = 80, specificity = 89

############################### Decision Tree #############################
#Decision tree with party
library(party)
mytree <- ctree(y~., Train1, controls=ctree_control(mincriterion=0.99, minsplit=200))
print(mytree)
plot(mytree,type="simple")
library(rpart.plot)
rpart.plot(mytree)

#Misclassification error for Train
tab<-table(predict(mytree), Train1$y)
print(tab)
1-sum(diag(tab))/sum(tab) # error 0.124
confusionMatrix(tab) # accu = 87.51 spec = 88.25 sens = 86.73

#Misclassification error for Test
test_pred <- predict(mytree,validation)
tab1<-table(test_pred, validation$y)
print(tab1)
1-sum(diag(tab1))/sum(tab1) # error 0.1501
confusionMatrix(tab1) # accu = 84.99 spec = 85.04 sens = 84.93


############################ C50 #################################
##Build classification model using C50
library(C50)

#a. Build model
DT_C50 <- C5.0(y~.,data=Train1)
summary(DT_C50)
#write(capture.output(summary(DT_C50)))
#b. Predictions
pred_Train = predict(DT_C50,newdata=Train1, type="class")
pred_Test = predict(DT_C50, newdata=validation, type="class")

#c.Error Metrics on train and test
confusionMatrix(Train1$y,pred_Train) # acc = 87.15 sen = 87.29 spec =87.02
confusionMatrix(validation$y,pred_Test) # acc = 84.57 sen = 84.06 spec = 85.04 

## Build classification model using cross validation

c50_cv <- train(y ~ .,  data=Train1, method="C5.0", family="binomial",
                     trControl = ctrl, tuneLength = 5)

pred_Train = predict(c50_cv, newdata=Train1)
pred_Test = predict(c50_cv, newdata = validation)
confusionMatrix(data=pred_Train, Train1$y) # acc=87.65 sensitivity = 86.73 specificity = 88.51
confusionMatrix(data=pred_Test, validation$y) # acc = 84.88 sen = 84.93 spec = 85.04

##conclusion: with cross validation and normal c5.0 performs same 

#### go with som ensamble methods (bagging)
################### Random Forest ##################
library(randomForest)
set.seed(222)
rf <- randomForest(y~., data=Train1,
                   ntree = 30,
                   mtry = 4,
                   importance = TRUE,
                   proximity = TRUE)
print(rf) # class.error 0.15,0.10
attributes(rf) 

# Prediction & Confusion Matrix - train data
library(caret)
p1 <- predict(rf, Train1)
confusionMatrix(p1, Train1$y) # accu = 90.7, sen = 87.57, spec = 93.77

# # Prediction & Confusion Matrix - validation data
p2 <- predict(rf, validation)
confusionMatrix(p2, validation$y) # accu = 85.52 sen = 82.31, spec = 88.52

###error metric scores are good lets check for error rate
# Error rate of Random Forest
plot(rf) # Error rate is constant after 30 trees

# Tune mtry
t <- tuneRF(Train1[,-1], Train1[,1],
            stepFactor = 0.5,
            plot = TRUE,
            ntreeTry = 30,
            trace = TRUE, 
            improve = 0.05) ## OOB error = 13.26% at mtry =4

# No. of nodes for the trees
hist(treesize(rf),
     main = "No. of Nodes for the Trees",
     col = "green") # it is in between 200-250

# Variable Importance
varImpPlot(rf,
           sort = T,
           main = "Top 10 - Variable Importance") 
importance(rf)
varUsed(rf)

#* change the parameters accordindly and run again
# even after tuning nothing much improved lets do cross validation with random forest

parameterGrid <- expand.grid(mtry=c(3,4,5))
modelRandom_cv <- train(y~., 
                     data = Train,
                     method = "rf",
                     trControl = ctrl,
                     tuneGrid=parameterGrid )

p1 <- predict(modelRandom_cv, Train1)
confusionMatrix(p1, Train1$y) # accu = 90.36, sen = 87.01, spec = 93.51

# # Prediction & Confusion Matrix - validation data
p2 <- predict(modelRandom_cv, validation)
confusionMatrix(p2, validation$y) # accu = 88.16 sen = 85.50, spec = 90.98

# quite good check it in grader tool
modelRandom_cv_pred <- predict(modelRandom_cv,Test)
submission <- data.frame(test$ID,modelRandom_cv_pred)
colnames(submission)<- c("ID","y")
write.csv(submission,"submission6.csv") ## Grader score = 87%

##conclusion: Random forest performing well comparitive with c5.0

###################k fold cross validation with cart ################
# train the model 
model_rpart<- train(y~., data=Train, trControl=ctrl, method="rpart")

# make predictions
pred_train<- predict(model_rpart,Train1)
pred_test <- predict(model_rpart,validation)

# summarize results
confusionMatrix(pred_train,Train1$y) #acc = 83.17 sen = 86.07  spec = 80.44
confusionMatrix(pred_test,validation$y) #acc = 80.87 sen = 84.06 spec = 77.87

#####################Boosting algorithoms##################
########## GBM ##########
# Load H2o library
# install.packages("h2o")
library(h2o)
# Start H2O on the local machine using all available cores and with 4 gigabytes of memory
h2o.init()

# Import a local R train data frame to the H2O cloud
train.hex <- as.h2o(x = Train1, destination_frame = "train.hex")

# Prepare the parameters for the for H2O gbm grid search
ntrees_opt <- c(5, 10, 15, 20, 30)
maxdepth_opt <- c(2, 3, 4, 5)
learnrate_opt <- c(0.01, 0.05, 0.1, 0.15 ,0.2, 0.25)
hyper_parameters <- list(ntrees = ntrees_opt, 
                         max_depth = maxdepth_opt, 
                         learn_rate = learnrate_opt)

# Build H2O GBM with grid search
grid_GBM <- h2o.grid(algorithm = "gbm", grid_id = "grid_GBM.hex",
                     hyper_params = hyper_parameters, 
                     y = "y", x = setdiff(names(train.hex), "y"),
                     training_frame = train.hex)
# Remove unused R objects
rm(ntrees_opt, maxdepth_opt, learnrate_opt, hyper_parameters)

# Get grid summary
summary(grid_GBM)

# Fetch GBM grid models
grid_GBM_models <- lapply(grid_GBM@model_ids, 
                          function(model_id) { h2o.getModel(model_id) })

# Function to find the best model with respective to AUC
find_Best_Model <- function(grid_models){
  best_model = grid_models[[1]]
  best_model_AUC = h2o.auc(best_model)
  for (i in 2:length(grid_models)) 
  {
    temp_model = grid_models[[i]]
    temp_model_AUC = h2o.auc(temp_model)
    if(best_model_AUC < temp_model_AUC)
    {
      best_model = temp_model
      best_model_AUC = temp_model_AUC
    }
  }
  return(best_model)
}

# Find the best model by calling find_Best_Model Function
best_GBM_model = find_Best_Model(grid_GBM_models)

rm(grid_GBM_models)

# Get the auc of the best GBM model
best_GBM_model_AUC = h2o.auc(best_GBM_model)

# Examine the performance of the best model
best_GBM_model

# View the specified parameters of the best model
best_GBM_model@parameters

# Important Variables.
varImp_GBM <- h2o.varimp(best_GBM_model)

# Import a local R test data frame to the H2O cloud
test.hex <- as.h2o(x = validation, destination_frame = "test.hex")

# Predict on same test data set
predict.hex = h2o.predict(best_GBM_model, 
                          newdata = test.hex[,setdiff(names(test.hex), "y")])

data_GBM = h2o.cbind(test.hex[,"y"], predict.hex)

# Copy predictions from H2O to R
pred_GBM = as.data.frame(data_GBM)

# evaluate the prediction on test data
confusionMatrix(pred_GBM$y,pred_GBM$predict) # accu = 83.93, sens = 83.70, spec = 84.15
# evaluate the prediction on train data
predict.hex = h2o.predict(best_GBM_model, 
                          newdata = train.hex[,setdiff(names(test.hex), "y")])

data_GBM = h2o.cbind(train.hex[,"y"], predict.hex)

# Copy predictions from H2O to R
pred_GBM = as.data.frame(data_GBM)
confusionMatrix(pred_GBM$y,pred_GBM$predict) # accu = 90.63 sens = 90.75 spec = 90.53

# Import a local R test data frame to the H2O cloud
Test.hex <- as.h2o(x = Test, destination_frame = "Test.hex")
# Predict on same test data set
predict.hex = h2o.predict(best_GBM_model, 
                          newdata = Test.hex[,setdiff(names(test.hex), "y")])
# Copy predictions from H2O to R
pred_GBM = as.data.frame(predict.hex)

#Shutdown H2O
h2o.shutdown(F)

submission <- data.frame(test$ID,pred_GBM$predict)
colnames(submission)<- c("ID","y")
write.csv(submission,"submission7.csv") ## Grader score = 87%

################### SVM ####################

###create dummies for factor varibales 
dummies <- dummyVars(y~.,data=Train)

x.train = predict(dummies, newdata = Train1)
y.train = Train1$y
x.validation = predict(dummies, newdata = validation)
y.validation = validation$y

# Building the model on train data
model  =  svm(x = x.train, y = y.train, type = "C-classification", kernel = "radial", cost = 10)
summary(model)

# Predict on train and test using the model
pred_train<-predict(model,x.train)
pred_test<-predict(model,x.validation)
# Build Confusion matrix
confusionMatrix(pred_train,y.train) # acc = 93.89 sen = 92.62 spec = 95.09
confusionMatrix(pred_test,y.validation) #acc = 83.09 sens = 81  spec = 85

#################### K fold cross validation for svm ###########
mod <- train(y~., data=Train, method = "svmLinear", trControl = ctrl)
head(mod$pred)

#Predictions and confusion matrix
pred_train <- predict(mod,Train1)
test_kf_svm <- predict(mod,validation)
confusionMatrix(pred_train,Train1$y) # accu = 87.06 sens = 85.98 spec = 88.07
confusionMatrix(test_kf_svm,validation$y) #acc = 84.99 sens = 84.93 spec = 85.04

###################### KSVM (different kernal) ###################
library(kernlab)
dummies <- dummyVars(y~.,data=Train)

x.train = predict(dummies, newdata = Train1)
y.train = Train1$y
x.validation = predict(dummies, newdata = validation)
y.validation = validation$y

#Build model using ksvm with "rbfdot" kernel
kern_rbf <- ksvm(x.train,y.train,
                 type='C-svc',kernel="rbfdot",kpar="automatic",
                 C=10, cross=5)
kern_rbf

# Predict on train and test using the model
pred_train2<-predict(kern_rbf,x.train)
pred_test2<-predict(kern_rbf,x.validation)

# Build Confusion matrix
confusionMatrix(pred_train2,y.train) # acc = 92.26 sen = 91.21 spec = 93.25
confusionMatrix(pred_test2,y.validation) # acc = 83.19 sen =81.66 spec = 84.63
str(train)


############################# XGBoost ################################

# cross validation structure/parameters
ControlParamteres <- trainControl(method = "cv",
                                  number = 5,
                                  savePredictions = TRUE,
                                  classProbs = TRUE
                                  )
## Model parameters tuning
parametersGrid <-  expand.grid(eta = c(0.001), colsample_bytree=c(0.5,0.6),
                               max_depth=c(3,4,5), #(default value is 6 you can increse or decrese it takes 1 to infinate )
                               nrounds=1000, #(default is 0 it ranges from 0 to infinate larger values more conservative algoritham)
                               gamma=10, #(default is 0 it ranges from 0 to infinate larger values more conservative algoritham)
                               min_child_weight=2, 
                               subsample = 0.1 #(in b/w 0 and 1 lower values prevent over fitting)
                               )

##Model Building 
modelxgboost <- train(y~., 
                      data = Train,
                      method = "xgbTree",
                      trControl = ControlParamteres,
                      tuneGrid = parametersGrid
                      )

modelxgboost

#predictions
pred_test <- predict(modelxgboost,validation)
pred_train <- predict(modelxgboost,Train1)

# Build Confusion matrix
confusionMatrix(pred_train,Train1$y) # acc = 87.06 sen = 85.98 spec = 88.07
confusionMatrix(pred_test,validation$y) #acc = 84.14 sens = 84.06  spec = 84.22

XGboost_pred <- predict(modelxgboost,Test)
submission <- data.frame(test$ID,log_reg2_pred)
colnames(submission)<- c("ID","y")
write.csv(submission,"submission_XG.csv") # grader score 87%

################# Normal XGBoost #############################
## create n-1 dummy variables
library(dummies)
library(mlr)
dummy_train_x <- as.data.frame(createDummyFeatures(Train1[,-1],method = "reference"))
dummy_train_y <- ifelse(test = (Train1$y == "pass"),yes = 1,no = 0)

train_matrix <- xgb.DMatrix(data = as.matrix(dummy_train_x), label = dummy_train_y)


dummy_validation_x <- as.data.frame(createDummyFeatures(validation[,-1],method = "reference"))
dummy_validation_y <- ifelse(test = (validation$y == "pass"),yes = 1,no = 0)

test_matrix <- xgb.DMatrix(data = as.matrix(dummy_validation_x), label = dummy_validation_y)

# Parameters
nc <- length(unique(dummy_train_y))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)
# eXtreme Gradient Boosting Model

bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 200,
                       watchlist = watchlist,
                       eta = 0.05,
                       max.depth = 3,
                       gamma = 1,
                       subsample = 1,
                       colsample_bytree = 1,
                       seed = 333)
bst_model

# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = 'blue')
lines(e$iter, e$test_mlogloss, col = 'red')

min(e$test_mlogloss) #select minimum error point
e[e$test_mlogloss == 0.37398,] #171th iteration

# Feature importance
imp <- xgb.importance(colnames(train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)

# Prediction & confusion matrix - test data
p <- predict(bst_model, newdata = test_matrix)
pred <- matrix(p, nrow = nc, ncol = length(p)/nc) %>%
  t() %>%
  data.frame() %>%
  mutate(label = dummy_validation_y, max_prob = max.col(., "last")-1)
table(Prediction = pred$max_prob, Actual = pred$label)
confusionMatrix(pred$max_prob,pred$label) #acc = 84.88 sens = 84.50 spec = 85.25


################# with Genetic algoritham ##################
x <- Train1[,2:22]
y <- Train1[,1]

ga_ctrl <- gafsControl(functions = rfGA,
                       method = "repeatedcv",
                       repeats = 5)
rf_ga <- gafs(x = x, 
            y = y,
            iters = 200,
            gafsControl = ga_ctrl,
            method = "glm")
plot(rf_ga)
