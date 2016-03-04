#Course Project - Practical ML Coursera
#Credits : https://rpubs.com/flyingdisc/practical-machine-learning-xgboost

#Zarmeen Nasim

library(dplyr)
library(xgboost)
library(caret)
library(Ckmeans.1d.dp)
library(corrplot)
setwd('D:\\DataScience\\Practical Machine learning')
seed <- 2131

train <- read.csv("pml-training.csv",header = TRUE,sep=',',stringsAsFactors = FALSE)
test <- read.csv("pml-testing.csv",header = TRUE,sep=',',stringsAsFactors = FALSE)
#Dimensions of data
dim(train)
dim(test)
#Structure of data
str(train)
str(test)

# target outcome
outcome = as.factor(train[, "classe"])
levels(outcome)

# convert character levels to numeric
# convert character levels to numeric
num.class = length(levels(outcome))
levels(outcome) = 1:num.class
head(outcome)

# remove outcome from train
# filter columns on: belt, forearm, arm, dumbell
filter = grepl("belt|arm|dumbell", names(train))
train = train[, filter]
test = test[, filter]
train$classe <- NULL

dim(train)
dim(test)

########################################################
#Low variance filtering - default variance cutof : 0.05
########################################################
nsv <- nearZeroVar(train,saveMetrics = TRUE,names = TRUE)
columnsToFilter <- nsv[nsv$zeroVar == TRUE,]
lowvarianceFilter <- row.names(columnsToFilter)
print(lowvarianceFilter) #empty:every column has a variance above threshold

############################
# Missing Value
############################
train[is.na(train)] <- -1
test[is.na(test)] <- -1

# convert data to matrix
train.matrix = as.matrix(train)
mode(train.matrix) = "numeric"
test.matrix = as.matrix(test)
mode(test.matrix) = "numeric"
# convert outcome from factor to numeric matrix 
#   xgboost takes multi-labels in [0, numOfClass)
y = as.matrix(as.integer(outcome)-1)

# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)
# set random seed, for reproducibility 
set.seed(1234)
# k-fold cross validation, with timing
nround.cv = 150
bst.cv <- xgb.cv(param=param, data=train.matrix, label=y,nfold=4, nrounds=nround.cv, prediction=TRUE, verbose=TRUE,missing = NaN)

tail(bst.cv$dt)
# index of minimum merror
min.merror.idx = which.min(bst.cv$dt[, test.merror.mean]) 
min.merror.idx 
# minimum merror
bst.cv$dt[min.merror.idx,]

# get CV's prediction decoding
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")
# confusion matrix
confusionMatrix(factor(y+1), factor(pred.cv))


# real model fit training, with full data
system.time( bst <- xgboost(param=param, data=train.matrix, label=y, 
                            nrounds=min.merror.idx, verbose=1,missing = NaN) )
# xgboost predict test data using the trained model
pred <- predict(bst, test.matrix,missing = NaN)  
head(pred, 10) 

# decode prediction
pred = matrix(pred, nrow=num.class, ncol=length(pred)/num.class)
pred = t(pred)
pred = max.col(pred, "last")
pred.char = toupper(letters[pred])

#Feature Importance
# get the trained model
model = xgb.dump(bst, with.stats=TRUE)
# get the feature real names
names = dimnames(train.matrix)[[2]]
# compute feature importance matrix
importance_matrix = xgb.importance(names, model=bst)

# plot
gp = xgb.plot.importance(importance_matrix)
print(gp) 
write.csv(pred.char, "answers.csv",quote = FALSE)


