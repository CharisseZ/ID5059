# --------------------------------Import required Packages ------------------------------------

# Visualisation
library(ggplot2)
library(grid)

# Data handling
library(tidyverse)
library(data.table)
library(tfestimators)
library(Matrix)

# Model Creation
library(caret)
library(xgboost)
library(MLmetrics)
library(randomForest)
library(nnet)
library(mgcv)
library(e1071)


# -------------------------------------- Read in Data ------------------------------------------
source('func_ModelFunctions.R')
# Read in data
train <- as.data.table(fread('train.csv'))
test <- as.data.table(fread('test.csv'))
response <- "target"
test$target <- NA
data <- rbind(train, test)
test <- select(test, -"target")
# ------------------------------- Exploratory Data Analysis ------------------------------------

# Count missing values
perc_na <- function(data, na_val) {
  return(sum(data == na_val)/(nrow(data)*ncol(data))*100)
}

perc_na(train, -1)
perc_na(test, -1)

claimproportion <- function(data) {
  return(100*sum(data$target == 1)/nrow(data))
}

claimproportion(train)

# Using information provided, store the type of all the data
meta = data.frame(varname = names(train), type = rep(0, ncol(train)))
meta$varname = names(train)
meta[grepl("cat", names(train)),]$type <- "categorical"
meta[grepl("bin", names(train)),]$type <- "binary"
meta[which(sapply(train, class) == "integer" & meta$type == 0) ,]$type <- "ordinal"
meta[which(sapply(train, class) == "numeric" & meta$type == 0) ,]$type <- "continuous"
meta[meta$varname == "target",]$type <- "response"
meta$type <- as.factor(meta$type)
summary(meta)
# ----------------------------------------- Data Preparation ----------------------------------------
# This section is very messy
test_id <- test$id

# create features for model
data[, amount_nas := rowSums(data == -1, na.rm = T)]
data[, high_nas := ifelse(amount_nas>4,1,0)]
data[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
data[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
data[, ps_custom_bin := ifelse( ps_ind_05_cat == -1 |  ps_ind_05_cat == -2 | ps_car_07_cat == -1 | ps_car_11_cat == 41, 1, 0) ]

# Create train with feature engineering
trainFeat <- data[!is.na(data$target),]
testFeat <- data[is.na(data$target),]
trainFeat <- func_dataClean(trainFeat, response, bool_UseOneHotEncoding = T, bool_encodeVariableAsFactor = T)
testFeat <- func_dataClean(testFeat, response, bool_UseOneHotEncoding = T, bool_encodeVariableAsFactor = T)
testFeat <- testFeat[,-1]
trainFeat$target <- ifelse(trainFeat$target == "Yes", 1, 0)
set.seed(5059)
trainIndex <- createDataPartition(trainFeat$target, p = 0.7, list = FALSE, times = 1)
trainFeat <- trainFeat[trainIndex,]
validFeat <- trainFeat[-trainIndex,]

# Create a train/test with no one hot encoding
train_notonehot <- func_dataClean(train, response, bool_UseOneHotEncoding = F, bool_encodeVariableAsFactor = T)
test_notonehot <- func_dataClean(test, response, bool_UseOneHotEncoding = F, bool_encodeVariableAsFactor = T)
train_notonehot$target <- ifelse(train_notonehot$target == "Yes", 1, 0)

# Create a train/test with one hot encoding
train <- func_dataClean(train, response, bool_UseOneHotEncoding = T)
test <- func_dataClean(test, response, bool_UseOneHotEncoding = T)
train$target <- ifelse(train$target == "Yes", 1, 0)

# # Create train with a column recording NAs
# setDT(train)
# setDT(test)
# trainNAcol <- train[, amount_nas := factor(rowSums(train == -1, na.rm = T))]
# testNAcol <- test[, amount_nas := factor(rowSums(test == -1, na.rm = T))]
# trainNAcol$target <- ifelse(trainNAcol$target == "Yes", 1, 0)


# Create an undersampled set
set.seed(5059)
oneset <- train[train$target==1,][sample(nrow(train[train$target==1,]), 21694),]
zeroset <- train[train$target==0,][sample(nrow(train[train$target==0,]), 195246),]
train_undersample <- rbind(oneset, zeroset)

# Create an over and under sampled set
set.seed(5059)
oneset <- train[train$target==1,][sample(nrow(train[train$target==1,]), replace = T, 200000),]
zeroset <- train[train$target==0,][sample(nrow(train[train$target==0,]), 200000),]
train_oversample <- rbind(oneset, zeroset)

# Main training set will be train_undersample as models are able to be evaluated faster
train <- train_undersample
set.seed(5059)
trainIndex <- createDataPartition(train$target, p = 0.7, list = FALSE, times = 1)
validsplit <- train[-trainIndex,]
trainsplit <- train[trainIndex,]

set.seed(5059)
trainIndex <- createDataPartition(train_oversample$target, p = 0.7, list = FALSE, times = 1)
train_oversamplesplit <- train_oversample[trainIndex,]
valid_oversamplesplit <- train_oversample[-trainIndex,]


# ================================================ MODELS =================================================
# --------Gradient Boosting Models ----------
labels1 <- trainsplit$target 
ts_label1 <- validsplit$target

labels2 <- train_oversamplesplit$target
ts_label2 <- valid_oversamplesplit$target

labels3 <- trainFeat$target
ts_label3 <- validFeat$target

setDT(trainsplit)
setDT(validsplit)
setDT(train_oversamplesplit)
setDT(valid_oversamplesplit)
setDT(trainFeat)
setDT(validFeat)
setDT(test)
setDT(testFeat)

# Use one hot encoding to prepare data for xgboost package

# One hot encodes the factor variables
new_test <- model.matrix(~.+0,data = test)
new_testFeat <- model.matrix(~.+0, data = testFeat)
new_tr1 <- model.matrix(~.+0,data = trainsplit[,-c("target")]) 
new_ts1 <- model.matrix(~.+0,data = validsplit[,-c("target")])
new_tr2 <- model.matrix(~.+0,data = train_oversamplesplit[,-c("target")]) 
new_ts2 <- model.matrix(~.+0,data = valid_oversamplesplit[,-c("target")]) 
new_tr3 <- model.matrix(~.+0,data = trainFeat[,-c("target")]) 
new_ts3 <- model.matrix(~.+0,data = validFeat[,-c("target")])

# Stores data for xgboost package
dtest <- xgb.DMatrix(data = new_test)
dtestFeat <- xgb.DMatrix(data = new_testFeat)
dtrain1 <- xgb.DMatrix(data = new_tr1,label = labels1) 
dvalid1 <- xgb.DMatrix(data = new_ts1,label=ts_label1)
dtrain2 <- xgb.DMatrix(data = new_tr2,label = labels2) 
dvalid2 <- xgb.DMatrix(data = new_ts2,label = ts_label2)
dtrain3 <- xgb.DMatrix(data = new_tr3,label = labels3)
dvalid3 <- xgb.DMatrix(data = new_ts3,label = ts_label3)

xgb_normalizedgini <- function(preds, dtrain){
  
  actual <- getinfo(dtrain, "label")
  
  score <- NormalizedGini(preds,actual)
  
  return(list(metric = "NormalizedGini", value = score))
  
}

xgb_params <- list(colsample_bytree = 0.4, #variables per tree ,
                   
                   subsample = 0.5, #data subset per tree 
                   
                   booster = "gbtree",
                   
                   max_depth = 5, #tree levels
                   
                   eta = 0.02, #shrinkage
                   
                   eval_metric = xgb_normalizedgini,
                   
                   objective = "reg:logistic",
                   
                   seed = 5059
)

xgb_params3 <- list(booster="gbtree",
                    
                    objective="binary:logistic",
                    
                    eta = 0.02 ,
                    
                    gamma = 1,
                    
                    max_depth = 6,
                    
                    min_child_weight = 1,
                    
                    subsample = 0.8,
                    
                    colsample_bytree = 0.6,
                    
                    eval_metric = xgb_normalizedgini,
                    
                    seed = 5059
)

watchlist1 <- list(train=dtrain1, valid=dvalid1)
watchlist2 <- list(train=dtrain2, valid=dvalid2)
watchlist3 <- list(train = dtrain3, vaild=dvalid3)


set.seed(5059)
GB_Model1 <- xgb.train(params = xgb_params,
                       
                       data = dtrain1,
                       
                       print_every_n = 5,
                       
                       watchlist = watchlist1,
                       
                       nrounds = 500)

set.seed(5059)
GB_Model2 <- xgb.train(params = xgb_params,
                       
                       data = dtrain2,
                       
                       print_every_n = 5,
                       
                       watchlist = watchlist2,
                       
                       nrounds = 500)

set.seed(5059)
GB_Model3 <- xgb.train(params = xgb_params3,
                       
                       data = dtrain3,
                       
                       print_every_n = 25,
                       
                       watchlist = watchlist3,
                       
                       nrounds = 500)
mat1 <- xgb.importance(feature_names = colnames(new_tr1),model = GB_Model1)
mat2 <- xgb.importance(feature_names = colnames(new_tr2),model = GB_Model2)
mat3 <- xgb.importance(feature_names = colnames(new_tr3),model = GB_Model3)
xgb.plot.importance(importance_matrix = mat1[1:20]) 
xgb.plot.importance(importance_matrix = mat2[1:20]) 
xgb.plot.importance(importance_matrix = mat3[1:20]) 

GBpred1 <- data.frame(id = test_id, target = predict(GB_Model1, dtest)) # Gini 0.28026
GBpred2 <- data.frame(id = test_id, target = predict(GB_Model2, dtest)) # Gini 0.28226
GBpred3 <- data.frame(id = test_id, target = predict(GB_Model3, dtestFeat)) # 0.27253

GBpred1 %>% write_csv("GBpred1.csv")
GBpred2 %>% write_csv("GBpred2.csv")
GBpred3 %>% write_csv("GBpred3.csv")

# ------------------ Neural Net -------------------

my_control <- trainControl(
  method ="cv",
  number = 5,
  verboseIter = F,
  savePredictions = T,
  allowParallel = T,
  summaryFunction=twoClassSummary,
  classProbs = T,
  index = createResample(train$target, 5)
)

x_train <- as.data.frame(train)[,-1]
is.data.frame(x_train)

y_train <- as.factor(ifelse(train$target==0,"No", "Yes"))
NN_Model <- caret::train(x_train,
                         y_train,
                         trControl = my_control,
                         method = "pcaNNet",
                         metric = "ROC",
                         tuneLength = 5
)

NNpred <- predict(NN_Model, test, type = "prob")

NNpred <- pred$Yes # gini 0.25102



# ----------------------- Random Forest --------------------

# build a random forest model with ntree 8000, mtry 10 (because it creates the highest gini score)
# sampling 1% of trainig data to build models and then combine many models into one model at the end
model_list <- list()
for (i in 1:100) {
  set.seed(5059*i)
  sample_row <- sample(nrow(train), 0.01 * nrow(train))
  s_data <- train[sample_row,]
  s_model <- randomForest(y = factor(s_data[, 1]), x = s_data[, -1], ntree=80, mtry =10)  
  model_list[[i]] <- s_model
}

combine_forests <- function(forestlist) {
  rf <- forestlist[[1]]
  for (i in 2:length(forestlist)) {
    rf <- randomForest::combine(rf, forestlist[[i]])
  }
  return(rf)
}

RF_Model <- combine_forests(model_list)
RFPred <- predict(object = RF_Model, newdata = test, type = "prob")


# ------------------- GAM --------------------
originData <- fread('train.csv')
# 
set.seed(5059)
sample.data <- sample(nrow(originData), nrow(originData)/2)
traingam <- originData[sample.data, ]
testgam <- fread('test.csv')

# remove the columns which contain too many missing value 
traingam <- select(traingam, -ps_reg_03) 
traingam <- select(traingam, -ps_car_03_cat) 
traingam <- select(traingam, -ps_car_05_cat)

# remove the columns which contain too many missing value 
testgam <- select(testgam, -ps_reg_03) 
testgam <- select(testgam, -ps_car_03_cat) 
testgam <- select(testgam, -ps_car_05_cat)
# ------------------------ Convert the data into factors and logical data first!!!!!

fac.data <- traingam %>%
  mutate_at(vars(ends_with("cat")), list(factor)) %>%
  mutate_at(vars(ends_with("bin")), list(as.logical)) %>%
  mutate(target = as.factor(target))


test.fac.data <- testgam %>%
  mutate_at(vars(ends_with("cat")), list(factor)) %>%
  mutate_at(vars(ends_with("bin")), list(as.logical))





############ please make sure the "cat" data has been put into factors !!!
# the model with B-spline
set.seed(5059)
ini.mod.gam.cr.fac <- bam(target ~ s(ps_ind_01, k = 7, bs = "cr") + s(ps_ind_03, bs = "cr") + s(ps_ind_14, k = 4, bs = "cr") + 
                            s(ps_ind_15, k = 13, bs = "cr") + s(ps_reg_01, k = 1, bs = "cr") + s(ps_reg_02, k = 2, bs = "cr") + 
                            s(ps_car_11, k = 3, bs = "cr") + s(ps_car_12, k = 2, bs = "cr") + s(ps_car_13, k = 4, bs = "cr") + 
                            s(ps_car_15, k = 4, bs = "cr") + s(ps_car_14, k = 1, bs = "cr") + ps_ind_02_cat + ps_ind_04_cat +
                            ps_ind_05_cat + ps_car_01_cat + ps_car_02_cat + ps_car_04_cat + ps_car_06_cat + ps_car_07_cat + 
                            ps_car_08_cat + ps_car_09_cat + ps_car_10_cat + ps_car_11_cat + ps_ind_18_bin + ps_ind_17_bin + 
                            ps_ind_16_bin + ps_ind_13_bin + ps_ind_12_bin + ps_ind_11_bin + ps_ind_10_bin + ps_ind_09_bin + 
                            ps_ind_08_bin + ps_ind_07_bin + ps_ind_06_bin + ps_ind_04_cat + ps_reg_01 + ps_car_02_cat + 
                            ps_car_07_cat + ps_car_08_cat + ps_car_14,
                          family = binomial(link = 'logit'), data = fac.data, method = "GCV.Cp")


summary.gam(ini.mod.gam.cr.fac)

GAMpred <- data.frame(id = test_id, target = predict(ini.mod.gam.cr.fac, test.fac.data, type = "response")) # Gini 0.26485
GAMpred %>% write_csv("GAMpred")

# ----------------------- Logistic Model -----------------------
fac.data <- train_notonehot %>%
  mutate_at(vars(ends_with("cat")), list(factor)) %>%
  mutate_at(vars(ends_with("bin")), list(as.logical)) %>%
  mutate(target = as.factor(target))


test.fac.data <- test_notonehot %>%
  mutate_at(vars(ends_with("cat")), list(factor)) %>%
  mutate_at(vars(ends_with("bin")), list(as.logical))

Logit_Model <- glm(target ~.,family=binomial(link='logit'),data=fac.data)

LogPred=data.frame(id = test_id, target = predict(Logit_Model, test.fac.data, type="response")) # Gini 0.26556
LogPred %>% write_csv("LogPred.csv")

#------------------------ naive bayes model --------------------
nbayes.mod <- naiveBayes(factor(target) ~ ., data = fac.data)

# get prediction class
set.seed(5059)
predictedClass <- predict(nbayes.mod, newdata = fac.data, type = "raw")

# get prediction class
set.seed(5059)
predicted.table <- predict(nbayes.mod, newdata = test.fac.data, type = "class")
table(predicted.table, test$target)


# number [9] is auc
nb.auc <- roc(test$target ~ predictedClass[,2], data = test)$auc
nb.gini <- 2*nb.auc - 1
nb.gini     # 0.2311129

# ------------------------ Ensemble Prediction -------------------------

ensemble_predictions <- function(pred_df, test_id) {
  
  preds <-  rowMeans(pred_df)
  
  return(data.frame(id = test_id, target = preds))
  
}

Enspred <- ensemble_predictions(data.frame(GBpred1$target, GBpred2$target, GBpred3$target, GAMpred, LogPred), test_id)
Enspred %>% write_csv("Enspred.csv")
