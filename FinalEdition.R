# --------------------------------Import required Packages ------------------------------------

# Visualisation
library(ggplot2)
library(grid)

# Data handling
library(tidyverse)
library(data.table)
library(tfestimators)

# Model Creation
library(caret)
library(xgboost)
library(MLmetrics)
library(randomForest)
library(nnet)
library(mgcv)

# -------------------------------------- Read in Data ------------------------------------------
source('func_ModelFunctions.R')
# Read in data
train <- as.data.table(fread('train.csv'))
test <- as.data.table(fread('test.csv'))
response <- "target"
# ------------------------------- Exploratory Data Analysis ------------------------------------

# Count missing values
perc_na <- function(data, na_val) {
  return(sum(data == na_val)/(nrow(data)*ncol(data))*100)
}

perc_na(train, -1)
perc_na(test, -1)

claimproportion <- function(data) {
  return(sum(data$target == 1)/nrow(data))
}

claimproportion(train)
claimproportion(test)

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
test_id <- test$id

train_notonehot <- func_dataClean(train, response, bool_UseOneHotEncoding = F, bool_encodeVariableAsFactor = T)
test_notonehot <- func_dataClean(test, response, bool_UseOneHotEncoding = F, bool_encodeVariableAsFactor = T)

train <- func_dataClean(train, response, bool_UseOneHotEncoding = T)
test <- func_dataClean(test, response, bool_UseOneHotEncoding = T)



train$target <- ifelse(train$target == "Yes", 1, 0)
train_notonehot$target <- ifelse(train_notonehot$target == "Yes", 1, 0)
# Undersample the 0's so 1's make up 10% of data
set.seed(5059)
oneset <- train[train$target==1,][sample(nrow(train[train$target==1,]), 21000),]
zeroset <- train[train$target==0,][sample(nrow(train[train$target==0,]), 189000),]
train <- rbind(oneset, zeroset)

set.seed(5059)
oneset_notonehot <- train_notonehot[train_notonehot$target==1,][sample(nrow(train_notonehot[train_notonehot$target==1,]), 21000),]
zeroset_notonehot <- train_notonehot[train_notonehot$target==0,][sample(nrow(train_notonehot[train_notonehot$target==0,]), 189000),]
train_notonehot <- rbind(oneset_notonehot, zeroset_notonehot)

set.seed(5059)
trainIndex <- createDataPartition(train$target, p = 0.7, list = FALSE, times = 1)

validsplit <- train[-trainIndex,]
trainsplit <- train[trainIndex,]
# ================================================ MODELS =================================================
# --------Gradient Boosting Model ----------
labels <- trainsplit$target 
ts_label <- validsplit$target
setDT(trainsplit)
setDT(validsplit)
setDT(test)

# Use one hot encoding to prepare data for xgboost package

# One hot encodes the factor variables
new_tr <- model.matrix(~.+0,data = trainsplit[,-c("target")]) 
new_ts <- model.matrix(~.+0,data = validsplit[,-c("target")])
new_test <- model.matrix(~.+0,data = test)

# Stores data for xgboost package
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dvalid <- xgb.DMatrix(data = new_ts,label=ts_label)
dtest <- xgb.DMatrix(data = new_test)

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

watchlist <- list(train=dtrain, valid=dvalid)

set.seed(5059)

xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 5, nfold = 5, nrounds=500, maximize = TRUE)

set.seed(5059)

gb_dt <- xgb.train(params = xgb_params,
                   
                   data = dtrain,
                   
                   print_every_n = 5,
                   
                   watchlist = watchlist,
                   
                   nrounds = 500)

mat <- xgb.importance(feature_names = colnames(new_tr),model = gb_dt)
xgb.plot.importance(importance_matrix = mat[1:20]) 

GBpred <- data.frame(id = test_id, target = predict(gb_dt, dtest))

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
neuralnetmodel <- caret::train(x_train,
                               y_train,
                               trControl = my_control,
                               method = "pcaNNet",
                               metric = "ROC",
                               tuneLength = 5
)

NNpred <- predict(neuralnetmodel, test, type = "prob")

NNpred <- pred$Yes

NNpred <- data.frame(id = test_id, target = predictions)

# ----------------------- Logistic Model -------------------
logistmodel <- glm(target ~.,family=binomial(link='logit'),data=train)

prob=predict(logistmodel, train, type="response")
NormalizedGini(y_pred = prob, y_true = train$target)
prob=predict(logistmodel, test, type="response")
NormalizedGini(y_pred = prob, y_true = testing_onehot$target)
LogModelpred <- data.frame(id = test_id, target = prob)
LogModelpred %>% write_csv('logmod2.csv')

# ----------------------- Random Forest --------------------

# build a random forest model with ntree 8000, mtry 10 (because it creates the highest gini score)
# sampling 1% of trainig data to build models and then combine many models into one model at the end
model_list <- list()
for (i in 1:100) {
  set.seed(5059*i)
  sample_row <- sample(nrow(train), 0.01 * nrow(train))
  s_data <- train[sample_row,]
  s_model <- randomForest(y = s_data[, 1], x = s_data[, -1], ntree=80, mtry =10)  
  model_list[[i]] <- s_model
}
combine_forests <- function(forestlist) {
  rf <- forestlist[[1]]
  for (i in 2:length(forestlist)) {
    rf <- randomForest::combine(rf, forestlist[[i]])
  }
  return(rf)
}


rf.all <- combine_forests(forestlist)
print(rf.all)
p1 <- predict(rf.all, training_onehot, type="prob")
NormalizedGini(y_pred = p1[,2], y_true = training_onehot_numeric$target)
p2 <- predict(rf.all, testing_onehot, type="prob")
NormalizedGini(y_pred = p2[,2], y_true = testing_onehot_numeric$target)

# ------------------- GAM --------------------
set.seed(5059)

fac.data <- train_notonehot %>%
  mutate_at(vars(ends_with("cat")), list(factor)) %>%
  mutate_at(vars(ends_with("bin")), list(as.logical)) %>%
  mutate(target = as.factor(target))


test.fac.data <- test_notonehot %>%
  mutate_at(vars(ends_with("cat")), list(factor)) %>%
  mutate_at(vars(ends_with("bin")), list(as.logical))


ini.mod.gam.cr.fac <- bam(target ~ s(ps_ind_01, k = 7, bs = "cr") + s(ps_ind_03, bs = "cr") + s(ps_ind_14, k = 4, bs = "cr") + 
                            s(ps_ind_15, k = 13, bs = "cr") + s(ps_reg_01, k = 1, bs = "cr") + s(ps_reg_02, k = 2, bs = "cr") + 
                            s(ps_car_11, k = 3, bs = "cr") + s(ps_car_12, k = 2, bs = "cr") + s(ps_car_13, k = 4, bs = "cr") + 
                            s(ps_car_15, k = 4, bs = "cr") + s(ps_car_14, k = 1, bs = "cr") + ps_ind_02_cat + ps_ind_04_cat +
                            ps_ind_05_cat + ps_car_01_cat + ps_car_02_cat + ps_car_04_cat + ps_car_06_cat + ps_car_07_cat + 
                            ps_car_08_cat + ps_car_09_cat + ps_car_10_cat + ps_car_11_cat + ps_ind_18_bin + ps_ind_17_bin + 
                            ps_ind_16_bin + ps_ind_13_bin + ps_ind_12_bin + ps_ind_11_bin + ps_ind_10_bin + ps_ind_09_bin + 
                            ps_ind_08_bin + ps_ind_07_bin + ps_ind_06_bin + ps_ind_04_cat + ps_reg_01 + ps_car_02_cat + 
                            ps_car_07_cat + ps_car_08_cat + ps_car_14,
                          family = binomial(link = 'logit'), data = train_notonehot, method = "GCV.Cp")

summary.gam(ini.mod.gam.cr.fac)

GAMpred <- predict(ini.mod.gam.cr.fac, test_notonehot, type = "response")

#------------------------ naive bayes model --------------------
nbayes.mod <- naiveBayes(factor(target) ~ ., data = train)

# get prediction class
set.seed(5059)
predictedClass <- predict(nbayes.mod, newdata = test[, -1], type = "raw")

# get prediction class
set.seed(5059)
predicted.table <- predict(nbayes.mod, newdata = test[, -1], type = "class")
table(predicted.table, test$target)


# number [9] is auc
nb.auc <- roc(test$target ~ predictedClass[,2], data = test)$auc
nb.gini <- 2*nb.auc - 1
nb.gini     # 0.2311129

# ------------------------ Ensemble Prediction -------------------------

ensemble_predictions <- function(pred_vector, test_id) {
  
  preddat <- data.frame(pred_vector)
  
  preds <-  rowMeans(preddat)
  
  return(data.frame(id = test_id, target = preds))
  
}

ensemble_predictions(c(GBpred, NNpred, LogModelpred, GAMpred), test_id)




