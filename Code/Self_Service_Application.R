source("FinalEdition.R")

# Visualisation
library(ggplot2)
library(grid)

# Data handling
library(tidyverse)
library(data.table)
library(Matrix)

# Model Creation
library(caret)
library(xgboost)
library(MLmetrics)
library(randomForest)
library(nnet)
library(mgcv)
library(e1071)

# Could source a file containing just models, however this is over 1gb so not suitable for upload
# For substitute, representation

# source("ModelFile.RData")

# Minimum_threshold is probability at which people will be classified high-risk
# Default is produced from technical analysis

probabilitypredictions <- function(data, minimum_threshold = 0.305) {
  # Prepare multiple data sets required for each model
  test <- data
  test_id <- data$id
  response <- data$target
  setDT(test)
  new_test <- model.matrix(~.+0,data = test)
  dtest <- xgb.DMatrix(data = new_test)
  data[, amount_nas := rowSums(data == -1, na.rm = T)]
  data[, high_nas := ifelse(amount_nas>4,1,0)]
  data[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
  data[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
  data[, ps_custom_bin := ifelse( ps_ind_05_cat == -1 |  ps_ind_05_cat == -2 | ps_car_07_cat == -1 | ps_car_11_cat == 41, 1, 0) ]
  testfeat <- data
  new_testfeat <- model.matrix(~.+0,data = testfeat)
  dtestfeat <-  xgb.DMatrix(data = new_testeat)
  test_notonehot <- func_dataClean(test, response, bool_UseOneHotEncoding = F, bool_encodeVariableAsFactor = T)
  test <- func_dataClean(test, response, bool_UseOneHotEncoding = T)
  test.fac.data <- test_notonehot %>%
    mutate_at(vars(ends_with("cat")), list(factor)) %>%
    mutate_at(vars(ends_with("bin")), list(as.logical))
  
  # Find component predictions
  GBpred1 <- predict(GB_Model1, dtest)
  GBpred2 <- predict(GB_Model2, dtest)
  GBpred3 <- predict(GB_Model3, dtestfeat)
  Logitpred <-predict(predict(Logit_Model, test, type="response"))
  GAMpred <- predict(ini.mod.gam.cr.fac, test_notonehot, type = "response")
  
  # Find ensemble predictions
  Predictions <- ensemble_predictions(data.frame(GBpred, NNpred, Multilogpred, logpred, GAMpred))
  # Denote if high risk according to input threshold
  Predictions$highrisk <- (Predictions$target > minimum_threshold)  
  return(Predictions)
}

ensemble_predictions <- function(pred_df, test_id) {
  
  preds <-  rowMeans(pred_df)
  
  return(data.frame(id = test_id, target = preds))
  
}

# As we can't load in models this is a representation of how the ensemble part works, predictions from all 5 component models are loaded in
GBpred1 <- load("ComponentPredictions/GBpred1.csv")
GBpred2 <- load("ComponentPredictions/GBpred2.csv")
GBpred3 <- load("ComponentPredictions/GBpred3.csv")
LogPred <- load("ComponentPredictions/LogPred.csv")
GAMpred <- load("ComponentPredictions/GAMpred.csv")

test_id <- GBpred1$id
Predictions <- ensemble_predictions(data.frame(GBpred1$target, GBpred2$target, GBpred3$target, LogPred$target, GAMpred$target), test_id = test_id)
Predictions$highrisk <- (Predictions$target > 0.305)  





