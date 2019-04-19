# _______________________________________
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# func_ModelFunctions.R
# Custom set of functions for data cleaning/prep & model tuning/creation
#

# /* ~~~~~~
#  *  Custom Data Clean FUnction
#  *  Goals ~ Remove unnessary columns, scale & allow one-hot encoding of categorical variables 
#  *  (1 stop shop, reducing code duplication)
#  *
#  *  To be done:   Enable Oversampling & Scaling of both Train/Test @ same time.
func_dataClean <- function( data, y_Response, bool_UseOneHotEncoding = FALSE, 
                            bool_encodeVariableAsFactor = FALSE ){
  data <- as.data.frame(data)
  print(paste('Start cleaning process - dataset contains ', length(names(data)), ' columns.', sep = ''))
  # update target value to factor variables
  column_yResponse <- grep(y_Response, names(data),ignore.case = TRUE) 
  if(class(data[, column_yResponse])!='factor'){
    print(paste('Change y response (',y_Response ,' column) to factor.',sep = ''))
    data[, column_yResponse] <- as.factor(ifelse(data[, column_yResponse]==0,"No","Yes"))
  }
  # now remove any column contain 'calc' (contains no prediction value)
  column_calc <- grep('calc', names(data),ignore.case = TRUE)
  if(length(column_calc)>0){
    print(paste('Remove any columns which contain the term calc ( ', length(column_calc), ' columns).', sep = '')) 
    data <- data[,-column_calc]
  }
  
  # remove any columns that contain single values only and scale x values
  print('Scale X Covariates.')
  data <- fun_scaleXCovariates(data, bool_UseOneHotEncoding)
  
  # appy one-hot encoding to factor columns if desired
  if(bool_UseOneHotEncoding){
    # now we can safely perform one-hot encoding
    data <- fun_factorOneHotEncoding(data, bool_encodeVariableAsFactor)
  }
  
  # remove id column (contains no prediction value)
  column_id <- match(tolower('id'), tolower(names(data)))
  if (!anyNA(column_id)){
    data <- data[,-column_id]
  }
  
  print(paste('Cleaned dataset contains ', length(names(data)), ' columns.', sep = ''))
  return(data)
}


# /* ~~~~~~
#  *  Custom One-Hot Encoding FUnction
#  *  Goals ~ generalise one-hot encoding technique for categorical variables
#  *  Allow users to choose whether they want the new encoded variables to be Factors or Numeric
#  *
fun_factorOneHotEncoding <- function(data, bool_encodeVariableAsFactor = FALSE){
  #names(data)[32] is ps_car_11_cat, which has 104 factor levels, 
  # for this reason will remove as one-hot ecoding would add 104 levels
  column_ToRemove <- 'ps_car_11_cat'
  column_massiveValueRange <- grep(column_ToRemove, names(data),ignore.case = TRUE)
  if(length(column_massiveValueRange)==1){ 
    data <- data[, -column_massiveValueRange]   
    print(paste('We will also remove ',column_ToRemove,' column as it would ', sep = ''))
    print('add 104 columns to the dataset (which might be overkill).')
  }
  # quick fix for now
  
  class_numbers <- c("numeric", 'integer')
  column_addTrace <- 0
  remove_trace <- 0
  column_names <- names(data)
  column_count <- length(column_names)
  for( i in 1:column_count){
    #i <- 4
    column_class <- class(data[,(i-remove_trace)])
    column_isNumber <- grep(column_class, class_numbers,ignore.case = TRUE)
    if( length(column_isNumber)>0){
      
      if( length(grep('_cat', column_names[i],ignore.case = TRUE)) >0){
        if(column_addTrace==0){ print('Changed following columns to one-hot encoding: ')}
        data[, (i-remove_trace)] <- as.factor(data[, (i-remove_trace)])
        dmy <- dummyVars(" ~ .", data = data[,c(1,(i-remove_trace))])
        trsf <- data.frame(predict(dmy, newdata = data[,c(1,(i-remove_trace))]))
        trsf <- trsf[,-1]
        if(bool_encodeVariableAsFactor) { trsf <- lapply(trsf, factor) }
        column_addTrace <- column_addTrace + length(names(trsf))
        data <- cbind(data, trsf)
        data <- data[, -(i-remove_trace)]
        remove_trace <- remove_trace + 1
      }
    }
  }
  return(data)
}  

# /* ~~~~~~
#  *  Custom ScaleXCovariate FUnction
#  *  Goals ~ generalise scaling of X-Covariates 
#  *  Also an opportunity to get rid of columns that contain single value only (i.e. add no predictive value)
#  * 
fun_scaleXCovariates <- function(data, bool_UseOneHotEncoding = FALSE){
  class_numbers <- c("numeric", 'integer')
  remove_trace <- 0
  column_names <- names(data)
  column_count <- length(column_names)
  for( i in 1:column_count){
    column_class <- class(data[,(i-remove_trace)])
    column_isNumber <- grep(column_class, class_numbers,ignore.case = TRUE)
    if( length(column_isNumber)>0){
      if(column_class=='integer'){
        data[,(i-remove_trace)] <- as.numeric(data[,(i-remove_trace)])
      }
      if( length(unique(data[,(i-remove_trace)]))==1){
        if(remove_trace==0){ print('Removed the following columns as they contain single value only: ')}
        print(column_names[i])
        data <- data[,-(i-remove_trace)]
        remove_trace <- remove_trace + 1
      } else {
        if( length(grep('_cat', column_names[i],ignore.case = TRUE))>0 ){
          if(!bool_UseOneHotEncoding){          
            data[,(i-remove_trace)] <-  fun_scaleSingleNumericColumn(data[,(i-remove_trace)])
          }
        } else {
          data[,(i-remove_trace)] <-  fun_scaleSingleNumericColumn(data[,(i-remove_trace)])
        }
      }
    }
  }
  return(data)
}

# /* ~~~~~~
#  *  Custom Scale Singel Numeric Column FUnction
#  *  Goals ~ support scaling of X-Covariates 
#  * 
fun_scaleSingleNumericColumn <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}




