# Lily Yan 
# IMT 572 Mini Project
# run every time library import
library(randomForest)
library(mlbench)
library(naivebayes)
library(Hmisc)
library(caret)
library(e1071)
library(party)
library(kernlab)
library(mfx)


rm(list=ls())

setwd("C:\\downloads")
d = read.csv("usopen-w.csv") # Tennis match dataset for women at the US Open Tournament 2013



#########################
# EXPLORATORY ANALYSIS
#########################
head(d)

# this shows us we have various NAs in variables ST3.1, ST3.2, ST4.1, ST4.2, ST5.1, ST5.2
summary(d)

View(d)

# running correlation analysis to understand relationships between variables
# in this case, this is seeing the correlation between y = match results and ST (set results)
# returns cor of 0.68 which is a strong positive correlation
# however this does not mean that variable ST1.1 is a strong predictor or the cause of win/loss
r_set_result = c(d$Result, d$ST1.1, d$ST1.2, d$ST2.1, d$ST2.2, d$ST3.1, d$ST3.2)
correlation_test = cor.test(d$Result, d$ST1.1)
print(correlation_test)

# cor for match result and set 2 results of player 1: 0.66
correlation_test = cor.test(d$Result, d$ST2.1)
print(correlation_test)

# cor for match result and set 2 results of player 2: -0.61 
correlation_test = cor.test(d$Result, d$ST2.2)
print(correlation_test)

# cor for match result and unforced errors by player 1: -0.26
correlation_test = cor.test(d$Result, d$UFE.1)
print(correlation_test)

# cor for match result and double faults committed by player 1: -0.34
correlation_test = cor.test(d$Result, d$DBF.1)
print(correlation_test)


hist(d$FSP.1) 
hist(d$FSP.2) 

#########################
# PRE-PROCESSING
#########################

# although my variables are all numeric values and my output is already a binary outcome
# I need to default and use as.factor()

# outcome default = 1 if match = win referenced to player 1.
d$Result[d$Result=="Win"] = "1"
d$Result[d$Result=="Lose"] = "0"
d$Result = as.factor(d$Result)

# mean imputation of missing values in the following variables.
d$NPA.1[is.na(d$NPA.1)] = mean(d$NPA.1, na.rm = TRUE)
d$NPA.2[is.na(d$NPA.2)] = mean(d$NPA.2, na.rm = TRUE)
d$NPW.1[is.na(d$NPW.1)] = mean(d$NPW.1, na.rm = TRUE)
d$NPW.2[is.na(d$NPW.2)] = mean(d$NPW.2, na.rm = TRUE)

summary(d)

# let's run logit using the following variables.
summary(glm(Result~FSW.1+FSW.2+ACE.1+WNR.1+DBF.2+UFE.1+UFE.2,d,family="binomial"))
logitmfx(Result~FSW.1+FSW.2+ACE.1+WNR.1+DBF.2+UFE.1+UFE.2,d)
# AIC 38.738

# then I will run a probit model on the same variables.
summary(glm(Result~FSW.1+FSW.2+ACE.1+WNR.1+DBF.2+UFE.1+UFE.2, data = d, 
    family=binomial(link="probit")))
probitmfx(Result~FSW.1+FSW.2+ACE.1+WNR.1+DBF.2+UFE.1+UFE.2, data = d)
# AIC 39.388

# tried logit model with different predictor combos.
summary(glm(Result~FSW.1+FSW.2+DBF.1+WNR.1+DBF.2+UFE.1+UFE.2,d,family="binomial"))
logitmfx(Result~FSW.1+FSW.2+DBF.1+WNR.1+DBF.2+UFE.1+UFE.2,d)
# AIC 41.625

#########################
# PREDICTION MODEL
#########################


# this implements cross-validation (keep the same across runs so you can compare on the same data)
set.seed(0)
train_Control = trainControl(method = "cv", number = 10)

# this trains the knn model using the cross-validation splits
knn_caret = train(Result~FSW.1+FSW.2+ACE.1+WNR.1+DBF.2+UFE.1+UFE.2,
                  data=d, method="knn", trControl=train_Control,
                  tuneLength=20)
knn_caret

# accuracy versus number of neighbors
plot(knn_caret)

new_df = data.frame(FSW.1=d$FSW.1,
                    FSW.2=d$FSW.2,
                    ACE.1=d$ACE.1,
                    WNR.1=d$WNR.1,
                    DBF.2=d$DBF.2,
                    UFE.1=d$UFE.1,
                    UFE.2=d$UFE.2,
                    Result=d$Result)

# creating prediction data to assess accuracy
predictions = predict(knn_caret, newdata = new_df)
confusion_matrix = confusionMatrix(predictions, reference = new_df$Result)
print(confusion_matrix)


# training decision tree model using cross-validation splits
tree_caret = train(Result~FSW.1+FSW.2+ACE.1+WNR.1+DBF.2+UFE.1+UFE.2,
                   data=d, method = "ctree2", trControl = train_Control,
                   tuneLength=25)
tree_caret
plot(tree_caret)

# creating prediction data 
new_df = data.frame(FSW.1=d$FSW.1,
                    FSW.2=d$FSW.2,
                    ACE.1=d$ACE.1,
                    WNR.1=d$WNR.1,
                    DBF.2=d$DBF.2,
                    UFE.1=d$UFE.1,
                    UFE.2=d$UFE.2,
                    Result=d$Result)

# creating confusion matrix to assess accuracy
predictions = predict(tree_caret, newdata = new_df)
confusion_matrix = confusionMatrix(predictions, reference = new_df$Result)
print(confusion_matrix)



# train Naive Bayes model
naiveB_model = naiveBayes(Result~FSW.1+FSW.2+ACE.1+WNR.1+DBF.2+UFE.1+UFE.2,
                           data=d)
# print summary of trained model 
print(naiveB_model)

# creating dataframe and predictions
new_df = data.frame(FSW.1=d$FSW.1,
                    FSW.2=d$FSW.2,
                    ACE.1=d$ACE.1,
                    WNR.1=d$WNR.1,
                    DBF.2=d$DBF.2,
                    UFE.1=d$UFE.1,
                    UFE.2=d$UFE.2,
                    Result=d$Result)
predictions = predict(naiveB_model, newdata = new_df)

confusion_matrix = table(predictions, new_df$Result)
print(confusion_matrix)






