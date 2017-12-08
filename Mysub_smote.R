rm(list=ls(all=TRUE))
setwd("C:/Users/dnsh7/Desktop/Analytics/INSOFE/Hackathon")
train <- read.csv("train_final.csv")

#----------------Data Exploration---------------------- 

#Data Summary 
summary(train)
dim(train)

#data types
sapply(train,class)

#class distribution 
y <- train$Target
cbind(freq=table(y), percentage=prop.table(table(y))*100)

#peek
head(train,n=10)

#-----------------Data Cleaning------------------------

#finding NAs and dealing with them 
sum(is.na(train))
train <- na.omit(train)
summary(train)
dim(train)

#removing rows 
train <- train[,-c(1,4,16,17)]
str(train)
# Deleted the above columns coz they have zero sd

#--------------------Data Visualisation-------------------
# Correlation Plot
library(corrplot)
# calculate correlations
correlations <- cor(train[,1:23])
# create correlation plot
corrplot(correlations, method="circle")

#--------------------Data Preparation----------------------
#Subsetting train data
train_CatAtr<-subset(train,select=c(land,logged_in,root_shell,su_attempted,Target))
train_NumAtr<-subset(train,select=-c(land,logged_in,root_shell,su_attempted,Target))
train_NumAtr<-data.frame(apply(train_NumAtr,2,function(x){as.numeric(x)}))
train_CatAtr<-data.frame(apply(train_CatAtr,2,function(x){as.factor(x)}))
train <- data.frame(train_CatAtr,train_NumAtr)
str(train)

# using SMOTE to created a "more balanced" version of the dataset
library(DMwR)
balanced <- SMOTE(Target~., train, perc.over=300, perc.under=100)



#-------------------Model Building--------------------------
#Using Random Forest on (balanced) Train data
library(randomForest)
library(caret)
Forest.pred <- randomForest(Target ~ ., data = balanced,ntree = 30,mtry=5)
pred <- predict(Forest.pred,balanced,type='response')
conftab_train <- table(Forest.pred$predicted,balanced$Target)
confusionMatrix(conftab_train)
print(Forest.pred)
sort(Forest.pred$importance)

#ROC curve
library(ROSE)
par(mar = rep(2, 4))
roc.curve(balanced$Target, pred)

#-------------------- Testing ------------------------------
#loading test data
test <- read.csv("test_final.csv")
test <- test[,-c(1,4,16,17)]
str(test)

#subsetting test data
test_CatAtr<-subset(test,select=c(land,logged_in,root_shell,su_attempted,Target))
test_NumAtr<-subset(test,select=-c(land,logged_in,root_shell,su_attempted,Target))
test_NumAtr<-data.frame(apply(test_NumAtr,2,function(x){as.numeric(x)}))
test_CatAtr<-data.frame(apply(test_CatAtr,2,function(x){as.factor(x)}))
test <- data.frame(test_CatAtr,test_NumAtr)
str(test)

#make predictions on unseen(test) data
pred.Forest.test <- predict(Forest.pred, newdata = test)
conftab_test <- table(pred.Forest.test,test$Target)
confusionMatrix(conftab_test)


#drawing the roc curve for test data
par(mar = rep(2, 4))
roc.curve(test$Target, pred.Forest.test)

#------------Predicting on Evaluation data--------------------
eval <- read.csv("eval_final.csv")
eval <- eval[,-c(1,4,16,17)]

#Subsetting eval data
eval_CatAtr<-subset(eval,select=c(land,logged_in,root_shell,su_attempted))
eval_NumAtr<-subset(eval,select=-c(land,logged_in,root_shell,su_attempted))
eval_NumAtr<-data.frame(apply(eval_NumAtr,2,function(x){as.numeric(x)}))
eval_CatAtr<-data.frame(apply(eval_CatAtr,2,function(x){as.factor(x)}))
eval <- data.frame(eval_CatAtr,eval_NumAtr)
str(eval)
pred.Forest.eval <- predict(Forest.pred, newdata= eval)

#writing the predictions into a file
write.csv(pred.Forest.eval, file = "C:/Users/dnsh7/Desktop/Analytics/INSOFE/Hackathon/Mysub2.csv",row.names = F,col.names = "Predictions")

