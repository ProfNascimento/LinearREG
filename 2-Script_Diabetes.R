require(mlbench)
## IMPORT 
data(PimaIndiansDiabetes2)
names(PimaIndiansDiabetes2)
str(PimaIndiansDiabetes2)


Amelia::missmap(PimaIndiansDiabetes2)
DB=imputeTS::na.mean(PimaIndiansDiabetes2)

summary(DB)
boxplot(DB$glucose ~ DB$diabetes,ylab="Glucose",xlab="Diabetes")

corr=cor(DB[,-9])
library(ggcorrplot)
ggcorrplot(corr, hc.order = TRUE, type = "lower",
           lab = TRUE)

## LOGISTIC REG. MODEL
logistic_model <- glm(diabetes ~ ., DB, family='binomial')
summary(logistic_model)

boxplot(logistic_model$fitted.values ~ DB$diabetes)

yhat=ifelse(logistic_model$fitted.values>0.38,"pos","neg")
xtab=table(yhat,DB$diabetes)

caret::confusionMatrix(xtab)

# MODEL 2 (OPTIMAL)
logit_2 <- MASS::stepAIC(logistic_model)
summary(logit_2)


## DECISION TREE
library(rpart)      # for regression trees
decisionTree_model <- rpart(diabetes ~ . , DB, method = 'class')

probability <- predict(decisionTree_model, DB, type = 'prob')
boxplot(probability[,2] ~ DB$diabetes)
predicted_val <- predict(decisionTree_model, DB, type = 'class')

library(rpart.plot) # for plotting decision tree
rpart.plot(decisionTree_model)

## ROC-CURVE
library(pROC)
roc(DB$diabetes~logit_2$fitted.values, 
    plot = TRUE, main = "ROC CURVE", col = "blue")
roc(DB$diabetes~probability[,2], 
    plot = TRUE, main = "ROC CURVE", col = "red", add = TRUE)
legend("bottom",c("LOG.REG","DEC.TREE"),col=c("blue","red"),lty=c(1,1))
auc(DB$diabetes~logit_2$fitted.values)
