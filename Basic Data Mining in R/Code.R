library(reshape2)
library(ggplot2)
library(plotly)
library(rpart)
library(partykit)
library(nnet)
library(adabag)
library(randomForest)
library(mvtnorm)
library(ROCR)
library(pROC)

bank <- read.csv(file="C:/Users/Luke/Documents/UCD/Data Mining/Labs & Assignments/bank.csv", header=TRUE, sep=",")
head(bank)
nrow(bank)
table(bank[,15])*100/sum(table(bank[,15]))

# check blanks
length(bank[,1])
for (i in 1:15){
  cat(i,":",length(which(is.na(bank[,i])==1)),"\n")
}

# Categorical Variables
cat_vars = c(2,3,4,5,7,8,10,14)
for (i in 1:8){
  bank[,cat_vars[i]] = as.factor(bank[,cat_vars[i]])
}

for (i in 1:8){
  bank_col = bank[,cat_vars[i]]
  full_table = table(bank_col,bank$y)
  row_no = full_table[1:length(table(bank_col))]
  row_yes = full_table[(length(table(bank_col))+1):(2*length(table(bank_col)))]
  full_table[1:length(table(bank_col))] = row_no/sum(row_no)
  full_table[(length(table(bank_col))+1):(2*length(table(bank_col)))] = row_yes/sum(row_yes)
  tab <- round((100*full_table)/(full_table[1:length(table(bank_col))]+full_table[(length(table(bank_col))+1):(2*length(table(bank_col)))]),digits=0)
  print(tab)
  print(colnames(bank)[cat_vars[i]])
  
  dat <- as.data.frame(tab)
  colnames(dat) = c(colnames(bank)[cat_vars[i]],"Subscribed?","Adjusted Percentage")
  if (max(dat[,3])-min(dat[,3])>50){
    scale = 10
  }
  else scale = 5
  # print(ggplot(dat, aes_q(x=as.name(names(dat)[1]), y=as.name(names(dat)[3]), fill=as.name(names(dat)[2]))) + 
  #   stat_summary(fun.y="mean", geom="bar",position="dodge") +
  #   scale_y_continuous(breaks = round(seq(0, max(dat[,3]), by = scale),1)))
  
  adjdat <- data.frame(dat[which(dat[,2]=="no"),c(1,3)])
  adjdat[,2] = dat[which(dat[,2]=="yes"),3]-dat[which(dat[,2]=="no"),3]
  adjdat[,3] = "Likely Yes (>=0)"
  adjdat[which(adjdat[,2]<0),3] = "Likely No (<0)"
  colnames(adjdat) <- c(colnames(adjdat)[1],"Likelihood of Term Deposit (%Yes-No)",">=0 OR <0")
  if (max(adjdat[,2])-min(adjdat[,2])>50){
    scale = 10
  }
  else scale = 5
  print(ggplot(adjdat, aes_q(x=as.name(names(adjdat)[1]), y=as.name(names(adjdat)[2]),fill=as.name(names(adjdat)[3]))) + 
    stat_summary(fun.y="mean", geom="bar") +
      labs(title="Bar Chart of Likelihood Percentages") +
    scale_y_continuous(breaks = round(seq(min(adjdat[,2]), max(adjdat[,2]), by = scale),1))) 
  
  if (i==1){
    print(ggplot(adjdat, aes_q(x=as.name(names(adjdat)[1]), y=as.name(names(adjdat)[2]),fill=as.name(names(adjdat)[3]))) + 
    stat_summary(fun.y="mean", geom="bar") +
    labs(title="Bar Chart of Likelihood Percentages") +
    scale_x_discrete(labels = abbreviate) +
    scale_y_continuous(breaks = round(seq(min(adjdat[,2]), max(adjdat[,2]), by = scale),1)))
  }
}


# Numerical Variables
num_vars = c(1,6,9,11,12,13)

for (i in 1:6){
  cat("\n\n",colnames(bank)[num_vars[i]], ": ","\n","Yes: ","\n",
            summary(bank[which(bank[,15]=="yes"),num_vars[i]])[4],"\n",
            "No: ","\n" ,
            summary(bank[which(bank[,15]=="no"),num_vars[i]],"\n")[4])
}
summary(bank$age)
summary(bank$balance)
summary(bank$campaign)
summary(bank$day)
summary(bank$pdays)
summary(bank$previous)

for (i in 1:6){
  print(ggplot(data=bank, aes_q(x=as.name(names(bank)[num_vars[i]]),fill=as.name(names(bank)[15]))) + 
          labs(title="Distribution of Values for Subcribers (Y) & Unsubscribers (N)") +
          geom_histogram(aes_q(x=as.name(names(bank)[num_vars[i]]),y=as.name("..density..")), position="dodge"))
}

ggplot(data=bank[which(bank$campaign>4),], aes(x=campaign,fill=y)) + labs(title="Distribution of Values for Subcribers (Y) & Unsubscribers (N)") + geom_histogram(aes(x=campaign,y=..density..), position="dodge")
ggplot(data=bank[which(bank$pdays>2),], aes(x=pdays,fill=y)) +labs(title="Distribution of Values for Subcribers (Y) & Unsubscribers (N)") + geom_histogram(aes(x=pdays,y=..density..), position="dodge")
ggplot(data=bank[which(bank$previous>2),], aes(x=previous,fill=y)) + labs(title="Distribution of Values for Subcribers (Y) & Unsubscribers (N)") + geom_histogram(aes(x=previous,y=..density..), position="dodge")
ggplot(data=bank, aes(x=balance,fill=y)) + labs(title="Distribution of Values for Subcribers (Y) & Unsubscribers (N)") + geom_histogram(aes(x=previous,y=..density..), position="dodge")

round(cor(bank[,num_vars],method="pearson"),2)

for (i in 1:6){  
  cat(colnames(bank[num_vars[i]]),":", t.test(bank[which(bank[,15]=="no"),num_vars[i]],
               bank[which(bank[,15]=="yes"),num_vars[i]])$p.value, "\n")
  i = i+1
}

# normalise all numerical variables
for (i in 1:6){
  bank[,num_vars[i]] = (bank[,num_vars[i]]-min(bank[,num_vars[i]]))/(max(bank[,num_vars[i]])-min(bank[,num_vars[i]]))
}

bank2 = bank
bank2$age = bank2$age*87
ggplot(data=bank2, aes(x=age,fill=y)) + geom_histogram(aes(x=age,y=..density..), position="dodge")


# normalise all ordinal variables (just month)
# bank[which(bank$month=="jan"),16] = 1
# bank[which(bank$month=="feb"),16] = 2
# bank[which(bank$month=="mar"),16] = 3
# bank[which(bank$month=="apr"),16] = 4
# bank[which(bank$month=="may"),16] = 5
# bank[which(bank$month=="jun"),16] = 6
# bank[which(bank$month=="jul"),16] = 7
# bank[which(bank$month=="aug"),16] = 8
# bank[which(bank$month=="sep"),16] = 9
# bank[which(bank$month=="oct"),16] = 10
# bank[which(bank$month=="nov"),16] = 11
# bank[which(bank$month=="dec"),16] = 12
# bank[,16] = round((bank[,16])/12,digits=2)
# table(bank[,16])
# bank[,10] = bank[,16]
# bank = bank[,1:15]





set.seed(100)
N <- nrow(bank)
splits <- rep(1:3,ceiling(N/3))
splits <- sample(splits)
splits <- splits[1:N]
# Run through rest using splits==3 then 2 then 1
indtest <- (1:N)[(splits==3)]
indrest <- setdiff(1:N,indtest)
n <- length(indrest)
K <- 10
num = ceiling(n/K)
pred<-matrix(NA,length(indrest),8)
pred[,1] = bank[indrest,15]

for (iter in 1:K){
  ind = (iter*num-(num-1)):(min(iter*num,n))
  indvalid <- indrest[ind]
  indtrain <- setdiff(indrest,indvalid)
  
  fit.tree <- rpart(y~.,data=bank,subset=indtrain)
	pred[ind,2] <- predict(fit.tree,type="class",newdata=bank[indvalid,])
	
	fit.log <- multinom(y~.,data=bank,subset=indtrain)
	pred[ind,3] <- predict(fit.log,type="class",newdata=bank[indvalid,])
	
	fit.bag <- bagging(y~.,data=bank,subset=indtrain)
	pred[ind,4] <- predict(fit.bag,type="class",newdata=bank[indvalid,])$class
	
	fit.boost <- boosting(y~.,data=bank,boos=FALSE,coeflearn="Freund",subset=indtrain)
	pred[ind,5] <- predict(fit.boost,type="class",newdata=bank[indvalid,])$class
  
	fit.rf <- randomForest(y~.,data=bank,subset=indtrain)
	pred[ind,6] <- predict(fit.rf,type="class",newdata=bank[indvalid,])
}

pred[which(pred[,4]=="no"),4] = "1"
pred[which(pred[,5]=="no"),5] = "1"
pred[which(pred[,4]=="yes"),4] = "2"
pred[which(pred[,5]=="yes"),5] = "2"
pred[,8] = "1"
pred[,7] = "1"
rand = runif(length(indtest),0,1)
pred[which(rand<=0.1),7] = "2"
results <- matrix(NA,7,9)
results <- data.frame(results)
rownames(results)<-c("CA","No_Recall", "No_Precision", "No_F1",
                     "Yes_Recall", "Yes_Precision", "Yes_F1")
colnames(results)<-c("Tree","Log", "Bag", "Boost","Forest","Random","ZeroR","Max","Margin %")
for (i in 1:6){
  tab <- table(pred[,1],pred[,i+1]) # rows=truth, cols=prediction
  results[1,i] = round(sum(diag(tab))*100/sum(tab),3) # Total Classification Accuracy
  # NO
  results[2,i] = round(tab[1]*100/sum(tab[c(1,3)]),3) # Recall
  results[3,i] = round(tab[1]*100/sum(tab[c(1,2)]),3) # Precision
  results[4,i] = round(2*results[3,i]*results[2,i]/(results[3,i]+results[2,i]),3) # f1
  # Yes
  results[5,i] = round(tab[4]*100/sum(tab[c(2,4)]),3) # Recall
  results[6,i] = round(tab[4]*100/sum(tab[c(3,4)]),3) # Precision
  results[7,i] = round(2*results[6,i]*results[5,i]/(results[6,i]+results[5,i]),3) # f1
}

tab <- table(pred[,1],pred[,8]) # rows=truth, cols=prediction
results[1,7] = round(sum(diag(tab))*100/sum(tab),3) # Total Classification Accuracy
# NO
results[2,7] = 100 # Recall
results[3,7] = round(tab[1]*100/sum(tab[c(1,2)]),3) # Precision
results[4,7] = round(2*results[3,7]*results[2,7]/(results[3,7]+results[2,7]),3) # f1
# Yes
results[5,7] = 0 # Recall
results[6,7] = 0 # Precision
results[7,7] = 0 # f1

for (r in 1:nrow(results)){
  results[r,8] = colnames(results)[which.is.max(results[r,1:7])]
  results[r,9] = round(100*(as.double(max(results[r,1:7]))-as.double(sort(results[r,1:7])[4]))/(as.double(sort(results[r,1:7])[4])),2)
}

colnames(pred) = c("True","Tree","Log","Bag","Boost","RF","ZeroR","Random")
for (i in 2:7){
  for (j in (i+1):8){
    n00 = n11 = n10 = n01 = whoops = 0
    for (r in 1:nrow(pred)){
      if ((!(pred[r,1]==pred[r,i]))&&
          (pred[r,i]==pred[r,j])){
        n00 = n00 + 1
      } else if ((pred[r,1]==pred[r,j])&&
               (pred[r,i]==pred[r,j])) {
        n11 = n11 + 1
      } else if ((pred[r,1]==pred[r,j])&&(!(pred[r,i]==pred[r,j]))){
        n10 = n10 + 1
      } else if ((pred[r,1]==pred[r,i])&&(!(pred[r,i]==pred[r,j]))){
        n01 = n01 + 1
      } else {
        whoops = whoops + 1
      }
    }
  if(whoops>0|n01+n00+n10+n11!=nrow(pred)){print("WHOOPS")}
  mcnemar <- ((abs(n01-n10)-1)^2)/(n01+n10)
  cat(colnames(pred)[i],"&",colnames(pred)[j],": ",mcnemar,"\n")
  }
}

fitboost <- boosting(y~.,data=bank,boos=FALSE,coeflearn="Freund",subset=indrest)
boost_pred <- predict(fitboost,type="class",newdata=bank[indtest,])$class
boost_tab <- table(bank[indtest,15],boost_pred)
round(100*sum(diag(boost_tab))/sum(boost_tab),2)
# ZeroR
#round(100*table(bank[indtest,15])[1]/sum(table(bank[indtest,15])),2)
# Random
# rand = runif(length(indtest),0,1)
# rand_pred = c(rep("no",length(indtest)))
# rand_pred[which(rand<=0.1)] = "yes"
# rand_tab <- table(bank[indtest,15],rand_pred)
# round(100*sum(diag(rand_tab))/sum(rand_tab),2)


# ROC Curve & AUC for Test Set
# Boost
fitboost <- boosting(y~.,data=bank,boos=FALSE,coeflearn="Freund",subset=indrest)
boost_prob <- predict(fitboost,type="class",newdata=bank[indtest,])$prob
colnames(boost_prob) = c("Prob(No)","Prob(Yes)")
fitbag <- bagging(y~.,data=bank,subset=indrest)
bag_prob <- predict(fitbag,type="class",newdata=bank[indtest,])$prob
colnames(bag_prob) = c("Prob(No)","Prob(Yes)")

truth = matrix(0,length(indtest),2)
colnames(truth) = c("No=1","Yes=1")
truth[which(bank[indtest,15]=="no"),1]=1
truth[which(bank[indtest,15]=="yes"),2]=1
no_roc_pred <- prediction(boost_prob[,1], truth[,1])
no_perf <- performance(no_roc_pred, "tpr", "fpr")
yes_roc_pred <- prediction(boost_prob[,2], truth[,2])
yes_perf <- performance(yes_roc_pred, "tpr", "fpr")
no_roc_pred2 <- prediction(bag_prob[,1], truth[,1])
no_perf2 <- performance(no_roc_pred2, "tpr", "fpr")
yes_roc_pred2 <- prediction(bag_prob[,2], truth[,2])
yes_perf2 <- performance(yes_roc_pred2, "tpr", "fpr")

plot(no_perf, col="blue",main="ROC Curve for Predicting No to Term Deposit")
plot(no_perf2, add=TRUE, col="purple")
legend("bottomright",legend=c("Boosting", "Bagging"),
       col=c("blue", "purple"), lty=1:1, cex=0.8)
abline(0,1,col="red",lty=2)

plot(yes_perf, col="blue",main="ROC Curve for Predicting Yes to Term Deposit")
plot(yes_perf2, add=TRUE, col="purple")
legend("bottomright",legend=c("Boosting", "Bagging"),
       col=c("blue", "purple"), lty=1:1, cex=0.8)
abline(0,1,col="red",lty=2)

AUC = matrix(0,2,2)
colnames(AUC) = c("No","Yes")
rownames(AUC) = c("Boost","Bag")
AUC[1,1] = auc(truth[,1],boost_prob[,1])
AUC[1,2] = auc(truth[,2],boost_prob[,2])
AUC[2,1] = auc(truth[,1],bag_prob[,1])
AUC[2,2] = auc(truth[,2],bag_prob[,2])
print(AUC)

# variable importance plot (for both gain in Gini Index)
imp_boost = sort(fitboost$importance,decreasing = TRUE)
imp_bag = sort(fitbag$importance,decreasing = TRUE)
imp = data.frame(matrix(NA,length(colnames(bank[,1:14])),3))
imp[,1] = colnames(bank[,1:14])
for (i in 1:nrow(imp)){
  imp[i,2] = eval(parse(text=paste0("imp_boost['",imp[i,1],"']",sep="")))
  imp[i,3] = eval(parse(text=paste0("imp_bag['",imp[i,1],"']",sep="")))
}
colnames(imp) = c("Variables","Boost","Bag")

ggplot(imp, aes(x=reorder(Variables, -Boost), y=Boost),main="Bar Plot of variable Importance") + 
  stat_summary(fun.y="mean", geom="bar") +
  scale_x_discrete(labels = abbreviate) +
  labs(title="Bar Chart of Variable Importance in Boosting Model",
        x ="Variables", y = "Mean Increase in Gini Index")

ggplot(imp, aes(x=reorder(Variables, -Bag), y=Bag),main="Bar Plot of Variable Importance") + 
  stat_summary(fun.y="mean", geom="bar") +
  scale_x_discrete(labels = abbreviate) +
  labs(title="Bar Chart of Variable Importance in Bagging Model",
        x ="Variables", y = "Mean Increase in Gini Index")

nor_imp = imp
nor_imp[,2] = nor_imp[,2]/sum(nor_imp[,2])
nor_imp[,3] = nor_imp[,3]/sum(nor_imp[,3])
ggplot(melt(nor_imp, id=c("Variables")), aes(x=Variables, y=value, fill=variable),main="Bar Plot of Variable Importance") + 
  stat_summary(fun.y="mean", geom="bar", position="dodge") +
  scale_x_discrete(labels = abbreviate) +
  labs(title="Bar Chart of Relative Variable Importance in Both Models",
        x ="Variables", y = "Normalised Mean Increase in Gini Index")

# closer look at model
fit.tree <- rpart(y~.,data=bank,subset=indrest)
plot(fit.tree,margin=0.05,main="Decision Tree")
text(fit.tree, use.n=TRUE, all=TRUE, cex=.7,col="black")

fit.log <- multinom(y~.,data=bank,subset=indrest)
ncoef = length(summary(fit.log)$coefficients)
logcoef = data.frame(matrix(0,ncoef,6))
logcoef[,1] = names(summary(fit.log)$coefficients)
logcoef[,2] = summary(fit.log)$coefficients
logcoef[,4] = summary(fit.log)$coefficients/summary(fit.log)$standard.errors
for (i in 1:ncoef){
  if (logcoef[i,2]<0){
    logcoef[i,3]="Negative"
  }  else {
    logcoef[i,3]="Positive"
  }
  logcoef[i,5] = (1 - pnorm(abs(logcoef[i,4]), 0, 1)) * 2
  if (logcoef[i,5]<0.05){
    logcoef[i,6]="Yes"
  }  else {
    logcoef[i,6]="No"
  }
}
rankings = rank(logcoef[,5])
colnames(logcoef) = c("Variables","Coefficients","Effect on Subscribership",
                      "Z-Scores","PValues","Significant")
logcoef <- logcoef[order(logcoef$PValues),] 
rownames(logcoef) <- NULL
logcoef

toplogcoef <- logcoef[which(logcoef[,6]=="Yes"),]
toplogcoef


# try best model on the test set
fitboost <- boosting(y~.,data=bank,boos=FALSE,coeflearn="Freund",subset=indrest)
boost_pred <- predict(fitboost,type="class",newdata=bank[indtest,])$class
boost_tab <- table(bank[indtest,15],boost_pred)
tab <- boost_tab
ca = round(sum(diag(tab))*100/sum(tab),3) # Total Classification Accuracy
# No
rn = round(tab[1]*100/sum(tab[c(1,3)]),3) # Recall
pn = round(tab[1]*100/sum(tab[c(1,2)]),3) # Precision
fn = round(2*rn*pn/(rn+pn),3) # f1
# Yes
ry = round(tab[4]*100/sum(tab[c(2,4)]),3) # Recall
py = round(tab[4]*100/sum(tab[c(3,4)]),3) # Precision
fy = round(2*ry*py/(ry+py),3) # f1
cat(ca,rn,pn,fn,ry,py,fy)

fitboost <- boosting(y~.,data=bank,boos=FALSE,coeflearn="Freund")
boost_pred <- predict(fitboost,type="class")$class
boost_tab <- table(bank[indtest,15],boost_pred)
tab <- boost_tab
ca = round(sum(diag(tab))*100/sum(tab),3) # Total Classification Accuracy
# No
rn = round(tab[1]*100/sum(tab[c(1,3)]),3) # Recall
pn = round(tab[1]*100/sum(tab[c(1,2)]),3) # Precision
fn = round(2*rn*pn/(rn+pn),3) # f1
# Yes
ry = round(tab[4]*100/sum(tab[c(2,4)]),3) # Recall
py = round(tab[4]*100/sum(tab[c(3,4)]),3) # Precision
fy = round(2*ry*py/(ry+py),3) # f1
cat(ca,rn,pn,fn,ry,py,fy)
