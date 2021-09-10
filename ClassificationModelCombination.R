#1. introduction of classification
#the field of supervised machine learning offers numerous classification methods that can be used to predict categorical outcomes, including logistic regression, decision trees, random forests, support vector machiens and neural networks
#supervised learning starts with a set of observations containing values for both the predictor variables and outcome

pkgs <-c("rpart","rpart.plot","party","randonForest", "e1071")
install.packages(pkgs, depend=T)

#2. preparing the data

#data contains 699 fine-needle aspirate samples, where 458 are benign and 241 are malignant
#11 variables and doesnt include the variable names in the file
#sixteen samples have missing data and are coded in the text file with a question mark(?)

loc <-"http://archive.ics.uci.edu/ml/machine-learning-databases/"
ds <-"breast-cancer-wisconsin/breast-cancer-wisconsin.data"
url <-paste(loc, ds, sep="")
breast <-read.table(url, sep=",", header=F,na.strings="?")
names(breast) <-c("ID","clumpThickness","sizeUniformity",
                  "shapeUniformity","maginalAdhesion",
                  "singleEpithelialCellSize", "bareNuclei",
                  "blandChromatin","normalNucleoli","mitosis","class")
df<-breast[-1]
df$class <-factor(df$class, levels=c(2,4), labels=c("benign","malignant"))
set.seed(1234)
train <-sample(nrow(df),0.7*nrow(df))
df.train<-df[train,]
df.validate<-df[-train,]
table(df.train$class)
table(df.validate$class)

#3. logistic regression: used to predict a binary outcome from a set of numeric variables

fit.logit <-glm(class~., data=df.train, family=binomial())
summary(fit.logit)

prob <-predict(fit.logit, df.validate, type="response")#by default, the predict function predicts the log odds of having a malignant outcome
logit.pred <-factor(prob >.5, levels=c(FALSE, TRUE),#cases with probabilities greater than .5 are classified into the malignant group and cases with probabilities less than or equal to .5 are classified as benign
                    labels=c("benign","malignant"))
logit.perf <-table(df.validate$class, logit.pred, dnn=c("Actual", "Predicted"))#cross-tabulation of actual status and predicted statis
logit.perf#type="response", the probability of obatining a malignant classification is returned 

#first, a logistic regression model is fit using class as the dependent variable and the remaining variables a predictors
#the model is based on the cases in the df.train data frame


#the confusion matrix shows that 118 cases that were benign were classified as benign, and 76 cases that were malignant were classified as malignant, ten cases in the df.validate data frame had missing predictor data and could not be included in the validation

#three of the predictor variables(sizeUniformity, shapeUniformity and singleEpithelialCellSize) have coefficients that dont differ from zero at the p<.10 level
#should we do with predictor variables that have nonsignificant coefficients?

#in a prediction context, it is often useful to remove such variables from the final model
#this is especially important in situations where a large number of non-informative predictor variables are adding what is essentially noise to the system

#in this case, stepwise logistic regression can be used to generate a smaller model with fewer variables
logit.fit.reduced <-step(fit.logit)#to obtain a more parsimonious model, the reduced model excludes the three variables mentined previsouly, when used to predict outcomes n the validation dataset, this reduced model makes fewer errors

#3. decision trees
#creating a set of binary splits on the predictor variables in order to create a tree tha tcan be used to classify new observations into one of two groups
#two types: classical trees, and conditional inference trees

#4. classical decision trees
#steps of algorithm:
#a. choose the predictor variable tha tbest splits the data into two groups that the purity(homogeneity) of the outcome in the two groups is maximized .
# if the predictor is continuous, choose a cut-point that maximizes purity for the two groups created. if the predictor variable is categorical(not applicable in this case), combine the categories to obtain two groups with maximum purity
#b. separate the data into these two groups ,and continue the process for each subgroup
#c. repeat steps a and b until a subgroup contains fewer than a minimum number of observations or no splits decrease the impurity beyond a specified threshold
# the subgroups in the final set are called terminal nodes, each terminal node is classified as one category of the outcome or the other based on the most frequent value of the outcome for the sample in that node
#d. to classify a case, run it down the tree to a terminal node, and assign it the model outcome value assigned in step c

#decision trees can be grown and pruned using the rpart() and prune() functions in the rpart package

library(rpart)
set.seed(1234)
dtree <-rpart(class~., data=df.train, method="class",
              parms=list(split="information"))#tree growing using the rpart() function, we can use print(dtree) and summary(dtree) to examine the fitted model
dtree$cptable#in order to choose a final tree size, examine the cptable component of the list returned by rpart()
plotcp(dtree)#plots cross-validated error against the complexity parameters, good choice for the finall tree size is the smallest tree whose cross validaed erro r is wihin one sd of the minimim corss validated error value
dtree.pruned<-prune(dtree, cp=.0125)#complexity parameter cp is used to penalize larger trees. tree size is defined by the number of branch splits(nsplit)
library(rpart.plot)
prp(dtree.pruned, type=2, extra=104,
    fallen.leaves = T, main="Decision Tree")
dtree.pred <-predict(dtree.pruned, df.validate, type="class")
dtree.perf <-table(df.validate$class, dtree.pred, dnn=c("Actual","Predicted"))
dtree.perf

#rel error column contains the error rate for a tree of a given size in the training sample
#the cross-validated error(xerror) is based on 10-fold cross validation(also using the training sample)
#the xstd column contains the standard error of the cross validation error

#the minimum cross-validated error is .18 with a standard error of .0326
#in this case, the smallest tree with a cross validated error within .18+/-.0326(that is ,between .15 and .21) is selected
#we can select the tree size associated with the largest complexity parameter below the line, resulst suggest a tree with three splits

#prune() uses the complexity parameter to cut back a tree to the desired size, it takes the full tree and snips off the least important splits based on the desired complexity parametrer
#a tree in this case with three splits has a complexity parameter of .0125, sothe statement prune(dtree, cp=.0125) returns a tree with the desired size

#prp() in the rpart.plot package is used to draw an attractive plot of the final decision tree
#type=2 draws the split labels below each node
#extra=104 includes the probabilities for each class, along with the percentage of observations in each node
#fallen.leaves=T option displays the terminal nodes at the botton of the graph

#to classify an observation, start at the top of the tree, moving to the left brank if a condition is true or the right otherwise. Continue moving down the tree until we hit a terminal node, classfity the observation using the label of the node

#finally, the predict() is used to classify each observation in the validation sample, a cross-tabulation of the actual status against the predicted status is provided
#the overall accuracy was 96% in the validation sample.
#unlike the logistic regression example, all 210 cases in the validation sample could be classified by the final tree. N
#Note that decision trees can be biased toward selecting predictors that have many levels or many missing values

#5. conditional inference trees

#difference from traditional trees is that the variables and spliys are selected based on significance tests rather than purity/homogeneity measures
#the significance tests are permutation tests

#steps of conditional inference trees
#calculate p-values for the relationship between each predictor and the outcome variable
#select the predictor with the lower p value
#explore all possible binary splits on the chosen predictor and dependent variable(using permutation tests) and pick the most significant split
#separate the data into these two groups, and continue the process for each subgroup
#continue untial splits are no longer significant or the minimum node size is reached

ctree()#in the party package
library(party)
fit.ctree <-ctree(class~.,data=df.train)
plot(fit.ctree, main="Conditional Inference Tree")

ctree.pred <-predict(fit.ctree, df.validate, type="response")
ctree.perf<-table(df.validate$class, ctree.pred, dnn=c("Actual","Predicted"))
ctree.perf

#note that pruning isnt required for conditional inference trees, the process is somewhat more automated
#additionally, the party package has attractive plotting options

#6. displaying an rpart() tree with a ctree()-like graph
#dispaly the resulting tree using a plot in using the partykit package 
plot(as.party(an.rpart.tree))#to create the desired graph


#6. random forest
#ensemble learning approach to supervised learning
#multiple predictive models are developed, and the results are aggregated to improve classification rates

#comprehensive introduction to random forests, written by Leo Breiman and Adele Culter, at http://mng.bz/7Nul

#the algorithm for a random forest involves sampling cases and variables to create a large number of decision trees
#each case is classified by each decision tree
#the most common classification for that case is then used as the outcome

#assume that N is the number of cases in the training sample and M is the number of variables
#a. grow a large number of decision trees by sampling N cases with replacement from the training set
#b. sample m<M variables at each node. These variables are considered candidates for splitting in tha node. The value m is the same for each node.
#c. grow each tree fully without pruning(the minimum node size is set to 1)
#d. terminal nodes are assigned to a class based on the model of cases in that node
#e. classify new cases by sending them down all the trees and taking a vote-majority rules

#an out-of-bad OOB error estimate is obtained by classifying the cases that arent selected when building a tree. This is an advantage when a validation sample is unavailable
#random forests also provide a natural measure of variable importance

randomForest()#in the randomForest package, default number of trees is 500 and the default number of variables sampled at each node is sqrt(M), and the minimum node size is 1

library(randomForest)
set.seed(1234)
fit.forest <-randomForest(class~.,data=df.train,
                          na.action=na.roughfix,
                          importance=T)#grow 500 traditional decision trees by sampling 489 observations with replacement from the training sample and sampling 3 variables at each node of each tree, the na.action=na.roughfix option replaces missing values on numeric variables with column medians
fit.forest
importance(fit.forest, type=2)
forest.pred <-predict(fit.forest, df.validate)
forest.perf<-table(df.validate$class, forest.pred, dnn=c("Actual","Predicted"))
forest.perf
#random forest can prove a natural measure of variable importance, requested with the information=T option and printed with the importance() function
#the relative importancem easure specified by the type =2 option is the total decrease in node impurities(heterogeneity) from splitting on that variable, averaged over all trees
#Node impurity is measured with the Gini coefficient, sizeUniformity is the most important variable and mitosis the least important

#whereas the randomForest package provides forests based on traditional decision trees, the cforest() function in the party package can be used to generate random forests based o nconditional inference trees
#if predictor varibales are highly correlated, a random forest using conditional inference trees may provide better predictions

#random forests tent to be very accurate compared wit hother classfication methods. It can handle large problems and can handle large amounts of missing data in the training set and can handle cases in which the number of variables is much greater than the number of observations
#the provision of OOB error rates and measures of variable importance are also significant advantages

#the significant disadvantage is that it is difficult to understand the classification rules(there are 500 trees!) and communicate them to others
#additionally, we need to store the entire forest in order to classify new cases

#7. support vector machines
#support vector machines(SVMs) are a group of supervised machine-learning models that can be used for classification and regression
#sucess of them is in developing accurate prediction models, and in part because of the elegant mathematics that underlie the approach

#SVMs seek an optimal hyperplane for seperating two classes in a multidimensional space, hyperplane is chosen to maximize the margin between the two classes' closest points
#the points on the boundary of the margin are called support vectors(thet help define the margin), and the middle of the margin is the separating hyperplane

#for an N-dimensional space(that is, with N predictor variables), the optimal hyperplance(also called alinear decision surface) has N-1 dimensions, if there are two variables, the surface is a line, for three variables the surface is a plane, for 10 variables, the surface is a 9-dimensional hyperplane

#the optimal hyperplane is identified using quadratic programming to optimize the margin under the constraint that the data points on one side have an outcome value of +1 and the data on the other side has an outcome value of -1
#if the points are "almost" separable(not all the points are on one side or the other), a penalizing term is added to the optimization in order to account for errors and "soft" margins are produced

#but the data may be fundamentally nonlinjear. SVMs use kernel functions to transform the data into higher dimensions, in the hope that they will become more linearly separable
#one way to do this is to transform the two dimensional data into three dimensions using
(X,Y)->(X^2,2XY,Y^2)->(Z1,Z2,Z2)
#then we can separate the triangles from the circles using a rigid sheet of paper(that is, a two dimensional plane in what is now a three-dimensional space)

#the mathematics of SVMs is complex, look at Statnikov, Aliferis, Hardlin, & Guyon(2011) offering a lucid and intuitive presentation of SVMs that goes into quite a bit of conceptual detail without getting bogged down in higher math

#SVMs are available in two packages
ksvm()#in the kernlab package, this one is powerful
svm()#in the e1071 package, this one is easier to use

library(e1071)
set.seed(1234)
fit.svm <-svm(class~., data=df.train)
fit.svm
svm.pred <- predict(fit.svm, na.omit(df.validate))
svm.perf <-table(na.omit(df.validate)$class,
                 svm.pred, dnn=c("Actual","Predicted"))
svm.perf

#predictor variables with larger variances typically have a greater influence on the development of SVMs
#svm() function scales each varaible to a mean of 0 and sd of 1 before fitting the model by default

#we can see the predictive accuracy is good but not quite as good as that found for the random forest approach 
#unlike the random forest approach, the SVM is also unable to accommodate missing predictor values when classifying new cases

#8. tuning an SVM
#by default, the svm() function uses a radial basis function(RBF) to map samples into a higher-dimensional space(the kernel trick)
#the RBF kernel is often a good choice because it is a nonliear mapping that can handle relations between class labels and predictors that are nonlinear

#when fitting an SVM with the RBF kernel, two parameters can affect the results: gamma and cost.
#gamma is a kernel parameter that controls the shape of the separating hyperplane.
#larger values of gamma typically result in a larger number of support vectors
#gamma can also be thought of as a parameter that that controls how widely a training sample "reaches," with larger values meaning far and smaller values meaning close
#gamma must be greater than zero

#the cost parameter represents the cost of making errors
#a large value severely penalizes erros and leads to a more complex classification boundary
#there will be less misclassificatins in the training sample, but overfitting may result in poor predictive ability in new samples
#smaller values lead to a flatter classification boundary but may result in underfitting. Like gamma, cost is always positive

#by default, the svm() function sets gamma to 1/(number of predictors) and cost to 1
#but a different combination of gamma and cost may lead to a more effective model
#we can try fitting SVMs by varying parameter values one at a time, but a grid search is more efficient
#we can specify a range of values for each parameter using the tune.svm(), which fits every combination of values and reports on the performance of each

set.seed(1234)
tuned <-tune.svm(class~., data=df.train, #fit with an RBF kernel and varying values of gamma and cost
                 gamma=10^(-6:1),#eight values of gamma(ranging from 0.000001 to 10) and 21 values of cost
                 cost=10^(-10:10))
tuned#in all, 168 models(8*21) are fit and compared
fit.svm<-svm(class~.,data=df.train, gamma=.01, cost=1)
svm.pred<-predict(fit.svm, na.omit(df.validate))
svm.perf<-table(na.omit(df.validate)$class, svm.pred,dnn=c("Actual","Predicted"))
svm.perf

#the model with fewest 10 fold cross validated errors in the training sample has gamma=0.01 and cost=1
#by using these parameter values, a new SVM is fit to the training sample
#the model is then used to predict outcomes in the validation sample and the number of errors is dispalyed
#tunning the model decreased the nummber of errors slightly(from seven to six)

#in many cases, tunning the VM parameters will lead to greater gains
#SVM works well in many situations and can handle situations in which the number of variables is much larger than the number of observations
#one drawback of SVMs is that, like random forests, the resulting classification rules are difficult to understand and communicate
#they are esentially a black box.
#additionally, SVMs dont scale as well as random forests when building models from large training samples.
#but once a successful model is built, classifying new observations does scale well

#9. choosing a best predictive solution
#accurancy: how often the classifier is correct, it is not sufficient
#additional informmation needed to evaluate the utility of a classification scheme

#measure of predictive accuracy

#sensitivity: probability of getting a positive classification when the true outcome is positive(also called true positive rate or recall)
#specificity: probability of getting a negative classification when the true outcome is negative(also called true negative rate) 
#positive predictive value: probability that an observation with a positive classification is correctly identifeid as positive(also called precision)
#nagative predictive value: probability that an observation with a negative classification is correctly identified as negative
#accuracy: proportion of observations correctly identified(also called ACC)

performance <- function(table, n=2){
  if(!all(dim(table)==c(2,2)))
    stop("Must be a 2 x 2 table")
  tn=table[1,1]
  fp=table[1,2]
  fn=table[2,1]
  tp=table[2,2]
  sensitivity=tp/(tp+fn)
  specificity=tn/(tn+fp)
  ppp=tp/(tp+fp)
  npp=tn/(tn+fn)
  hitrate=(tp+tn)/(tp+tn+fp+fn)
  result <-paste("Sensitivity =", round(sensitivity,n),
                 "\nSpecificity =", round(specificity,n),
                 "\nPositive Predictive Value = ", round(ppp,n),
                 "\nNegative Predictive Value = ", round(npp, n),
                 "\nAccuracy = ", round(hitrate,n),"\n", se="")
  cat(result)}

#performance() function takes a table containing the true outcome(rows) and predicted outcome(columns) and returns the five accuracy measures

#the number of true negaties, false positives, false negatives and true positives are extracted
#then these counts are used to calculate the sensitivity, specificity, positive and negative predictive values and accuracy

#performance of breast cancer data classifiers
performance(logit.perf)

performance(dtree.perf)

performance(ctree.perf)

performance(forest.perf)

performance(svm.perf)

#we can often improve a classification system by trading specificity for sensitivity and vice versa

#in the logistic regression model, predict() was used to estimate the probability that a case belonged in the malignant group
#if the probability was greater than .5, the case was assigned to that group
#the .5 value is called the threshold or cutoff value
#if we vary this threshold, we can increase the sensiticity of the classficiation model at the expense of its specificity 

#predict() can generate prob for decision trees, random forests and SMs as well

#the impact of varying the threshold value is typically assessed using a receiver operating characteristic(ROC) curve
#ROC curve plots sensitivity versus specificity for a range of threshold values
#we can then select a threshold with the best balance of sensitivity and specificity for a given problem

#ROCR and pROC package generate ROC curves
#learn more, see Kuhn&Johnson(2013)
#advanced discussion is offered by Fawcett(2005)

#10. using the rattle package for data mining

#Raeele R anlaytic Tool to Learn Easily offers a GUI for data mining in R
#been using as well as other unsupervised and supervised data models not covered here

install.packages("rattle")
library(rattle)

#OS-specific installation directions and troubleshooting suggestions offered at http://rattle.togaware.com

library(rattle)
rattle()

loc<-"http://archive.ics.uci.edu/ml/machine-learning-databases/"
ds<-"pima-indians-diabetes/pima-indians-diabetes.data"
url<-paste(loc,ds,sep="")
#diabetes<-read.table(url,sep=",",header=F)
#the source above is not available

diabetes<-read.table('C:/Users/Urchin/Downloads/diabetes.csv', sep=",", header=F)
names(diabetes)<-c("npregant","plasma","bp","triceps",
                   "insulin","bmi","pedigree","age","class")
diabetes$class <-factor(diabetes$class, levels=c(0,1),labels=c("normal","diabetic"))
library(rattle)
rattle()
install.packages("RGtk2")

