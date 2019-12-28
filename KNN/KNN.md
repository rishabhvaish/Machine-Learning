KNN
================
Rishabh Vaish
11/10/2019

## Question 1 \[50 Points\] KNN

Write an R function to fit a KNN regression model. Complete the
following steps -

1.  \[15 Points\] Write a function myknn(xtest, xtrain, ytrain, k) that
    fits a KNN model that predict a target point or multiple target
    points xtest. Here xtrain is the training dataset covariate value,
    ytrain is the training data outcome, and k is the number of nearest
    neighbors. Use the \(l2\) norm to evaluate the distance between two
    points. Please note that you cannot use any additional R package
    within this function.

### Answer:

Following is the my\_knn function which takes (xtest, xtrain, ytrain, k)
as input in matrix form and returns the vector containing Y predicticted
(ypred).

``` r
myknn <- function(xtest, xtrain, ytrain, k)
{
  #Assuming Matrix inputs, checking dimensions
  xtest_dim<- dim(xtest)
  xtrain_dim <- dim(xtrain)
  ytrain_dim <- dim(ytrain)
  
  #Create the Ypred output matrix
  ypred <- matrix(data=NA,nrow=xtest_dim[1],ncol= 1)
  
  # Main function to calculate distances and get the top k ytrain
  for ( xtest_row in 1:xtest_dim[1] ){
    #Create the test row vector
    xtest_vector = xtest[xtest_row,]
    
    # To find distance step 1 - subtract xtest_vector from xtrain
    d1 <- sweep(xtrain,2,xtest_vector)
    # To find distance step 2 - sum the squares of rows and take square root
    d2 <- matrix(sqrt(rowSums( d1^2)), xtrain_dim[1],1)
    
    #Combine Ytrain with Distance matrix
    dist_ytrain <- cbind(d2, ytrain)
    colnames(dist_ytrain) <- c('Distance','Ytrain')
    
    #Order by distance
    ordered_dist_ytrain <- dist_ytrain[order(dist_ytrain[,'Distance']),]
    
    # Set Ypred = Average of top K values of ordered_dist_ytrain
    ypred[xtest_row,] <- mean(ordered_dist_ytrain[1:k,'Ytrain'])
  }
  
  return(ypred)
}
```

2.  \[10 Points\] Generate 1000 observations from a five-dimensional
    normally distribution: \[N (\mu, \Sigma_{5x5})\] where
    \(\mu = (1, 2, 3, 4, 5)^T\) and \(\Sigma_{5x5}\) is an
    autoregressive covariance matrix, with the \((i, j)^{th}\) entry
    equal to \(0.5^ {\mid i-j \mid }\). Then, generate outcome values Y
    based on the linear model
    \[ Y = X_1 + X_2 + (X_3 - 2.5)^2 + \epsilon\] where \(\epsilon\)
    follows i.i.d. standard normal distribution. Use set.seed(1) right
    before you generate this entire data. Print the first 3 entries of
    your data.

### Answer:

Using the nvrnorm function in MASS library to generate five-dimensional
normally distributed X. I have used rnorm function to set \(\epsilon\)
in true function.

``` r
library(MASS)
#Set mean
mu <- c(1, 2, 3, 4, 5)
#Create variance matrix 
sigma <- matrix(0,5,5)
for (i in 1:5){
  for (j in 1:5){
    sigma[i,j] = 0.5 ^ (abs(i-j))
  }
}

#Set seed and generate data
set.seed(1)
x <- mvrnorm(n = 1000, mu, sigma)
y <- matrix(x[,1] + x[,2] + (x[,3] - 2.5)^2 + rnorm(length(x)), nrow(x), 1 )

#Combine data and set variable names
data <- cbind(x,y)
colnames(data) <- c( "x1", "x2", "x3", "x4", "x5", "y")
head(data,3)
```

    ##              x1       x2       x3       x4       x5        y
    ## [1,]  2.0770490 3.555163 2.641969 3.902436 5.108741 4.135994
    ## [2,]  2.4780195 2.161175 2.188487 2.796376 5.330744 5.365376
    ## [3,] -0.1413538 2.630428 4.666608 4.493909 5.698190 5.505069

3.  \[10 Points\] Use the first 400 observations of your data as the
    training data and the rest as testing data. Predict the Y values
    using your KNN function with k = 5. Evaluate the prediction accuracy
    using mean squared error
    \[\frac{1}{N}\sum_{i}\left ( y_i - \hat{y}_i\right )^2\]

### Answer:

Using first 400 rows as training data and testing on the next 600 rows
yields an MSE = 2.19

``` r
#setting training and testing data
xtrain <- x[1:400,]
xtest <- x[401:1000,]
ytrain <- y[1:400]
ytest <- y[401:1000]
#Using K = 5
k = 5

#Using the function 
ypred <- myknn(xtest, xtrain, ytrain, 5)

#Computing error
knn_error <- colSums((ytest - ypred)^2)/length(ytest)
#MSE with K=5
knn_error
```

    ## [1] 2.191387

4.  \[15 Points\] Compare the prediction error of a linear model with
    your KNN model. Consider k being 1, 2, 3, . . ., 9, 10, 15, 20, . .
    ., 95, 100. Demonstrate all results in a single, easily
    interpretable figure with proper legends

### Answer:

Created a matrix (knn\_errorMatrix) to store MSE while varying K from 1
to 100.

``` r
#Define the max K value
k_range <- 100
# Create an error matrix
knn_errorMatrix = matrix(NA, k_range, 2)
colnames(knn_errorMatrix) <- c("k", "MSE")

#Looping through each K and storing the MSE
for (k in 1:k_range)
{
  ypred <- myknn(xtest, xtrain, ytrain, k)
  knn_errorMatrix[k, 1] <- k
  knn_errorMatrix[k, 2] <- colSums((ytest - ypred)^2)/length(ytest)
}
#Sample of matrix
head(knn_errorMatrix)
```

    ##      k      MSE
    ## [1,] 1 3.179724
    ## [2,] 2 2.423957
    ## [3,] 3 2.273419
    ## [4,] 4 2.252683
    ## [5,] 5 2.191387
    ## [6,] 6 2.171828

I have used the lm function in R to fit a linear model for comparison
with KNN. It yields an MSE = 3.24

``` r
#creating testing and training dataframe for fitting lm model 
train_df <- data.frame(cbind(xtrain,ytrain))
colnames(train_df) <- c('x1','x2','x3','x4','x5', 'ytrain')
test_df <- data.frame(xtest)
colnames(test_df) <- c('x1','x2','x3','x4','x5')

#Fitting the lm model
linear_model = lm(ytrain ~ x1+x2+x3+x4+x5, data = train_df)
#Predicting 
ypred_lm <- matrix(predict(linear_model, data.frame(test_df)), length(ytest), 1)

#Calculating the MSE
error_lm <- colSums((ytest - ypred_lm)^2)/length(ytest)
error_lm
```

    ## [1] 3.247147

Making a figure to show the variation of MSE with K for KNN and
comparing it with the MSE of simple linear regression model (lm)

``` r
#Plotting the figure and adding the legend
plot( 1:k_range, knn_errorMatrix[ , 2], pch = 19, cex = 0.4, xlab = "K", ylab = "MSE")
lines(1:k_range, knn_errorMatrix[ , 2], type = "s", col = "darkorange", lwd = 1.5)
lines(1:k_range, rep(error_lm, k_range), type = "s", col = "deepskyblue", lwd = 1.5)
title(main="MSE Comparison (KNN vs Linear regression)")
legend("topleft", legend=c("KNN ( K in 1:100 )", "Linear Regression"),
       col=c("darkorange", "deepskyblue"), lty=1, cex=0.8)
```

![](KNN_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

## Question 2 \[50 Points\] Linear Regression through Optimization

Linear regression is most popular statistical model, and the core
technique for solving a linear regression is simply inverting a matrix:

\[\hat\beta = \left ( X^TX \right )^{-1}X^Ty\]

However, lets consider alternative approaches to solve linear regression
through optimization. We use a gradient descent approach. We know that
\(\hat\beta\) can also be expressed as
\[\hat\beta = arg min\ l(\beta) = argmin\ \frac{1}{2n} \sum_{i}^{2n}\left ( y_i - x_i^T\beta \right ) x_i\]

To perform the optimization, we will first set an initial beta value,
say \(\beta\) = 0 for all entries, then proceed with the updating
\[ \beta^{new} = \beta^{old} + \frac{\partial l(\beta)}{\partial \beta} \times \delta\]

where \(\delta\) is some small constant, say 0.1. We will keep updating
the beta values by setting \(\beta^{new}\) as the old value and
calcuting a new one untill the difference between \(\beta^{new}\) and
\(\beta^{old}\) is less than a prespecified threshold \(\epsilon\),
e.g., \(\epsilon = 10^{-6}\). You should also set a maximum number of
iterations to prevent excessively long runing time.

1.  \[35 Points\] Based on this description, write your own R function
    mylm\_g(x, y, delta, epsilon, maxitr) to implement this optimization
    version of linear regression. The output of this function should be
    a vector of the estimated beta value

### Answer:

Created a mylm\_g function in R which takes (x, y, delta, epsilon,
maxitr) as inputs in matrix form of following dimensions - x is \[n,p\]
y is \[n,1\]

The function implements gradient descent approach for optimisation and
outputs the beta estimates in matrix of \[1,p\]

``` r
mylm_g <- function(x, y, delta, epsilon, maxitr){
  #beta matrix
  beta <- matrix(0,1,ncol(x))
  for(i in 1:maxitr){
    #gradient 
    gradient <- (-1/nrow(x))*colSums((y - beta%*%t(x))%*%x)
    #calculating new beta
    beta_new <- beta - gradient*delta
    #checking distance between new and old beta
    beta_dist <- dist(rbind(beta, beta_new))
    #checking for threshold
    if (beta_dist < epsilon){
      break
    }
    #updating the beta values
    beta <- beta_new
  }  
  return(beta)
}
```

2.  \[15 Points\] Test this function on the Boston Housing data from the
    `mlbench` package. Documentation is provided
    [here](https://www.rdocumentation.org/packages/mlbench/versions/2.1-1/topics/BostonHousing)
    if you need a description of the data. We will remove `medv`, `town`
    and `tract` from the data and use `cmedv` as the outcome. We will
    use a scaled and centered version of the data for estimation. Please
    also note that in this case, you do not need the intercept term. And
    you should compare your result to the `lm()` function on the same
    data. Experiment on different `maxitr` values to obtain a good
    solution. However your function should not run more than a few
    seconds.

<!-- end list -->

``` r
  library(mlbench)
  data(BostonHousing2)
  X = BostonHousing2[, !(colnames(BostonHousing2) %in% c("medv", "town", "tract", "cmedv"))]
  X = data.matrix(X)
  X = scale(X)
  Y = as.vector(scale(BostonHousing2$cmedv))
```

### Answer:

Estimating beta using the function created above and comparing it with
the beta values generated by using lm function. The values are similar.
The difference between the 2 beta estimates is shown

``` r
  # Setting the parameters
  delta = 0.1
  epsilon = 10^-7
  maxitr = 5000
  
  #Estimating beta using the function
  beta <- mylm_g(X,Y,delta,epsilon,maxitr)
  
  #creating dataframe and fitting lm
  df <- data.frame(cbind(X,Y))
  linear_model = lm(Y ~ lon+lat+crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat, data = df)
  beta_lm <- matrix(linear_model$coefficients[2:length(linear_model$coefficients)], 1, ncol(X))
  
  #comparing the 2 beta values
  beta_compare <- cbind(t(beta),t(beta_lm),t(abs(beta-beta_lm)))
  colnames(beta_compare) <- c("Beta", "Beta_LM", "Error")
  beta_compare
```

    ##               Beta      Beta_LM        Error
    ##  [1,] -0.032316756 -0.032316441 3.147665e-07
    ##  [2,]  0.030244764  0.030245087 3.228742e-07
    ##  [3,] -0.097935206 -0.097935969 7.635663e-07
    ##  [4,]  0.118271740  0.118273098 1.357273e-06
    ##  [5,]  0.011386300  0.011390378 4.078428e-06
    ##  [6,]  0.071312754  0.071312253 5.001286e-07
    ##  [7,] -0.199702924 -0.199703772 8.477011e-07
    ##  [8,]  0.287233466  0.287232811 6.555650e-07
    ##  [9,]  0.007564405  0.007564852 4.470164e-07
    ## [10,] -0.321039510 -0.321039342 1.676267e-07
    ## [11,]  0.290840609  0.290850755 1.014614e-05
    ## [12,] -0.236514862 -0.236526155 1.129274e-05
    ## [13,] -0.206804452 -0.206804965 5.135301e-07
    ## [14,]  0.091235358  0.091235409 5.052057e-08
    ## [15,] -0.417972446 -0.417972819 3.727051e-07
