# Topic: 
Multiple linear regression : Defination,key concept

objectives of multiple linear regression : Modeling relationship,forecatsing new observations
Assumptions linear regression : Linearity,independence,homoscedasticity,normal in residuals,no perfect multi colienarity
The linear equation : ..
Derivation of OLS parameters : OLS details,The model setup,objective,minimize the ssr,solving for betai, assumptions of OLS

Derivation of GLS parameters: the model setup,objective,minimizing the GSSR,solving for beta1,assumptions of GLS:(Linearity parameters,covariance str omega,must be estimated,errors may have heteroscedasticity or auto corelation)

comparison of OLS and GLS

# OLS vs GLS Comparison

This document provides a comparison between Ordinary Least Squares (OLS) and Generalized Least Squares (GLS) regression techniques.

## Comparison Table

| Criterion                    | OLS                                                                  | GLS                                                             |
|------------------------------|----------------------------------------------------------------------|-----------------------------------------------------------------|
| Model                        | \( Y = X\beta + \epsilon \)                                         | \( Y = X\beta + \epsilon \), with non-constant error variance or correlated errors |
| Assumptions                  | Errors are homoscedastic and uncorrelated                           | Errors can be heteroscedastic and/or correlated                 |
| Error Term                   | \( \epsilon \sim N(0, \sigma^2 I) \)                               | \( \epsilon \sim N(0, \Sigma) \) where \(\Sigma\) is a known covariance matrix |
| Efficiency                    | Less efficient if errors are heteroscedastic or correlated          | More efficient under violations of OLS assumptions               |
| Objective                    | Minimize the sum of squared residuals                                | Minimize the weighted sum of squared residuals                  |
| Weightage of Errors          | All errors weighted equally                                          | Errors weighted inversely to their variance                      |
| Error Variance               | Constant variance (\(\sigma^2\))                                    | Non-constant variance based on error structure                   |
| Correlation of Errors        | Assumed to be zero                                                   | Can be present and accounted for                                |
| Covariance of Str            | Assumed to be zero                                                   | Can be non-zero, managed through the covariance structure       |
| Use Cases                    | Simple linear regression scenarios                                   | Situations with known heteroscedasticity or correlated errors   |
| Estimation Complexity         | Simpler and faster to compute                                        | More complex, requires estimating the covariance matrix         |


Modle evaluation : R2 : Defination,Formula,Interpretation |,Adjusted R2: defination,formula,Interpretation |, F statistics : Defination ,formula,interpretation |, Residual analysis : Purpose,methods: Residuals plots,Histogram of residuals,Q-Q plots,mean squared errors/Root mean sq error : (defination,formula,interpretation) |, AIC/BIC : Defination,Formulas:Aic,BIC,interpretation |. 


**Polynomial regression :** Defination of multivariate polynomial regression,mathematical formulation, assumptions of multivariate polynomial regression : (linearity in parameters,independence of errors,homoscedasticity,normality of residuals,no perfect multicolienarity),when to use multivraiate polynomial regression,parameter estimation in multivariate polynomial regression(step by step calc: define the objective function,partial dervatives,set derivatives to zero,solve),interpretation of coefficients





Verifying assumptions of multiple linear regression : Linearity , How to check ,mathematical approach |, Independence :  defination,How to check ,formula |,Homscedasticity : Defination,How to check,formula | Normality in residuals : defination,how to check,formula |, No perfect multicolinearity: defination,how to check:(vif),formula of VIF,interpretation.



## Overfitting and underfitting : 

what is overfitting,characteristics of overfitting ,why overfitting , example.

what is undertaking,charateristics of under fitting,why underfitting happens,example

## Bias varriance Trade-off
Bias:HIgh bias,Varriance: defination,explaination,high variance,
## the bias varriance trade off: 
combating overfitting: regularization,cross validation,reduce model complexity,more data early stopping
combating underfitting : Increase model complexity,feature enginerring,decrease regularization


| Test                     | Assumptions Checked                                            |
|--------------------------|---------------------------------------------------------------|
| **t-test**               | - Normally distributed populations                             |
|                          | - Homogeneity of variances (for independent t-test)           |
|                          | - Independent samples (for independent t-test)               |
| **ANOVA**                | - Normally distributed populations                             |
|                          | - Homogeneity of variances                                    |
|                          | - Independent samples                                         |
| **Chi-Square Test**      | - Observations are independent                                 |
|                          | - Categorical data                                           |
|                          | - Expected frequency in each category is at least 5          |
| **Pearson Correlation**   | - Linear relationship between variables                       |
|                          | - Normally distributed variables                              |
| **Spearman's Rank Correlation** | - Monotonic relationship between variables              |
|                          | - Ordinal or continuous data                                 |
| **Regression Analysis**   | - Linearity of relationships                                   |
|                          | - Independence of errors                                     |
|                          | - Homoscedasticity (constant variance of errors)            |
|                          | - Normality of errors                                        |
| **Mann-Whitney U Test**   | - Independent samples                                         |
|                          | - Ordinal or continuous data                                 |
| **Kruskal-Wallis Test**   | - Independent samples                                         |
|                          | - Ordinal or continuous data                                 |
| **Shapiro-Wilk Test**     | - Normality of data distribution                             |
| **Levene's Test**         | - Homogeneity of variances                                   |

| Distribution                | Associated Test                                   |
|-----------------------------|---------------------------------------------------|
| **Normal Distribution**     | - t-test                                         |
|                             | - ANOVA                                          |
|                             | - Pearson Correlation                             |
|                             | - Linear Regression                               |
|                             | - Shapiro-Wilk Test (for normality)             |
|                             | - Kolmogorov-Smirnov Test (for normality)       |
| **Binomial Distribution**   | - Chi-Square Test (Goodness of Fit)             |
|                             | - Binomial Test                                   |
|                             | - Fisher's Exact Test                            |
| **Poisson Distribution**    | - Poisson Test (Goodness of Fit)                 |
|                             | - Chi-Square Test (for count data)               |
| **Exponential Distribution**| - Kolmogorov-Smirnov Test (for exponentiality)  |
|                             | - Anderson-Darling Test (for exponentiality)     |
| **Uniform Distribution**    | - Chi-Square Test (Goodness of Fit)              |
| **Geometric Distribution**  | - Chi-Square Test (Goodness of Fit)              |
| **Log-Normal Distribution** | - Kolmogorov-Smirnov Test (for log-normality)    |
|                             | - Shapiro-Wilk Test (for log-normality)          |


# Regularization Technique in ML :
Defination,need for regularization,which and where to apply regularization

Avaialble regularization technique : L1 regularization(Lasso): Defination,mathematical derivation,effect,assumptions,use when you feel there exist some irrelevant column,L2 regularization (Ridge) : Defination,mathemathical derivation,effect,assumptions

Elastic net regularization : defination,mathematical derivation,effect
group lasso : defination,mathematical foormulation,effect.
Sparse group lasso : Defination,mathematical formulation ,effect.

**OPTIMIZATION :**
Gradient descent ; What is gradient descent?,Mathematical defination of gradient descent,steps in gradient descent(initialize the parameters,calculate gradient,updating parameters,converegence check):-> show all steps with example code
need for this hradient descent algorithm: FInding optimal parameter (OLS,GLS,Polynoial regression),Scalability of large datasets,flexibility, Handling complex loss functions,convergence control,implemnting regularization techniques

Variants of gradient descent: Batch gradient descent (defination,pros,cons,update rule),Stochastic gradient descent (SGD),Mini batch gradient descent (Defination,pros,cons,updated rule)

Learning rate and it simpact ; too small,Too large,optimal rate

convergence criteria : Cost function value,parameter change , maximum iterations

Stochastic gradient descent(SGD) : Defination ,Mathematical concept, steps,adbantage,disadvantage

Mini batch gradient descent : Defination ,Mathematical concept, steps,adbantage,disadvantage

Conjugate gradient method : Defination,Steps,Use cases,Advantages, disadvantages.


Quasi newtons methods (BFGS) : Defination,Steps,Use cases,Advantages, disadvantages.
> [!IMPORTANT]
> BFGS More used in deep neural network (Depth should be more than 10)

Levenberg marquardt algorithm (LM),L-BFGS



> [!NOTE]
> Ramsey reset test

> [!IMPORTANT]
> Logging in python
