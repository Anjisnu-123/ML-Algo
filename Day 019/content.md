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



Verifying assumptions of multiple linear regression : Linearity , How to check ,mathematical approach |, Independence :  defination,How to check ,formula |,Homscedasticity : Defination,How to check,formula | Normality in residuals : defination,how to check,formula |, No perfect multicolinearity: defination,how to check:(vif),formula of VIF,interpretation.

> [!NOTE]
> Ramsey reset test
