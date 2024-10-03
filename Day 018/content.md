# Topic 
Introduction to machine learning,
What is machine learning : Defination of ML,importance of ML

| Aspect                | Classification                | Description                                            | Example                        |
|-----------------------|-------------------------------|--------------------------------------------------------|--------------------------------|
| Learning Type         | Supervised                    | Learns from labeled data                               | Regression, Classification     |
|                       | Unsupervised                  | Learns from unlabeled data                             | Clustering, Association        |
|                       | Reinforcement Learning        | Learns by interacting with the environment, receiving rewards/penalties | Game AI like AlphaGo |
| Unsupervised Learning  | Classification                | N/A                                                    | N/A                            |
|                       | Regression                    | N/A                                                    | N/A                            |
| Training Approach      | Batch Learning                | Trains on the entire dataset at once                  | Traditional training methods    |
|                       | Online Learning               | Trains incrementally as new data arrives              | Real-time prediction systems    |
| Algorithm             | Instance-Based Learning       | Learns from specific instances of the training data   | K-Nearest Neighbors            |
|                       | Model-Based Learning          | Learns a model based on the training data             | Decision Trees, Neural Networks|


Supervised learning : Defination,need,assumptions: - (Relationship b/w input and outpt remains stable over time,Training data is representative of real world data,the data contains no significant noise),importance,special characteristics,industry use cases

Unsupervised learning : Defination,need,assumptions: - (The data has underlying patterns or str that can be uncovered,Similarity within the data can help define clusters or assciatinons),importance,special characteristics,industry use cases

Reinforcement learning : Defination,need,assumptions: - (The env can provide feedback,the agent can learn a policy to maximize cumulartive rewards over time),importance,special characteristics,industry use cases(others and portfolio learning)

| Attribute                | Supervised Learning                        | Unsupervised Learning                          | Reinforcement Learning                           |
|--------------------------|-------------------------------------------|------------------------------------------------|-------------------------------------------------|
| Labeled Data             | Requires labeled data                     | Uses unlabeled data                            | Not explicitly labeled; feedback through rewards/penalties |
| Output                   | Predicts a specific output label         | Groups data or finds patterns without specific labels | Learns policies to maximize cumulative reward     |
| Learning Approach        | Learns from the entire dataset at once   | Learns patterns and structures in the data    | Learns through trial-and-error interactions with the environment |
| Use Case Complexity      | Generally less complex; clear objectives | Can be more complex; hidden structures        | Often very complex; requires balancing exploration and exploitation |
| Feedback Type            | Direct feedback (correct vs. incorrect)  | No direct feedback; relies on patterns         | Delayed feedback based on actions taken         |

Key concept of ML : Training vs Testing,feature labels,Model evaluation metrics supervised-unsupervised and reniforcement learning.

Supervised learning : simple linear reg,multiple linear regression,classifiation,model evaluation

Simple linear regression : def,objective,The linear equn,assumptions of LG,Estimating the parameters(OLS),step by step derivation of the OLS estimates
Evaluating the model : step 1 : Calculate the predicted ycap Values.calculate the residuals,calculate the sum of squared residuals,calculate R-squared(Coefficient of determination),Hypothesis testing(Significance of slope){Step 1: calcualte the standard error of slope,caculate t statistics},confidence intervals for the slope,prediction for new xvalues,Summary
model evaluation : model coefficients: intercept (beta0),slope(beta1),Predicted values- (predicted y for new x values),Model evaluation results: t stat,R sq value,confidence interval for slope.

verfiying assumptions:  linearity , independence of residuals (Durbin watson test),durbin watson stat,interpretation of durbin watson statistics(formula too),interpretation of drubin watson statistics.

verifying honscedasticity of residuals : the residuals should have constatnt variance across all levels of independent varriable.,steps to check homoscedasticity: (1.Mathematical Explaination,Breusch pagan test)
verifying normality of residuals: steps to check : 1.Mathematical expression, 2. statistical test for normality: shapiro wilk test,kolmogorov smirnov test
