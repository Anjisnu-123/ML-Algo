Linear support vector machines (SVM)
------------------------------------
Used for classification tasks. They work by finding the optimal hyperplane that separates different classes in the feature space. 
wT*x +b=0
The margin is defined as the distance between the decision boundary (hyperplane) and the nearest data points from either class. 
importance : maximizing the margin and robustness to noise
Objective function, optimization process, regularization, statistica interpretation, evaluation metrics
when to choose linear SVM : margin and decision boundary: (svm,logostic regression),Nature of data : (svm,logostic regression),underlying mathematical basis :(svm,logostic regression),overfitting and regularization : (svm,logostic regression), interpretibility : (svm,logostic regression),performance with imbalanced data: (svm,logostic regression).



Topic : K-nearest neighbours (KNN) --> introduction,key idea,choosing k (small value of overfitting,large value underfitting)
------------------------------------
How knn works : choose value of k,calc dist b/w the new data point and all other training data points,identify k nearest  neighbour in new data point,for classification assins the class that is most common among k nwighbour,
for regression,take the average of the values of k neares neighbours to predict the value
Distance metrics : Eucledian dist,manhatten dist,minkwski dist,hamming dist
Knn for classification : it is often used for cl, decision ruie : count the freq of each class label among the k nearest neighbours,Assign the class label with highest freq.
knn for regression,
pros and cons of knn : (pros: simple and intutive,no training phase,adaptive to multi-class classification), (cons : computationally expensive,sensitive to the choice of k,sensitive to irrelevant features 
and the scale of data)
knn feature scaling : minmax scaling,standardization (z-score)
Evaluation metrics: for classification : accuracy,precion,recall,f1 score,conf matrix,roc-auc curves, for regression : MAE,MSE,R-squared(also known as coefficient of determination)
K-Nearest neighbour(KNN) : Optimization of KNN,use cases of KNN : image recog,recommendation system,medical diagonisis,example scenario



Non linear SVm with kernels : description about svm
----------------------------
key concept in SVM : Hyperplane,support vectors,margin,kernel,non linear decision boundaries,regularization parameter
how non linear SVM works: Basic steps
kernel in non linear svm : kernel function,common kernel function : polynomial kernel,Radial basis function (RBF),example : suitable for what!!,sigmoid kernel,example : suitable for what!!
mathematical formulation,Decision boundary in non linear SVM : diagram of decision boundary : 1. non separable data in 2d,mapping to higher dimension
hyperparameter in Non linear SVM : regularization parameter (C) ,kernel parameters
pros and cons of non linear SVM: Pros: effective in high dimension,works well with small datasets,clear margin of separation, Cons : computational complexity,choice of kernel,not ideal for noisy data
use case of non linear SVM : image classification,text catagorization,bioinformetics
Example walkthrough


Decisions Trees : Structure of decision tree: Root node,internal nodes,branches,Leaf nodes |,Recursive partioning: (At each step building the tree :Select feature,Threshold for splitting,
----------------
Recursive splitting,stopping criterion).
Mathematical formulation of splitting criteria: how do we choose the best split at each node?- ginni impurity(for classification trees),Entropy (information gain for,classification trees),
Mean squared error (for regression trees).
decision boundary,Tree pruning: pre pruning,post pruning,advantages of decision trees : interpretability,non parametric,handles both numerical and catagorical, fast inference
Limitations : overfitting ,instability,biased towards  dominant features
Regularization in decision trees : max depth,min samples per leaf,max leaf nodes


Naive bayes : introduction to naive bayes,Key characteristics : probabilistic nature,indeoendence assumptions,Bayes theorem,Highly scalable,great for high dimensional data.bayes theorem (foundation of naive bayes)
----------- 
Tyeps of naive bayes classifiers : Gausian naive bayes,multinomial naive bayes,bernouli naive bayes.
gausian naive bayes : introduction , str of gausian naive bayes : prior,likelihood,posterior, Mathematical formulation: bayes theorem,gausian likelihood (for contd feature),metrics to calculate: prior probability,Mean and var for each feature within class,posterior probability
interpretation of metrics,decision boundaries in gaussian naive bayes,pros and cons : pros -> simple and fast,effective with small data,scalability,Handles continuous data,cons : indeoendece assumptions
sensitive to outliars, requires normal dist,Regularization : varriance smoothing,log transforming of feature. use cases and examples : Medical diagonisis,example,iris flower classification ,example
