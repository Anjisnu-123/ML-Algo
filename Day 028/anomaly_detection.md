## Introduction to anomaly detection : isolation forst,one class svm,lof
Defination,why is anomaly detection important,intution behind anomaly detection,common assunptions: rarity of anomalies,separation form normal data,content specific anomaly,, Types of
anomaly : Point anomalies,contextual anomalies,collective anomalies, How anomaly detection helps,when to use anomaly detection,
Evaluation metrics used in anomaly detection algorithms : Anomaly score : defination,calculation,interpretation. ROC & auc curve: defination,calculation,interpretation,precision ,recall,f1
score,silhouette score: def ,internpretation

### Isolation forest : 
Defination,Assumptions,mathematical formulation,step by step process,anomaly score,how to use,when to use,pros and cons,use case,code,variants and extensions,limitations in real world applications

### one class SVM : 
defination,assumptions,mathematical formulation,How to use: initialize the svm,make prediction,when to use,pros and cons,use case: network inrusion detection,manufacturing quality control.
intution/visualization,comparison,evaluation metrics,complexity,real world challenges,code implementation,variants and extension,limitations in real world applications.

### Local outliar factor :
defination,assumptions: local density variation,neighbourhood,density reachability |,mathematical formulation : knn,reachability distance,LRD,LOF |,how to use,when to use,pros and cons,
usecase , **This is robust for noise** ,**Used in time series** , intution/visualization,comparison,lof vs dbsca, evaluation metrics,complexity,real world challenges.code implementation,varriants
and extensions : weighted lof,fast lof,loci,limitations in real world applications

### Robust prinicipal component analysis (RPCA)
 defination,assumptions : low bank structure,spareity of outliers,linear relationship |,mathematical formulation,how to use,when to use,pros and cons,use case |,example of rpca,step by step
 calculation,intution/visualization,comparison,evaluation metrics,complexity,real world challenges: parameter sensitivity,scalability  issues,implementation complexity.,code implemenattion,
 variants and extensions,limitations in real world challenges


 # Association rule mining :
 Introduction to association rule mining : what is association rule mining,why is it needed,when to use it,how much does it help in unsupervised learning
 common evaluation metrics for association rule mining: support,confidence,lift,leverage,conviction,formula ,interpretation




 Association rule mining : apriori algo,eclat algo,fp growth

## Apriori algo : **Used for market basket analysis**. defination,assumptions,mathematical formulation : key concept : transaction database,itemset,support,confidence,lift,algorithm steps,steps 2: generate association rules,how to use,when to use,pros and cons,use case,example,comparison,evaluation metrics,complexity,limitstions,optimization,applications,variants

## Eclat algorithm
Defination,assumptions: binary transaction,vertical layout,frequent itemsets,mathematical  formula,algorithm steps,when to use ,how to use,pros and cons,mathematical example,comparison,
evaluation metrics,complexity,limitations,real world challenges,software implementation,use cases

## Fp growth

Defination,problem solved,Assumptions: binary transactions,minimum suport threshould,mathematical formulation
