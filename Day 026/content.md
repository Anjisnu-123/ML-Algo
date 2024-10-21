## Unsupervised learning: 
types comes under this

what is unsupervised learning,objective key goal,why is unsupervised learning important: common task: clustering,dimensionality reduction,anomaly detection,association rule mining,pattern recog,
An intution behind unsupervised learning,common assumptions in unsupervised learning: data distribution,feature independence,smoothness,clusterability,density
When to use unsupervised learning: when data unlabeled,for data exploration,to detect anomalies,recomendation system.
Application in real world and industry data
advantages and challenges
common evaluation metrics in unsupervised learning


### What is clustering
why is clustering needed: data understanding,segmentation,anomaly detection,pattern recog
How clusters helps : insight generation,simplification,feature engineering,personalization

## Evaluationmetrics used for clustering algo
(inertia (within cluster sumof square),silhouette score,davies bouldin index(low dbi value indicate vetter clustering,0 perfect clustering),Dunn index(Higher value if dunn shoe better clustering
showing well separated clusters and compact clusters,Adjusted rand index (ARI),AIC,BIC,Fuzzy partition coefficient,Fuzzy silhouette coefficient,cluster validity index),cophenetic corelation coefficient: corelation,interpretation,REachability plot): description,calculation,interpretation

understanding time complexity and space complexity


Clustering algoriths:  partition based clustering: ke means,k medoids(PAM),hierarchical clustering: allogomerative clustering,divisive clustering,density based clustering,model based clustering(GMM),fuzzy clustering: f,graph based clusterng


### Kmeans clustering

Defination,asssumptions,mathematical formulation,algorithm steops,how to use: all steps,when to use:EDA,market segmentation,image compression,data reduction,pros and cons,USecase,
example,intution/visualization,comparison,evaluation metrics,complexity,real world challenges,software implementation,variants and extension(k means++,mini batch k means),limitations in real world applications

### K-medoids (PAM)

defination,assumptions,mathematical formulation,how to use: set the number of clusters k,initialize medoids by selecting k random data points,assign points to the nearest medoid,swap medoid with other points to reduce cost,continue until convergence,When to use,pros and cons,usecase,example,intution/visualization,comparison(vs kmeans,hierarchchial clustering,dbscan),Evaluation metrics(sillhouettee score,dunn index,davies-bouldin index),complexity,Real world challenges,python implementation,Variants and extensions(CLARA,CLARANS),limitations in real world applications

