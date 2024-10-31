# Topic

## Recommender system
---
What is recommender system,why are recommender system needed,Key concept in recommender system:User pref,item features,personalization,cold start problem:when there is insuff data about new users,user item matrix,industry use cases: Ecomerce platform,streaming service,social media platforms,online news and content provider,job portal,Metircs used in recommender system:
`[ MAE,RMSE,precision,recall,F1 score,Hit rate : (number of hits/Total number of users),interpretation,Diversity,serendipity: it measures how surprising or unexpected the recomendations are,Novelity,Mean resiprocal rank (MRR),normalized discounted cumulative gain,Personalization<logarithmic loss ]` : for alll this: [] calculation,interpretation

Collaborative filtering:user based,Item based,matrix factorization: singular value decomposistion,Alternative least square,content based filtering

User-based collaborative filtering : Defination,assumptions,mathematical formulation : {step 1: user item interaction matrix,step 2: computing user similarites,preprocess the data,filter for common items,caclualte cosine similarity},How to use,When to use,pros and cons,Use case,intution/visualization,comparison,evaluation metrics,complexity,real world challenges,code implementation

Item based collaborative filtering : Defination,Assumptions,Mathematical formula {step 1: user item interaction matrix,step 2: computing user similarites,step 3: prediction },How to use
When to use,pros and cons,use case, intution,comparison,Evaluation metrics,complexity,real world challeneges,code implementation,varriants and extension,Limitations in real world application

singular value decomposistion : Defination,assumption,mathematical formulation,how to use,when to use,pros and cons,use case,svd calculations,intution/visualization,comparison,evaluation metrics,complexity,real world challenges,spftware implementation,variants and extension,limitations in real world application

Alternative least sq method : Defination,Assumptions,Mathematical formulation: 1,objective function,2.alternating optimization,closed form solution,How to use,when to use,pros and cons,
usecase,complexity,real world challlenges,software implementation,variants and extensions,limitations in real world apllications

content based filtering : Defination,Assumptions,mathematical formulation and explanation,item profile,user profie,similarity calculation,update rule,when to use,pros and cons,use cases,mathematical example with detialed calc,evaluation metrics,complexity,real world challeneges,code,varriants and extension,challenges and limitations

---

## Graph based
----

Defination,Need for graph based learning,Graph representation : Nodes,edges,edge weight,directed vs undirected graph,Graph learning tasks: node classififcation,link prediction,graph clustering,graph prediction|,Types of graph based learning algorithm:Traditional algorithms,graph neural networks,graph representation learning. Industry use cases of graph based learning : social networks,Recomendation systems,molecular biology,fraud detection in financial networks,web search and ranking,knowledge graph,Traffic and transportation networks,supply chain optimization,evaluation metrics used in graph based learning-old accuracy,precision,recakl,f1,ruc-auc,avg shortest path length,Modularity,clustering coefficient,node degree distribution,betweenness centrality(Node importance)



Graph based learning techniques:  comunity detection in graph : (louvain method,girvan newman algorithm),page rank algorithm(for link analysis)

overview of community detection in graph : Introduction? why community detection,Louvain method : defination,assumptions,mathematical formulation,how to use where to use,when to use,pros and cons,usecase,Demo problem solved using louvain method,intution/visualization,comparison,evaluation metrics.complexity,rloeal world challenges,software implementation,variants and extension,Limitations in real world application


Girvan - newman algorithm : Defination,assumptions,Mathematical formulation,how to use,when to use,pros and cons,usecase,real world example,
example calculation of girvan newman algrithm,intution/visualization,comparison,evaluation metrics,complexity,real world hallenegs,software implementation,variants and extension,limitation in real world application

Pagerank algorithm : Defination,assumptions,Mathematical formulation,how to use,when to use,pros and cons,usecase,problem statement,intution/visualixation,comparison,evaluation metrics,complexity,real world challange,code implement,variants and extension,limitation in real world application

---

## Blind signal separation
---
Defination,Need of blind signal separation : Multichanel signal processing,Noise reduction ,Data analysis,key concept of BSS,Independence,non gausicanity,unsupervised learning,Industy use cases of blind signal separation: Tele communication,audio processing,biomedical engineering,remote sensing,Image processing,finance.

Evaluation metric for blind signal separation: SNR,kld,correlation coefficient,MAE,EVR,cumulative explained variance,reconstruction error,mean squared error,principal component corelation

Bilind signal separation techniques : ICA,PCA
**ICA :** Defination,assumptions: Statistical independence,non gausianity,Linear mixing,number of sources|,mathematical formulation,how to use,when to use,pros and cons,usecase,Example calc,intution/visualization,comparison,evaluation metrics,complexity,real world challenegs,code implementation,varinats and extension,limitations in real world applications
