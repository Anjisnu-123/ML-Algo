## Dimensionality reduction
Introduction : Defination,why dimensionality is reduction important: Handling high dimensional data,improves performance,avoids overfitting,enable visualization,enhanced interpretability,
Intution behind dimensionality reduction,common assumtions behind dimensionality reduction : data redundancy,lower intrinsic dimensionality,manifold assumption,noise assumption

Evaluation metrics for dimensionality reduction techniques : Explained varriance ratio:Description,calculation,interpretation,Accuracy: description,calculation,interpretation,Recall : description
calcl: (tp/tp+fn),interpretation,Precision : description,calculation: tp/(tp+fp),interpretation,F1 score : description,calculation: 1*((precision*recalll)/precision+recall),interpretation,
Silhouttte score: escription,calculation,interpretation,KL divergence: description,calculation,interpretation, Cross-entropy loss : Description,calculation,interpretation,stress function:
description,calculation,interpretation,Recostruction error: description,calculation,interpretation

PCA: Defination,Assumptions,Mathematical formulation: data standardization,covariance matrix computation,eigen value value and eigen vector decomposition,selcting pricncipal components,
project data onto principal components,varriance explained|, How to use,when to use,pros and cons,use case,example,intution/visualization,Comparison,evaluation metrics,Complexity,real
world challenge,variance and extensions,limitations in real world

LDA: Defination,Assumptions,Mathematical formulation: compute class means,compute within class scatter metric,compute b/w class scatter  matrix,solve the generalised eigen value problem,
select top discriminants,project data.How to use,when to use,pros and cons,example,intution/visualization,comparison,evaluation metrics,complexity,real world challenges,code implement,
varriants and extensions,limitations in real world apllications

SVD: Defination,assumptions,mathematical formulation,how to use,when to use,pros and cons,USe CASE,Example,intution and visualization,comparison,svd vs eigen values,Evaluation metrics,
complexity,Real world challenges,software implimentation,variants and extension: truncated svd,incremental svd,limitations in real world challenges,
