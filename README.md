# LinearREG

Este repositorio presenta 5 tópicos frecuentes en la literatura de Ciencia de Datos; 1) Regresion Lineal, 2) Regresion binária, 3) Redución de dimensionalidad, 4) Procesamiento de Languaje Natural, y 5) Procesamiento de imagen. 

# Regresión Lineal
El ejemplo adoptado fue obtenido del sitio web KAGGLE, que son datos de predicción de ventas según la inversion monetária en TV, Radio, Social Media, y Influencers.

$Y \in \mathcal{R}, Y \sim f(Xs)$

donde la función f(.) será la identidad, luego

$\mathcal{E}[Y|X_1,\cdots,X_k]=\beta_0+\beta_1 X_1 + \cdots + \beta_k X_k$

# Regresión Binária
El ejemplo adoptado fue obtenido del repositorio de la Universidad de California, donde se desea desarollar un modelo probabilístico que relacione con la caracterización de la Diabetes, en la población originaria Pima. 

$Y \in \{ 0,1 \}, Y \sim f(Xs)$

donde la función f(.) será la logit, $log(\frac{p}{(1-p)})$, luego

$\mathcal{E}[Y|X_1,\cdots,X_k]= \mathbb{P}(Y=1|X_1,\cdots,X_k) =p=\frac{1}{1+exp(\beta_0+\beta_1 X_1 + \cdots + \beta_k X_k)}$

# Principal Component Analysis (PCA)
Los Componentes Principales (PC) son una combinación lineal de todas las variables, clasificadas por orden decreciente de importancia y obtenidas a partir de la descomposición de la explicación de la varianza total.
