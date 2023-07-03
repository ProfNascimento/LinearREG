#----------------------------------------------------------------------------------#
## https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data
# Data of TV, Influencer, Radio, and Social Media Ads budget to predict Sales
#----------------------------------------------------------------------------------#

## DATA IMPORT
HSS <- read.csv("https://raw.githubusercontent.com/ProfNascimento/LinearREG/main/Data_HSS.csv")
str(HSS)

summary(HSS)
unique(HSS$Influencer)

## VISUALIZATION 
require(psych)
pairs.panels(
  HSS[,c(1:3,5)],
  method = "pearson", # correlation method
  hist.col = "#00AFBB",
  density = TRUE,  # show density plots
  ellipses = TRUE) # show correlation ellipses

## VISUALIZATION NAs
Amelia::missmap(HSS)
dev.off()

## CORRELATION AMONG FEATURES
cor(na.omit(HSS[,c(1:3,5)]))

## MODELO CON INTERCEPTO
fit=lm(Sales ~ ., data=HSS)
summary(fit)

## DUMMY DATASET
DB=fastDummies::dummy_cols(HSS,select_columns = "Influencer",remove_selected_columns = TRUE)
head(DB)

## MODELO SIN INTERCEPTO
fit2=lm(Sales ~ 0+., data=DB)
summary(fit2)

## RESIDUO DE LOS MODELOS
hist(residuals(fit),xlab="Modelo 1")

hist(residuals(fit2),xlab="Modelo 2")
car::qqPlot(residuals(fit2))

## ANOTHER WAY TO GARANTEE GENERALIZATION (HOLD-OUT)
## [80% TRAIN - 20% TEST]

#------------------------------------------------#
## REMOVE MULTICOLLINEARITY
## APPLYING PCA
CleanSet=na.omit(HSS[,c(1:3,5)])

library(factoextra)
res.pca <- prcomp(CleanSet, scale = TRUE)
fviz_eig(res.pca, addlabels=TRUE)

# Eigenvalues
get_eigenvalue(res.pca)

## DESCRIBING THE RESPONSE VARIABLE (Y)
fviz_pca_biplot(res.pca, repel = TRUE, 
                select.ind = list(cos2 = 15), # Top 5
                col.var = "#2E9FDF", # Variables color
                col.ind = "#696969"  # Individuals color
)

## Selecting the Principal Components 1 & 2
res.pca$rotation[,1:2]
res.pca$x[,1:2]

fit_final=lm(CleanSet$Sales ~ 0+res.pca$x[,1:2])
summary(fit_final)
car::qqPlot(residuals(fit_final))
