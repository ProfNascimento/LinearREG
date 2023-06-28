library(jpeg)
library(gridExtra)
library(grid)

DN <- readJPEG("ME2.jpg", native = FALSE)
red <- DN[,,1]
green <- DN[,,2]
blue <- DN[,,3]

rotate <- function(x) t(apply(x, 2, rev))
# RED CHANNEL
image(rotate(red), col = grey.colors(300, 0, 1))

# DECOMPOSE 
r_svd <- svd(t(red) %*% red)
g_svd <- svd(t(green) %*% green)
b_svd <- svd(t(blue) %*% blue)

ncolunas <- c(3, 5, 25, 100, 200, 300) # níveis
# de compressão
for(i in ncolunas) {
  # (novas variaveis)
  r_projecao <- red %*% r_svd$u[, 1:i]
  # (reconstrucao)
  r_approx <- r_projecao %*% t(r_svd$u[, 1:i])
  g_projecao <- green %*% g_svd$u[, 1:i]
  g_approx <- g_projecao %*% t(g_svd$u[, 1:i])
  b_projecao <- blue %*% b_svd$u[, 1:i]
  b_approx <- b_projecao %*% t(b_svd$u[, 1:i])
  imagemFinal <- array(NA, dim = c(nrow(red), ncol(red), 3))
  imagemFinal[,,1] <- r_approx
  imagemFinal[,,2] <- g_approx
  imagemFinal[,,3] <- b_approx
  imagemFinal <- ifelse(imagemFinal < 0, 0,
                        ifelse(imagemFinal > 1, 1, imagemFinal))
  grid.raster(imagemFinal)
  cat("Numero de componentes =", i, "\n",
      "Economia de espaço = ",
      paste0(round(100 - (prod(dim(r_projecao)) +
                            prod(dim(r_svd$u[, 1:i])))/prod(dim(red))*
                     100, 2),"%"))
}