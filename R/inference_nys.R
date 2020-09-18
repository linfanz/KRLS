# variance of coefficient
#' @export
inference.krls_nys <- function(mod){
  M <- mod$M
  R <- mod$R
  sigmasq <- mod$mse
  X <- mod$X_scaled
  I <- mod$I
  m <- length(I)
  Minv <- solve(M) # could potentially be faster
  # variance of coefficient
  vco <- sigmasq* Minv %*% crossprod(R, R) %*% Minv
  # variance of fitted values
  vfitted <- R %*% vco %*% t(R)
  # # partial difference just try one column
  # Z <- matrix(nrow = n, ncol = m)
  # for (i in 1:n) {
  #   for (j in 1:m) {
  #     Z[i,j] <- X[I[j],1] - X[i,1] 
  #   }
  # }
  # nb <- n*mod$b
  # ZR <- Z*R # n*m
  # pd <- sum(ZR %*% mod$dh)/nb*2
  # vpd <- 4*sum(ZR %*% vco %*% t(ZR))/nb^2  
  
  # derivatives for 
  n <- nrow(X)
  nb <- n*mod$b
  p <- ncol(mod$X_scaled)
  # Z[[i]] n*p
  Z <- lapply(1:p, function(k){
    t(sapply(1:n, function(i){
       X[I,k] - X[i, k]
    }))
  })
  # PD <- matrix(nrow = n, ncol = p)
  # for (i in 1:n) {
  #   for (k in 1:p) {
  #     PD[i,k] <- sum(mod$dh * Z[[k]][i, ] * R[i, ])*2/mod$b 
  #   }
  # }
  # for (k in 1:p) {
  #   PD[,k] <- colSums((mod$dh * R) * Z[[k]])*2/mod$b 
  # }
  
  avePD <- c()
  avePD_var <- c()
  for (k in 1:p){
    ZR <- Z[[k]]*R
    avePD[k] <- sum(ZR %*% mod$dh)*2/nb
    avePD_var[k] <- 4*sum(ZR %*% vco %*% t(ZR))/nb^2
  }
  
  return(list(vco=vco, vfitted=vfitted, avePD=avePD, avePD_var = avePD_var))
}