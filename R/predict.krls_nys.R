#' @export
predict.krls_nys <- function(mod, Xnew=NA, dh = T, scaling = NULL) {
  if (is.null(scaling)) {
    scaling = mod$scaling
  }  
  if (any(is.na(Xnew))) {
    return(mod$yh)
  } else {
    if(scaling){
      X_test <- scale(Xnew,
                      center = attributes(mod$X_scaled)$`scaled:center`,
                      scale = attributes(mod$X_scaled)$`scaled:scale`)
      if (dh){
        yh <- mult(kernel_parallel_2(X_test, mod$X_scaled[mod$I, ], mod$b), mod$dh )*
          attributes(mod$y_scaled)$`scaled:scale`+
          attributes(mod$y_scaled)$`scaled:center`
      }else{
        yh <- mult(kernel_parallel_2(X_test, mod$X_scaled, mod$b), mod$ch )*
          attributes(mod$y_scaled)$`scaled:scale`+
          attributes(mod$y_scaled)$`scaled:center`
      }
    }else{
      X_test <- as.matrix(Xnew)
      if (dh){
        yh <- mult(kernel_parallel_2(X_test, mod$X_scaled[mod$I, ], mod$b), mod$dh)
      }else{
        yh <- mult(kernel_parallel_2(X_test, mod$X_scaled, mod$b), mod$ch)
      }
    }
    return(yh)
  }
}