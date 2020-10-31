#' @export
krls_basic <- function(y_init, X_init, b = NULL, lambda = NULL, 
                 lambdaset = 10^(seq(-4, 1, length.out = 6)), 
                 folds = 5, scaling = T) {
    if (scaling){
        X <- scale(X_init)
        y <- scale(y_init)
    }else{
        X <- X_init
        y <- y_init
    }
    
    if (is.null(b)){
        b = ncol(X)
    }
    K <-  kernel_parallel(X, b)
    
    if (is.null(lambda)){ #select lambda with cross validation
        # assign label to each observation
        labels <- sample(folds, nrow(X), replace = T)
        cv_MSE <- sapply(lambdaset, function(lambda){
            MSE_vali <- sapply(1:folds, function(v){
                cv_idx <- which(labels != v)
                # validation MSE
                ch_vali <- krls_solver(y[cv_idx], K[cv_idx, cv_idx], lambda)$ch
                yh_vali <- mult(K[-cv_idx, cv_idx, drop = F], as.matrix(ch_vali))
                return(mean((yh_vali- y[-cv_idx])^2))
            })
            # average validation MSE
            return(mean(MSE_vali))
        })
        # select optimal lambda
        lambda <- lambdaset[which.min(cv_MSE)]
    }
    
    fit <- krls_solver(y, K, lambda)
    ch <- fit$ch
    if(scaling){
        yh <- fit$yh*attributes(y)$`scaled:scale`+
            attributes(y)$`scaled:center`
    }else{
        yh <- fit$yh
    }
    
    e <- y - yh
    mse <- mean(e^2) 
    mod <- list(X_scaled=X, y_scaled = y, ch=ch, yh=yh, e=e, mse=mse, b = b, lambda = lambda)
    class(mod) <- c("krls_basic","list")
    mod
}

predict.krls_basic <- function(mod, Xnew=NA, dh = F, scaling = T) {
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


# Adaptive selection ------------------------------------------------------

selection <- function(y, X, b, I,
                      lambda,
                      lambdaset, folds, 
                      m0, r0, p0, a0) {
    n <- nrow(X)
    if (m0 >= n) {
        I = 1:n
    }
    mse_tst <- matrix(0,r0,n-m0)
    I_rec <- matrix(0,r0,m0)
    
    if (is.null(I)){# when columns are not given
        if(is.null(lambda)){
            # get lambda based on the first batch
            I <- sample(n,m0)
            mod <- krls_basic(y[I], X[I , , drop=F], b= b, lambda = lambda,  
                        lambdaset = lambdaset, 
                        folds = folds, scaling = F)
            lambda <- mod$lambda
            yh_tst <- predict.krls_basic(mod, X[-I,,drop=F], scaling = F)
            mse_tst[1, ] <- (y[-I] - yh_tst)^2
            I_rec[1, ] <- I
            if (r0 > 1){
                for (t in 2:r0){
                    I <- sample(n,m0)
                    # fits full krls
                    mod <- krls_basic(y[I], X[I , , drop=F], b= b, lambda = lambda,  
                                lambdaset = lambdaset, 
                                folds = folds, scaling = F)  
                    yh_tst <- predict.krls_basic(mod, X[-I,,drop=F], scaling = F)
                    mse_tst[t, ] <- (y[-I] - yh_tst)^2
                    I_rec[t, ] <- I
                }
            }
        }else{ # use the given lambda
            for (t in 1:r0){
                I <- sample(n,m0)
                # fits full krls without scaling again in the function
                mod <- krls_basic(y[I], X[I , , drop=F], b= b, lambda = lambda,  
                     lambdaset = lambdaset, 
                     folds = folds, scaling = F)
                yh_tst <- predict.krls_basic(mod, X[-I,,drop=F], scaling = F)
                mse_tst[t, ] <- (y[-I] - yh_tst)^2
                I_rec[t, ] <- I
            }
        }
        
        pred_mse_vec <- rowMeans(mse_tst)
        i0 <- which.min( pred_mse_vec )
        I <- I_rec[i0,]
        press <- mse_tst[i0,]
        all_idx <- 1:n
        # select points with high MSE
        I1 <- all_idx[-I][press > p0*max(press)]
        # randomly select from the rest of points, total length should be a0 times larger
        I2 <- sample(all_idx[-c(I1, I)], size = round(length(I1)*a0))
        I <- c(I, I1, I2)
    }else{# when columns are given, return lambda from the full KRLS.
        if (is.null(lambda)){
            # I0 <- sample(I,m0)
            #mod <- krls(y[I0], X[I0,,drop=F], b= b, scaling = F)
            mod <- krls_basic(y[I], X[I , , drop=F], b= b, lambda = lambda,  
                        lambdaset = lambdaset, 
                        folds = folds, scaling = F)
            lambda <- mod$lambda
        }
    }
    
    #lambda <- lambda*n/m_0 # scale lambda with n?
    result <- list(I= I, lambda = lambda)
}
