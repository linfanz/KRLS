#' KRLS with Nystrom approximation
#' 
#' scaling up KRLS with Nystrom approximation
#' 
#' @param X_init explanatory variable, must be a matrix
#' @param y_init response variable, must be a matrix
#' @param I index set of selected data points from X_init, 
#' could be assigned or adaptively selected by the algorithm
#' @param lambda regularization parameter, could be assigned or adaptively selected by the algorithm
#' @param lambdaset if lambda not given, it will be selected from this candidate set
#' @param folds number of folds in cross validation to select hyperparameters
#' @param m0 number of data points randomly selected to perform the pilot testing, a full KRLS,
#'  in order to determine set I
#' @param r0 number of times for pilot testing, the one with the minimum MSE will 
#' be selected 
#' @param p0 a data point will be included in I if in the pilot testing it has squared residual
#' larger than p0*max(residuals)
#' @param a0 besides data points produce large residuals, we also randomly select from the rest of
#' data with size a0 times larger than those with large residuals
#' @param scaling indicate if data has been scaled or not, recommend to set to TRUE if prediction 
#' using the model is needed
#' 
#' @examples 
#'\dontrun{
#' fit_nys <- krls_nys(X, y, lambda = 1)
#' y_h <- predict.krls_nys(mod = fit_nys, Xnew=X_test)
#' inf_nys <- inference.krls_nys(fit_nys)
#' }
#' @export

krls_nys <- function(X_init, y_init, 
                     b = NULL, 
                     I = NULL, 
                     lambda = NULL,
                     lambdaset = 10^(seq(-4, 1, length.out = 6)), 
                     folds = 5, 
                     m0 = 100, r0 = 3, p0 = 0.8, a0 = 5, 
                     scaling = T){
    # scale input if needed
    if (scaling){
        X <- scale(X_init)
        y <- scale(y_init)
    }else{
        X <- X_init
        y <- y_init
    }
    
    # bandwidth is set to be the number of features by default
    if (is.null(b)){
        b = ncol(X)
    }
    
    # select columns and lambda
    if (is.null(I) | is.null(lambda)){
        select_result <- selection(y, X, b, I,
                                  lambda,
                                  lambdaset, folds, 
                                  m0, r0, p0, a0)
        I <- select_result$I
        lambda <- select_result$lambda
    }

    R <- kernel_parallel_2(X, X[I, , drop = F], b)
    D <- R[I, ]
    
    fit <- RD_solver(R, D, lambda, y)
    dh <- fit$dh 
    yh <- fit$yh
    M <- fit$M
    if(scaling){
        yh <- yh*attributes(y)$`scaled:scale`+
            attributes(y)$`scaled:center`
    }
    
    e <- y_init - yh
    mse <- mean(e^2)
 
    mod <- list(X_scaled=X, y_scaled=y, b=b, M= M, R = R,
                dh = dh, I =I, yh=yh, e=e, mse=mse, lambda = lambda,
                scaling = scaling)
    class(mod) <- c("krls_nys","list")
    mod
}