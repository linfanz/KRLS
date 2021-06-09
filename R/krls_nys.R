#' KRLS with Nystrom approximation
#' 
#' scaling up KRLS with Nystrom approximation
#' 

#' @param X an \eqn{n \times p} matrix containing explanatory variables
#' @param y an \eqn{n \times 1} matrix containing response variable
#' @param b bandwidth of the kernel, set to be the number of columns in \code{X} if not given
#' @param I index set of selected data points from X, 
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
#' @return A list of result
#' \item{X_scaled}{output of \code{scale(X)} when \code{scaling = T}; 
#' same as \code{X} when \code{scaling = F}}
#' \item{Y_scaled}{output of \code{scale(Y)} when \code{scaling = T}; 
#' same as \code{Y} when \code{scaling = F}}
#' \item{b}{bandwidth of the kernel}
#' \item{I}{the indices of selected columns in the Nystrom}
#' \item{R}{the \eqn{n \times m} subsampled kernel matrix}
#' \item{M}{\eqn{R^T R + \lambda D}}
#' \item{dh}{\eqn{(R^T R + \lambda D)^{-1}R^T y}}
#' \item{yfitted}{fitted value of response variable}
#' \item{e}{residual}
#' \item{mse}{means squared errors}
#' \item{lambda}{\eqn{\lambda} used in the Nystrom KRLS}
#' \item{scaling}{indicate if data has been scaled or not}
#' 
#' @details
#' KRLS with Nystrom approximation. Adaptively select \eqn{m} from \eqn{n} data points
#' to create the \eqn{n \times m} kernel matrix, then solve 
#' \deqn{\hat{d} = (R^T R + \lambda D)^{-1}R^T y}
#' Then the fitted value \eqn{\hat{y} = R \hat{d}}. 
#'
#' @seealso [KRLS2::inference.krls_nys()]
#' @examples 
#'\dontrun{
#' n <- 2*1e3
#' x1 <- 5*rnorm(n)
#' x2 <- 10*runif(n)
#' ypure <- 2*x1 + 2*x2^2
#' y <- ypure + sqrt(2)*sd(ypure)*rnorm(n)
#' result <- KRLS2::krls_nys(X=cbind(x1,x2), y=y)
#' yh <- result$yfitted
#' inf <- inference.krls_nys(result)
#' }
#' @export

krls_nys <- function(X, y, 
                     b = NULL, 
                     I = NULL, 
                     lambda = NULL,
                     lambdaset = 10^(seq(-4, 1, length.out = 6)), 
                     folds = 5, 
                     m0 = 100, r0 = 3, p0 = 0.8, a0 = 5, 
                     scaling = T){
    # scale input if needed
    y_init <- y  # keep the original values of y 
    if (scaling){
        X <- scale(X)
        y <- scale(y)
    }
    # else{
    #     X <- X_init
    #     y <- y_init
    # }

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
 
    mod <- list(X_scaled=X, y_scaled=y, b=b, I =I, R = R, M= M, 
                dh = dh,  yfitted=yh, e=e, mse=mse, lambda = lambda,
                scaling = scaling)
    class(mod) <- c("krls_nys","list")
    mod
}
