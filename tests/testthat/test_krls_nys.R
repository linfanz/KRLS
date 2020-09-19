context("krls_nys works properly")

Ntru <- 50
d <- 5 # number of covariates
# set all the bandwidth to be d
Xtru <- scale( matrix(runif(Ntru*d), nrow = Ntru, ncol = d) )
coeff <- rnorm(Ntru)
ftru <- function(X) {
  K_X_Xtru <- kernel_parallel_2(Xtru, X, d)
  t(K_X_Xtru) %*% coeff
}
sgm <- sqrt(2)*sd(ftru(Xtru))

# n <- 2000
# X <- scale( matrix(runif(n*d), nrow = n, ncol = d) )
# y_pure  <- ftru(X)
# y <- y_pure + rnorm(n, sd = sgm)
# 
# fit_krls_nys <- krls_nys(y, X, scaling = F)
# X_test <- matrix(runif(200*d), nrow = 200, ncol = d)
# yh <- predict.krls_nys(mod = fit_krls_nys, Xnew=X_test, dh= T, scaling = F)

# verify if the inference terms in Nystrom KRLS is the same as
n <- 300
X <- matrix(runif(n*d), nrow = n, ncol = d)
y_pure <- ftru(scale(X))
y <- y_pure + rnorm(n, sd = sgm)
# Nystrom 
fit_nys <- krls_nys(X, y,  I = 1:n, b = d, lambda = 1)
fit_krls <- krls(X,y, b = d, lambda = 1)
# compare the coefficients
head(fit_krls$coeffs)
head(fit_nys$dh)
mean((fit_krls$coeffs - fit_nys$dh)^2)
# compare the fitted values
head(fit_nys$yh)
head(fit_krls$fitted)
mean((fit_nys$yh - fit_krls$fitted)^2)

# inference items 
inf_krls <- inference.krls2(fit_krls)
inf_nys <- inference.krls_nys(fit_nys)
head(inf_krls$vcov.c[1, ])
head(inf_nys$vco[1, ])
norm(inf_nys$vco - inf_krls$vcov.c, type = "F")

inf_krls$avgderivatives
inf_nys$avePD

inf_krls$var.avgderivatives
inf_nys$avePD_var
