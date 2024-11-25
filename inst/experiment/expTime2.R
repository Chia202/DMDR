library(DMDR)
library(MASS)
library(microbenchmark)

gc()

set.seed(123)

p = 16
N = 2500
m = 10
n = N / m
d = 2

beta1 = c(1, 0, 1, 1, rep(0, p - 4))
beta2 = c(0, 1, -1, 1, rep(0, p - 4))
beta = cbind(beta1, beta2)
x = matrix(runif(N * p, min = -2, max = 2), ncol = p)
xBeta = x %*% beta

meanFunc = function(xBeta) {
    (xBeta[, 1]) / (0.5 + (1.5 + xBeta[, 2])^2)
}

varFunc = function(x) {
    exp(x[, 1])
}

meanY = meanFunc(xBeta)
sdY = sqrt(varFunc(x))

y = rnorm(N, mean = meanY, sd = sdY)
data = as.matrix(data.frame(x, y))
colnames(data) = c(paste0("x", 1:p), "y")

print("Start time:")
print(as.character(Sys.time()))

times = 30
print("Number of times:")
print(times)

testResult = microbenchmark(
    # dmdrDense(data, m, d),
    # dmdrDense(data, m, d, useParallel = TRUE),
    dmdrSparse(data, m, d),
    dmdrSparse(data, m, d, useParallel = TRUE),
    times = times
)

print(testResult)

print("End time:")
print(as.character(Sys.time()))
