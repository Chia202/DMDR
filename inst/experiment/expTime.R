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

covMatrix = outer(1:p, 1:p, function(i, j) {
    0.5^abs(i - j)
})

meanFunc = function(xBeta) {
    sin(2 * xBeta) + 2 * exp(2 + xBeta)
}

varFunc = function(xBeta) {
    log(abs(2 + xBeta) + 1)
}

beta = c(1, 0, 0, 1, 0, 1, 1, -1, rep(0, d * p - 8))
beta = matrix(beta, ncol = d, byrow = TRUE)
x = mvrnorm(N, mu = rep(0, p), Sigma = covMatrix)
xBeta = x %*% beta
y = rnorm(N, mean = apply(xBeta, 1, sum), sd = 2)#, sd = varFunc(apply(xBeta, 1, sum)))
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
    times = times,
    unit = "s"
)

# print(dim(data))
# print(dmdrDense(data, m, d))
# print(dmdrDense(data, m, d, useParallel = TRUE))
# print(dmdrSparse(data, m, d))
# print(dmdrSparse(data, m, d, useParallel = TRUE))

print(testResult)

print("End time:")
print(as.character(Sys.time()))
