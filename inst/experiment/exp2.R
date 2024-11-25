library(DMDR)
library(MASS)

set.seed(123)

p = 16
N = 2500
m = 25
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

print("Start time:")
print(as.character(Sys.time()))

nRep = 100
# disDense = numeric(nRep)
disDP = numeric(nRep)
# disSparse = numeric(nRep)
disSP = numeric(nRep)

for (i in 1:nRep) {
    set.seed(i)
    y = rnorm(N, mean = apply(xBeta, 1, sum), sd = 2)#, sd = varFunc(apply(xBeta, 1, sum)))
    data = as.matrix(data.frame(x, y))
    colnames(data) = c(paste0("x", 1:p), "y")

    cat("Replication:", i, "\n")
    # resDense = dmdrDense(data, m, d)
    resDP = dmdrDense(data, m, d, useParallel = TRUE)
    disDP[i] = trCor(beta, resDP)
    cat("disDP[i]:", disDP[i], "\n")
    # resSparse = dmdrSparse(data, m, d)
    resSP = dmdrSparse(data, m, d, useParallel = TRUE)
    disSP[i] = trCor(beta, resSP)
    cat("disSP[i]:", disSP[i], "\n")

    if (i %% 10 == 0) {
        cat(i, "replications done.\n")
        cat("Mean distance (Dense Projection):", mean(disDP[1:i]), "\n")
        cat("Mean distance (Sparse Projection):", mean(disSP[1:i]), "\n")
    }
}

cat("Dense Projection\n")
cat("Mean :", mean(disDP), "\n")
cat("Std  :", sd(disDP) / sqrt(nRep), "\n")

cat("Sparse Projection\n")
cat("Mean :", mean(disSP), "\n")
cat("Std  :", sd(disSP) / sqrt(nRep), "\n")

print("End time:")
print(as.character(Sys.time()))
