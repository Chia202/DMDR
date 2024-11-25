library(DMDR)
library(MASS)

set.seed(123)

p = 16
N = 2500
m = 5
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

print("Start time:")
print(as.character(Sys.time()))

meanY = meanFunc(xBeta)
sdY = sqrt(varFunc(x))

nRep = 100
disDP = numeric(nRep)
disSP = numeric(nRep)

for (i in 1:nRep) {
    set.seed(i)
    y = rnorm(N, mean = meanY, sd = sdY)
    data = as.matrix(data.frame(x, y))
    colnames(data) = c(paste0("x", 1:p), "y")

    cat("Replication:", i, "\n")
    resDP = dmdrDense(data, m, d, useParallel = TRUE)
    disDP[i] = trCor(beta, resDP)
    cat("disDP[i]:", disDP[i], "\n")

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
