# DMDR

This package implements the distributed mean dimension reduction algorithm proposed by [Zhu et al. (2025)](https://www3.stat.sinica.edu.tw/preprint/SS-2022-0157_Preprint.pdf). You can refer to my [report](https://github.com/Chia202/DMDR/inst/doc/DMDR-Report.pdf) for more details.

You can install this package by running the following code in your `R` console:

```R
install.packages("remotes")
remotes::install_github("Chia202/DMDR")
```

## Uasge

```R
library(DMDR)

set.seed(123)

p = 16    # Dimension of covariate
N = 2500  # Total sample size
m = 5     # Number of machines
n = N / m # Sample size on each machine
d = 2     # Number of directions

beta = c(1, 0, 0, 1, 0, 1, 1, -1, rep(0, d * p - 8))
beta = matrix(beta, ncol = d, byrow = TRUE)
x = matrix(rnorm(N * p), nrow = p)
y = rowSums(x %*% beta) + rnorm(N)

# Dense sulotion
resDP = dmdrDense(data, m, d, useParallel = TRUE)

# Sparse solution
resSP = dmdrSparse(data, m, d, useParallel = TRUE)

# Evaluation, the smaller, the better.
cat("Correlation distance lies in [0, 1]")
cat("  Dense : " trCor(beta, resDP))
cat("  Sparse: " trCor(beta, resSP))
```

## Reference

[1] Z. Zhu, W. Xu, and L. Zhu, “Distributed Mean Dimension Reduction Through Semi-parametric Approaches,” Statistics Sinca, 2025, doi: [10.5705/ss.202022.0157](https://www3.stat.sinica.edu.tw/preprint/SS-2022-0157_Preprint.pdf).

[2] Y. Ma and L. Zhu, “A Semiparametric Approach to Dimension Reduction,” Journal of the American Statistical Association, 2012, doi: [10.1080/01621459.2011.646925](https://www.tandfonline.com/doi/full/10.1080/01621459.2011.646925).
