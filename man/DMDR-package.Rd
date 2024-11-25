\name{DMDR-package}
\alias{DMDR-package}
\alias{DMDR}
\docType{package}
\title{
Distributed Martingale Difference Regression (DMDR)
}
\description{
The DMDR package provides tools for distributed parameter estimation and kernel-based methods in martingale difference regression. It enables efficient computation for large-scale datasets using distributed algorithms.
}
\details{
This package includes functions for:

- **Distributed parameter estimation**: Implements distributed algorithms for estimating parameters based on martingale difference regression.
- **Kernel-based methods**: Supports Epanichnikov kernels for univariate and multivariate inputs.
- **Matrix operations**: Functions for extracting and reconstructing lower triangular blocks of matrices.

Key functions:
- \code{\link{distributedAlgorithm}}: Distributed estimation algorithm.
- \code{\link{estimateScore}}: Main function for estimating scores and parameters.
- \code{\link{vecLower}} and \code{\link{vecInv}}: For matrix block operations.
- \code{\link{epanKernel}} and \code{\link{epanKernelMulti}}: Compute univariate and multivariate Epanichnikov kernel values.
}
\author{
Developed by Your Name.

Maintainer: Your Name \email{your_email@example.com}
}
\references{
- Hansen, B. E. (2008). Uniform Convergence Rates for Kernel Estimation with Dependent Data. *Econometric Theory*, 24(3), 726–748.
- Related work on martingale difference regression and distributed estimation.

For more information, refer to the vignettes or the reference manual.
}
\seealso{
\code{\link[stats]{lm}} for linear models, \code{\link[graphics]{plot}} for visualization, and \code{\link{distributedAlgorithm}} for distributed estimation.
}
\examples{
# Example: Running the distributed algorithm
data <- matrix(rnorm(1000), ncol = 10)  # Simulated data
result <- distributedAlgorithm(data, machines = 2, d = 3)
print(result)

# Example: Computing kernel values
u <- rnorm(1)
epan_value <- epanKernel(u)
cat("Epanichnikov Kernel Value:", epan_value, "\n")
}
\keyword{package}
