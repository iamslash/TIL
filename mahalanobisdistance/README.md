# Abstract

[마할라노비스 거리 | 공돌이수학](https://angeloyeo.github.io/2022/09/28/Mahalanobis_distance.html)

두점 a, b 의 유클리디안 거리와 c, d 의 유클리디안 거리가 같다. 그러나 a, b
사이에 많은 데이터가 밀집해 있다면 a, b 의 마할라노비스 거리가 c, d 의
마할라노비스 거리보다 가깝다. 마할라노비스 거리는 맥락을 고려한 거리이다.

The Mahalanobis distance is a measure of the distance between a point and a
distribution, named after the Indian statistician 
**Prasanta Chandra Mahalanobis**. 

It is used to determine the similarity between a point and a
given dataset or between two datasets. The Mahalanobis distance takes into
account the correlations between variables and scales the variables by their
variances, making it a more accurate measure of distance than the Euclidean
distance in multivariate analysis.

Mathematically, the Mahalanobis distance (MD) between a point `x` and a
distribution with mean `μ` and covariance matrix `Σ` is defined as:

```c
MD(x) = √((x - μ)^T * Σ^(-1) * (x - μ))

Where:

x is the data point
μ is the mean of the distribution
Σ is the covariance matrix of the distribution
^T denotes the transpose of a matrix
^(-1) denotes the inverse of a matrix
The Mahalanobis distance is widely used in various fields, including pattern recognition, classification, and anomaly detection.
```
