- [Abstract](#abstract)
- [Materials](#materials)
- [Introduction](#introduction)
- [Probability](#probability)

# Abstract

Summarize ["Machine Learning: A Probabilistic Perspective" (2012)](https://probml.github.io/pml-book/book0.html).

# Materials

> - [데이터 사이언스 스쿨](https://datascienceschool.net/intro.html)
> - [공돌이의 수학정리노트 (Angelo's Math Notes)](https://angeloyeo.github.io/)
> - ["Probabilistic machine learning": a book series by Kevin Murphy](https://github.com/probml/pml-book)
>   - ["Machine Learning: A Probabilistic Perspective" (2012)](https://probml.github.io/pml-book/book0.html)
>   - ["Probabilistic Machine Learning: An Introduction" (2022)](https://probml.github.io/pml-book/book1.html)
>   - ["Probabilistic Machine Learning: Advanced Topics" (2023)](https://probml.github.io/pml-book/book2.html)

# Introduction

1.1 Machine learning: what and why?
1.1.1 Types of machine learning
1.2 Supervised learning
1.2.1 Classification
1.2.2 Regression
1.3 Unsupervised learning
1.3.1 Discovering clusters
1.3.2 Discovering latent factors
1.3.3 Discovering graph structure
1.3.4 Matrix completion
1.4 Some basic concepts in machine learning
1.4.1 Parametric vs non-parametric models
1.4.2 A simple non-parametric classifier: K-nearest neighbors
1.4.3 The curse of dimensionality
1.4.4 Parametric models for classification and regression
1.4.5 Linear regression
1.4.6 Logistic regression
1.4.7 Overfitting
1.4.8 Model selection
1.4.9 No free lunch theorem

# Probability

2.1 Introduction
2.2 A brief review of probability theory
2.2.1 Discrete random variables
2.2.2 Fundamental rules
2.2.3 Bayes rule
2.2.4 Independence and conditional independence
2.2.5 Continuous random variables
2.2.6 Quantiles
2.2.7 Mean and variance
2.3 Some common discrete distributions
2.3.1 The binomial and Bernoulli distributions
2.3.2 The multinomial and multinoulli distributions
2.3.3 The Poisson distribution
2.3.4 The empirical distribution
2.4 Some common continuous distributions
2.4.1 Gaussian (normal) distribution
2.4.2 Degenerate pdf
2.4.3 The Student t distribution
2.4.4 The Laplace distribution
2.4.5 The gamma distribution
2.4.6 The beta distribution
2.4.7 Pareto distribution
2.5 Joint probability distributions
2.5.1 Covariance and correlation
2.5.2 The multivariate Gaussian
2.5.3 Multivariate Student t distribution
2.5.4 Dirichlet distribution
2.6 Transformations of random variables
2.6.1 Linear transformations
2.6.2 General transformations
2.6.3 Central limit theorem
2.7 Monte Carlo approximation
2.7.1 Example: change of variables, the MC way
2.7.2 Example: estimating π by Monte Carlo integration
2.7.3 Accuracy of Monte Carlo approximation
2.8 Information theory
2.8.1 Entropy
2.8.2 KL divergence
2.8.3 Mutual information

3 Generative models for discrete data
3.1 Introduction
3.2 Bayesian concept learning
3.2.1 Likelihood
3.2.2 Prior
3.2.3 Posterior
3.2.4 Posterior predictive distribution
3.2.5 A more complex prior
3.3 The beta-binomial model
3.3.1 Likelihood
3.3.2 Prior
3.3.3 Posterior
3.3.4 Posterior predictive distribution
3.4 The Dirichlet-multinomial model
3.4.1 Likelihood
3.4.2 Prior
3.4.3 Posterior
3.4.4 Posterior predictive
3.5 Naive Bayes classifiers
3.5.1 Model fitting
3.5.2 Using the model for prediction
3.5.3 The log-sum-exp trick
3.5.4 Feature selection using mutual information
3.5.5 Classifying documents using bag of words

4 Gaussian models
4.1 Introduction
4.1.1 Notation
4.1.2 Basics
4.1.3 MLE for an MVN
4.1.4 Maximum entropy derivation of the Gaussian *
4.2 Gaussian discriminant analysis
4.2.1 Quadratic discriminant analysis (QDA)
4.2.2 Linear discriminant analysis (LDA)
4.2.3 Two-class LDA
4.2.4 MLE for discriminant analysis
4.2.5 Strategies for preventing overfitting
4.2.6 Regularized LDA *
4.2.7 Diagonal LDA
4.2.8 Nearest shrunken centroids classifier *
4.3 Inference in jointly Gaussian distributions
4.3.1 Statement of the result
4.3.2 Examples
4.3.3 Information form
4.3.4 Proof of the result *
4.4 Linear Gaussian systems
4.4.1 Statement of the result
4.4.2 Examples
4.4.3 Proof of the result *
4.5 Digression: The Wishart distribution *
4.5.1 Inverse Wishart distribution
4.5.2 Visualizing the Wishart distribution *
4.6 Inferring the parameters of an MVN
4.6.1 Posterior distribution of µ
4.6.2 Posterior distribution of Σ *
4.6.3 Posterior distribution of µ and Σ *
4.6.4 Sensor fusion with unknown precisions *

5 Bayesian statistics
5.1 Introduction
5.2 Summarizing posterior distributions
5.2.1 MAP estimation
5.2.2 Credible intervals
5.2.3 Inference for a di!erence in proportions
5.3 Bayesian model selection
5.3.1 Bayesian Occam’s razor
5.3.2 Computing the marginal likelihood (evidence)
5.3.3 Bayes factors
5.3.4 Je!reys-Lindley paradox *
5.4 Priors
5.4.1 Uninformative priors
5.4.2 Je!reys priors *
5.4.3 Robust priors
5.4.4 Mixtures of conjugate priors
5.5 Hierarchical Bayes
5.5.1 Example: modeling related cancer rates
5.6 Empirical Bayes
5.6.1 Example: beta-binomial model
5.6.2 Example: Gaussian-Gaussian model
5.7 Bayesian decision theory
5.7.1 Bayes estimators for common loss functions
5.7.2 The false positive vs false negative tradeo!
5.7.3 Other topics *

6 Frequentist statistics
6.1 Introduction
6.2 Sampling distribution of an estimator
6.2.1 Bootstrap
6.2.2 Large sample theory for the MLE *
6.3 Frequentist decision theory
6.3.1 Bayes risk
6.3.2 Minimax risk
6.3.3 Admissible estimators
6.4 Desirable properties of estimators
6.4.1 Consistent estimators
6.4.2 Unbiased estimators
6.4.3 Minimum variance estimators
6.4.4 The bias-variance tradeo!
6.5 Empirical risk minimization
6.5.1 Regularized risk minimization
6.5.2 Structural risk minimization
6.5.3 Estimating the risk using cross validation
6.5.4 Upper bounding the risk using statistical learning theory *
CONTENTS xi
6.5.5 Surrogate loss functions
6.6 Pathologies of frequentist statistics *
6.6.1 Counter-intuitive behavior of confidence intervals
6.6.2 p-values considered harmful
6.6.3 The likelihood principle
6.6.4 Why isn’t everyone a Bayesian?

7 Linear regression
7.1 Introduction
7.2 Model specification
7.3 Maximum likelihood estimation (least squares)
7.3.1 Derivation of the MLE
7.3.2 Geometric interpretation
7.3.3 Convexity
7.4 Robust linear regression *
7.5 Ridge regression
7.5.1 Basic idea
7.5.2 Numerically stable computation *
7.5.3 Connection with PCA *
7.5.4 Regularization e!ects of big data
7.6 Bayesian linear regression
7.6.1 Computing the posterior
7.6.2 Computing the posterior predictive
7.6.3 Bayesian inference when σ2 is unknown *
7.6.4 EB for linear regression (evidence procedure)

8 Logistic regression
8.1 Introduction
8.2 Model specification
8.3 Model fitting
8.3.1 MLE
8.3.2 Steepest descent
8.3.3 Newton’s method
8.3.4 Iteratively reweighted least squares (IRLS)
8.3.5 Quasi-Newton (variable metric) methods
8.3.6 #2 regularization
8.3.7 Multi-class logistic regression
8.4 Bayesian logistic regression
8.4.1 Laplace approximation
8.4.2 Derivation of the BIC
8.4.3 Gaussian approximation for logistic regression
8.4.4 Approximating the posterior predictive
8.4.5 Residual analysis (outlier detection) *
8.5 Online learning and stochastic optimization
8.5.1 Online learning and regret minimization
xii CONTENTS
8.5.2 Stochastic optimization and risk minimization
8.5.3 The LMS algorithm
8.5.4 The perceptron algorithm
8.5.5 A Bayesian view
8.6 Generative vs discriminative classifiers
8.6.1 Pros and cons of each approach
8.6.2 Dealing with missing data
8.6.3 Fisher’s linear discriminant analysis (FLDA) *

9 Generalized linear models and the exponential family
9.1 Introduction
9.2 The exponential family
9.2.1 Definition
9.2.2 Examples
9.2.3 Log partition function
9.2.4 MLE for the exponential family
9.2.5 Bayes for the exponential family *
9.2.6 Maximum entropy derivation of the exponential family *
9.3 Generalized linear models (GLMs)
9.3.1 Basics
9.3.2 ML and MAP estimation
9.3.3 Bayesian inference
9.4 Probit regression
9.4.1 ML/MAP estimation using gradient-based optimization
9.4.2 Latent variable interpretation
9.4.3 Ordinal probit regression *
9.4.4 Multinomial probit models *
9.5 Multi-task learning
9.5.1 Hierarchical Bayes for multi-task learning
9.5.2 Application to personalized email spam filtering
9.5.3 Application to domain adaptation
9.5.4 Other kinds of prior
9.6 Generalized linear mixed models *
9.6.1 Example: semi-parametric GLMMs for medical data
9.6.2 Computational issues
9.7 Learning to rank *
9.7.1 The pointwise approach
9.7.2 The pairwise approach
9.7.3 The listwise approach
9.7.4 Loss functions for ranking

10 Directed graphical models (Bayes nets)
10.1 Introduction
10.1.1 Chain rule
10.1.2 Conditional independence

10.1.3 Graphical models
10.1.4 Graph terminology
10.1.5 Directed graphical models
10.2 Examples
10.2.1 Naive Bayes classifiers
10.2.2 Markov and hidden Markov models
10.2.3 Medical diagnosis
10.2.4 Genetic linkage analysis *
10.2.5 Directed Gaussian graphical models *
10.3 Inference
10.4 Learning
10.4.1 Plate notation
10.4.2 Learning from complete data
10.4.3 Learning with missing and/or latent variables
10.5 Conditional independence properties of DGMs
10.5.1 d-separation and the Bayes Ball algorithm (global Markov
properties)
10.5.2 Other Markov properties of DGMs
10.5.3 Markov blanket and full conditionals
10.6 Influence (decision) diagrams *

11 Mixture models and the EM algorithm
11.1 Latent variable models
11.2 Mixture models
11.2.1 Mixtures of Gaussians
11.2.2 Mixture of multinoullis
11.2.3 Using mixture models for clustering
11.2.4 Mixtures of experts
11.3 Parameter estimation for mixture models
11.3.1 Unidentifiability
11.3.2 Computing a MAP estimate is non-convex
11.4 The EM algorithm
11.4.1 Basic idea
11.4.2 EM for GMMs
11.4.3 EM for mixture of experts
11.4.4 EM for DGMs with hidden variables
11.4.5 EM for the Student distribution *
11.4.6 EM for probit regression *
11.4.7 Theoretical basis for EM *
11.4.8 Online EM
11.4.9 Other EM variants *
11.5 Model selection for latent variable models
11.5.1 Model selection for probabilistic models
11.5.2 Model selection for non-probabilistic methods
11.6 Fitting models with missing data
xiv CONTENTS
11.6.1 EM for the MLE of an MVN with missing data

12 Latent linear models
12.1 Factor analysis
12.1.1 FA is a low rank parameterization of an MVN
12.1.2 Inference of the latent factors
12.1.3 Unidentifiability
12.1.4 Mixtures of factor analysers
12.1.5 EM for factor analysis models
12.1.6 Fitting FA models with missing data
12.2 Principal components analysis (PCA)
12.2.1 Classical PCA: statement of the theorem
12.2.2 Proof *
12.2.3 Singular value decomposition (SVD)
12.2.4 Probabilistic PCA
12.2.5 EM algorithm for PCA
12.3 Choosing the number of latent dimensions
12.3.1 Model selection for FA/PPCA
12.3.2 Model selection for PCA
12.4 PCA for categorical data
12.5 PCA for paired and multi-view data
12.5.1 Supervised PCA (latent factor regression)
12.5.2 Partial least squares
12.5.3 Canonical correlation analysis
12.6 Independent Component Analysis (ICA)
12.6.1 Maximum likelihood estimation
12.6.2 The FastICA algorithm
12.6.3 Using EM
12.6.4 Other estimation principles *

13 Sparse linear models
13.1 Introduction
13.2 Bayesian variable selection
13.2.1 The spike and slab model
13.2.2 From the Bernoulli-Gaussian model to #0 regularization
13.2.3 Algorithms
13.3 #1 regularization: basics
13.3.1 Why does #1 regularization yield sparse solutions?
13.3.2 Optimality conditions for lasso
13.3.3 Comparison of least squares, lasso, ridge and subset selection
13.3.4 Regularization path
13.3.5 Model selection
13.3.6 Bayesian inference for linear models with Laplace priors
13.4 #1 regularization: algorithms
13.4.1 Coordinate descent
CONTENTS xv
13.4.2 LARS and other homotopy methods
13.4.3 Proximal and gradient projection methods
13.4.4 EM for lasso
13.5 #1 regularization: extensions
13.5.1 Group Lasso
13.5.2 Fused lasso
13.5.3 Elastic net (ridge and lasso combined)
13.6 Non-convex regularizers
13.6.1 Bridge regression
13.6.2 Hierarchical adaptive lasso
13.6.3 Other hierarchical priors
13.7 Automatic relevance determination (ARD)/sparse Bayesian learning (SBL)
13.7.1 ARD for linear regression
13.7.2 Whence sparsity?
13.7.3 Connection to MAP estimation
13.7.4 Algorithms for ARD *
13.7.5 ARD for logistic regression
13.8 Sparse coding *
13.8.1 Learning a sparse coding dictionary
13.8.2 Results of dictionary learning from image patches
13.8.3 Compressed sensing
13.8.4 Image inpainting and denoising

14 Kernels
14.1 Introduction
14.2 Kernel functions
14.2.1 RBF kernels
14.2.2 Kernels for comparing documents
14.2.3 Mercer (positive definite) kernels
14.2.4 Linear kernels
14.2.5 Matern kernels
14.2.6 String kernels
14.2.7 Pyramid match kernels
14.2.8 Kernels derived from probabilistic generative models
14.3 Using kernels inside GLMs
14.3.1 Kernel machines
14.3.2 L1VMs, RVMs, and other sparse vector machines
14.4 The kernel trick
14.4.1 Kernelized nearest neighbor classification
14.4.2 Kernelized K-medoids clustering
14.4.3 Kernelized ridge regression
14.4.4 Kernel PCA
14.5 Support vector machines (SVMs)
14.5.1 SVMs for regression
14.5.2 SVMs for classification
xvi CONTENTS
14.5.3 Choosing C
14.5.4 Summary of key points
14.5.5 A probabilistic interpretation of SVMs
14.6 Comparison of discriminative kernel methods
14.7 Kernels for building generative models
14.7.1 Smoothing kernels
14.7.2 Kernel density estimation (KDE)
14.7.3 From KDE to KNN
14.7.4 Kernel regression
14.7.5 Locally weighted regression

15 Gaussian processes
15.1 Introduction
15.2 GPs for regression
15.2.1 Predictions using noise-free observations
15.2.2 Predictions using noisy observations
15.2.3 E!ect of the kernel parameters
15.2.4 Estimating the kernel parameters
15.2.5 Computational and numerical issues *
15.2.6 Semi-parametric GPs *
15.3 GPs meet GLMs
15.3.1 Binary classification
15.3.2 Multi-class classification
15.3.3 GPs for Poisson regression
15.4 Connection with other methods
15.4.1 Linear models compared to GPs
15.4.2 Linear smoothers compared to GPs
15.4.3 SVMs compared to GPs
15.4.4 L1VM and RVMs compared to GPs
15.4.5 Neural networks compared to GPs
15.4.6 Smoothing splines compared to GPs *
15.4.7 RKHS methods compared to GPs *
15.5 GP latent variable model
15.6 Approximation methods for large datasets

16 Adaptive basis function models
16.1 Introduction
16.2 Classification and regression trees (CART)
16.2.1 Basics
16.2.2 Growing a tree
16.2.3 Pruning a tree
16.2.4 Pros and cons of trees
16.2.5 Random forests
16.2.6 CART compared to hierarchical mixture of experts *
16.3 Generalized additive models
16.3.1 Backfitting
16.3.2 Computational e"ciency
16.3.3 Multivariate adaptive regression splines (MARS)
16.4 Boosting
16.4.1 Forward stagewise additive modeling
16.4.2 L2boosting
16.4.3 AdaBoost
16.4.4 LogitBoost
16.4.5 Boosting as functional gradient descent
16.4.6 Sparse boosting
16.4.7 Multivariate adaptive regression trees (MART)
16.4.8 Why does boosting work so well?
16.4.9 A Bayesian view
16.5 Feedforward neural networks (multilayer perceptrons)
16.5.1 Convolutional neural networks
16.5.2 Other kinds of neural networks
16.5.3 A brief history of the field
16.5.4 The backpropagation algorithm
16.5.5 Identifiability
16.5.6 Regularization
16.5.7 Bayesian inference *
16.6 Ensemble learning
16.6.1 Stacking
16.6.2 Error-correcting output codes
16.6.3 Ensemble learning is not equivalent to Bayes model averaging
16.7 Experimental comparison
16.7.1 Low-dimensional features
16.7.2 High-dimensional features
16.8 Interpreting black-box models

17 Markov and hidden Markov models
17.1 Introduction
17.2 Markov models
17.2.1 Transition matrix
17.2.2 Application: Language modeling
17.2.3 Stationary distribution of a Markov chain *
17.2.4 Application: Google’s PageRank algorithm for web page ranking *
17.3 Hidden Markov models
17.3.1 Applications of HMMs
17.4 Inference in HMMs
17.4.1 Types of inference problems for temporal models
17.4.2 The forwards algorithm
17.4.3 The forwards-backwards algorithm
17.4.4 The Viterbi algorithm
17.4.5 Forwards filtering, backwards sampling
17.5 Learning for HMMs
17.5.1 Training with fully observed data
17.5.2 EM for HMMs (the Baum-Welch algorithm)
17.5.3 Bayesian methods for “fitting” HMMs *
17.5.4 Discriminative training
17.5.5 Model selection
17.6 Generalizations of HMMs
17.6.1 Variable duration (semi-Markov) HMMs
17.6.2 Hierarchical HMMs
17.6.3 Input-output HMMs
17.6.4 Auto-regressive and buried HMMs
17.6.5 Factorial HMM
17.6.6 Coupled HMM and the influence model
17.6.7 Dynamic Bayesian networks (DBNs)

18 State space models
18.1 Introduction
18.2 Applications of SSMs
18.2.1 SSMs for object tracking
18.2.2 Robotic SLAM
18.2.3 Online parameter learning using recursive least squares
18.2.4 SSM for time series forecasting *
18.3 Inference in LG-SSM
18.3.1 The Kalman filtering algorithm
18.3.2 The Kalman smoothing algorithm
18.4 Learning for LG-SSM
18.4.1 Identifiability and numerical stability
18.4.2 Training with fully observed data
18.4.3 EM for LG-SSM
18.4.4 Subspace methods
18.4.5 Bayesian methods for “fitting” LG-SSMs
18.5 Approximate online inference for non-linear, non-Gaussian SSMs
18.5.1 Extended Kalman filter (EKF)
18.5.2 Unscented Kalman filter (UKF)
18.5.3 Assumed density filtering (ADF)
18.6 Hybrid discrete/continuous SSMs
18.6.1 Inference
18.6.2 Application: Data association and multi-target tracking
18.6.3 Application: fault diagnosis
18.6.4 Application: econometric forecasting

19 Undirected graphical models (Markov random fields)
19.1 Introduction
19.2 Conditional independence properties of UGMs
19.2.1 Key properties
19.2.2 An undirected alternative to d-separation
19.2.3 Comparing directed and undirected graphical models
19.3 Parameterization of MRFs
19.3.1 The Hammersley-Cli!ord theorem
19.3.2 Representing potential functions
19.4 Examples of MRFs
19.4.1 Ising model
19.4.2 Hopfield networks
19.4.3 Potts model
19.4.4 Gaussian MRFs
19.4.5 Markov logic networks *
19.5 Learning
19.5.1 Training maxent models using gradient methods
19.5.2 Training partially observed maxent models
19.5.3 Approximate methods for computing the MLEs of MRFs
19.5.4 Pseudo likelihood
19.5.5 Stochastic maximum likelihood
19.5.6 Feature induction for maxent models *
19.5.7 Iterative proportional fitting (IPF) *
19.6 Conditional random fields (CRFs)
19.6.1 Chain-structured CRFs, MEMMs and the label-bias problem
19.6.2 Applications of CRFs
19.6.3 CRF training
19.7 Structural SVMs
19.7.1 SSVMs: a probabilistic view
19.7.2 SSVMs: a non-probabilistic view
19.7.3 Cutting plane methods for fitting SSVMs
19.7.4 Online algorithms for fitting SSVMs
19.7.5 Latent structural SVMs

20 Exact inference for graphical models
20.1 Introduction
20.2 Belief propagation for trees
20.2.1 Serial protocol
20.2.2 Parallel protocol
20.2.3 Gaussian BP *
20.2.4 Other BP variants *
20.3 The variable elimination algorithm
20.3.1 The generalized distributive law *
20.3.2 Computational complexity of VE
20.3.3 A weakness of VE
20.4 The junction tree algorithm *
20.4.1 Creating a junction tree
20.4.2 Message passing on a junction tree
20.4.3 Computational complexity of JTA
20.4.4 JTA generalizations *
20.5 Computational intractability of exact inference in the worst case
20.5.1 Approximate inference

21 Variational inference
21.1 Introduction
21.2 Variational inference
21.2.1 Alternative interpretations of the variational objective
21.2.2 Forward or reverse KL? *
21.3 The mean field method
21.3.1 Derivation of the mean field update equations
21.3.2 Example: mean field for the Ising model
21.4 Structured mean field *
21.4.1 Example: factorial HMM
21.5 Variational Bayes
21.5.1 Example: VB for a univariate Gaussian
21.5.2 Example: VB for linear regression
21.6 Variational Bayes EM
21.6.1 Example: VBEM for mixtures of Gaussians *
21.7 Variational message passing and VIBES
21.8 Local variational bounds *
21.8.1 Motivating applications
21.8.2 Bohning’s quadratic bound to the log-sum-exp function
21.8.3 Bounds for the sigmoid function
21.8.4 Other bounds and approximations to the log-sum-exp function *
21.8.5 Variational inference based on upper bounds

22 More variational inference
22.1 Introduction
22.2 Loopy belief propagation: algorithmic issues
22.2.1 A brief history
22.2.2 LBP on pairwise models
22.2.3 LBP on a factor graph
22.2.4 Convergence
22.2.5 Accuracy of LBP
22.2.6 Other speedup tricks for LBP *
22.3 Loopy belief propagation: theoretical issues *
22.3.1 UGMs represented in exponential family form
22.3.2 The marginal polytope
22.3.3 Exact inference as a variational optimization problem
22.3.4 Mean field as a variational optimization problem
22.3.5 LBP as a variational optimization problem
22.3.6 Loopy BP vs mean field
22.4 Extensions of belief propagation *
22.4.1 Generalized belief propagation
22.4.2 Convex belief propagation
22.5 Expectation propagation
22.5.1 EP as a variational inference problem
22.5.2 Optimizing the EP objective using moment matching
22.5.3 EP for the clutter problem
22.5.4 LBP is a special case of EP
22.5.5 Ranking players using TrueSkill
22.5.6 Other applications of EP
22.6 MAP state estimation
22.6.1 Linear programming relaxation
22.6.2 Max-product belief propagation
22.6.3 Graphcuts
22.6.4 Experimental comparison of graphcuts and BP
22.6.5 Dual decomposition

23 Monte Carlo inference
23.1 Introduction
23.2 Sampling from standard distributions
23.2.1 Using the cdf
23.2.2 Sampling from a Gaussian (Box-Muller method)
23.3 Rejection sampling
23.3.1 Basic idea
23.3.2 Example
23.3.3 Application to Bayesian statistics
23.3.4 Adaptive rejection sampling
23.3.5 Rejection sampling in high dimensions
23.4 Importance sampling
23.4.1 Basic idea
23.4.2 Handling unnormalized distributions
23.4.3 Importance sampling for a DGM: Likelihood weighting
23.4.4 Sampling importance resampling (SIR)
23.5 Particle filtering
23.5.1 Sequential importance sampling
23.5.2 The degeneracy problem
23.5.3 The resampling step
23.5.4 The proposal distribution
23.5.5 Application: robot localization
23.5.6 Application: visual object tracking
23.5.7 Application: time series forecasting
23.6 Rao-Blackwellised particle filtering (RBPF)
23.6.1 RBPF for switching LG-SSMs
23.6.2 Application: tracking a maneuvering target
23.6.3 Application: Fast SLAM

24 Markov chain Monte Carlo (MCMC) inference
24.1 Introduction
24.2 Gibbs sampling
24.2.1 Basic idea
24.2.2 Example: Gibbs sampling for the Ising model
24.2.3 Example: Gibbs sampling for inferring the parameters of a GMM
24.2.4 Collapsed Gibbs sampling *
24.2.5 Gibbs sampling for hierarchical GLMs
24.2.6 BUGS and JAGS
24.2.7 The Imputation Posterior (IP) algorithm
24.2.8 Blocking Gibbs sampling
24.3 Metropolis Hastings algorithm
24.3.1 Basic idea
24.3.2 Gibbs sampling is a special case of MH
24.3.3 Proposal distributions
24.3.4 Adaptive MCMC
24.3.5 Initialization and mode hopping
24.3.6 Why MH works *
24.3.7 Reversible jump (trans-dimensional) MCMC *
24.4 Speed and accuracy of MCMC
24.4.1 The burn-in phase
24.4.2 Mixing rates of Markov chains *
24.4.3 Practical convergence diagnostics
24.4.4 Accuracy of MCMC
24.4.5 How many chains?
24.5 Auxiliary variable MCMC *
24.5.1 Auxiliary variable sampling for logistic regression
24.5.2 Slice sampling
24.5.3 Swendsen Wang
24.5.4 Hybrid/Hamiltonian MCMC *
24.6 Annealing methods
24.6.1 Simulated annealing
24.6.2 Annealed importance sampling
24.6.3 Parallel tempering
24.7 Approximating the marginal likelihood
24.7.1 The candidate method
24.7.2 Harmonic mean estimate
24.7.3 Annealed importance sampling

25 Clustering
25.1 Introduction
25.1.1 Measuring (dis)similarity
25.1.2 Evaluating the output of clustering methods *
25.2 Dirichlet process mixture models
25.2.1 From finite to infinite mixture models
25.2.2 The Dirichlet process
25.2.3 Applying Dirichlet processes to mixture modeling
25.2.4 Fitting a DP mixture model
25.3 A"nity propagation
25.4 Spectral clustering
25.4.1 Graph Laplacian
25.4.2 Normalized graph Laplacian
25.4.3 Example
25.5 Hierarchical clustering
25.5.1 Agglomerative clustering
25.5.2 Divisive clustering
25.5.3 Choosing the number of clusters
25.5.4 Bayesian hierarchical clustering
25.6 Clustering datapoints and features
25.6.1 Biclustering
25.6.2 Multi-view clustering

26 Graphical model structure learning
26.1 Introduction
26.2 Structure learning for knowledge discovery
26.2.1 Relevance networks
26.2.2 Dependency networks
26.3 Learning tree structures
26.3.1 Directed or undirected tree?
26.3.2 Chow-Liu algorithm for finding the ML tree structure
26.3.3 Finding the MAP forest
26.3.4 Mixtures of trees
26.4 Learning DAG structures
26.4.1 Markov equivalence
26.4.2 Exact structural inference
26.4.3 Scaling up to larger graphs
26.5 Learning DAG structure with latent variables
26.5.1 Approximating the marginal likelihood when we have missing data
26.5.2 Structural EM
26.5.3 Discovering hidden variables
26.5.4 Case study: Google’s Rephil
26.5.5 Structural equation models *
26.6 Learning causal DAGs
26.6.1 Causal interpretation of DAGs
26.6.2 Using causal DAGs to resolve Simpson’s paradox
26.6.3 Learning causal DAG structures
26.7 Learning undirected Gaussian graphical models
26.7.1 MLE for a GGM
26.7.2 Graphical lasso
26.7.3 Bayesian inference for GGM structure *
26.7.4 Handling non-Gaussian data using copulas *
26.8 Learning undirected discrete graphical models
26.8.1 Graphical lasso for MRFs/CRFs
26.8.2 Thin junction trees

27 Latent variable models for discrete data
27.1 Introduction
27.2 Distributed state LVMs for discrete data
27.2.1 Mixture models
27.2.2 Exponential family PCA
27.2.3 LDA and mPCA
27.2.4 GaP model and non-negative matrix factorization
27.3 Latent Dirichlet allocation (LDA)
27.3.1 Basics
27.3.2 Unsupervised discovery of topics
27.3.3 Quantitatively evaluating LDA as a language model
27.3.4 Fitting using (collapsed) Gibbs sampling
27.3.5 Example
27.3.6 Fitting using batch variational inference
27.3.7 Fitting using online variational inference
27.3.8 Determining the number of topics
27.4 Extensions of LDA
27.4.1 Correlated topic model
27.4.2 Dynamic topic model
27.4.3 LDA-HMM
27.4.4 Supervised LDA
27.5 LVMs for graph-structured data
27.5.1 Stochastic block model
27.5.2 Mixed membership stochastic block model
27.5.3 Relational topic model
27.6 LVMs for relational data
27.6.1 Infinite relational model
27.6.2 Probabilistic matrix factorization for collaborative filtering
27.7 Restricted Boltzmann machines (RBMs)
27.7.1 Varieties of RBMs
27.7.2 Learning RBMs
27.7.3 Applications of RBMs

28 Deep learning
28.1 Introduction
28.2 Deep generative models
28.2.1 Deep directed networks
28.2.2 Deep Boltzmann machines
28.2.3 Deep belief networks
28.2.4 Greedy layer-wise learning of DBNs
28.3 Deep neural networks
28.3.1 Deep multi-layer perceptrons
28.3.2 Deep auto-encoders
28.3.3 Stacked denoising auto-encoders
28.4 Applications of deep networks
28.4.1 Handwritten digit classification using DBNs
28.4.2 Data visualization and feature discovery using deep auto-encoders
28.4.3 Information retrieval using deep auto-encoders (semantic hashing)
28.4.4 Learning audio features using 1d convolutional DBNs
28.4.5 Learning image features using 2d convolutional DBNs
28.5 Discussion
