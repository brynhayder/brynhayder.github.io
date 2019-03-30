---
layout: post
title:  "Notes on Gaussian Processes for Regression"
date:   2018-03-07 00:00:00 +0000
categories: jekyll update
usemathjax: true
---

**Disclaimer**

 These notes are mostly for my own purposes, so they may be a bit rubbish in some places.
 
---

## Motivation
We will be interested in regression problems with a single output dimension
<div>
    \begin{equation}
    y = f(\mathbf{x}) + \varepsilon
    \end{equation}
</div>
where $$\mathbf{x} \in \mathbb{R}^n$$, $$y \in \mathbb{R}$$ and $$\varepsilon \sim \mathcal{N}(0, \sigma^2)$$

We will find that characterising $$f$$ as a Gaussian process (GP) provides a flexible yet interpretable family of models for this problem.

![generic_fit](/images/Gaussian-Process-Regression/generic_fit.png)


## Definition

>A Gaussian process is a (possibly infinite) collection of random variables such that any finite subset forms multivariate Gaussian distribution

We are interested in the case in which the GP is parameterised by $$\mathbf{x} \in \mathbb{R}^n$$. That is, our GPs will provide a measure over functions $$f: \mathbb{R}^n \to \mathbb{R}$$.


We denote a GP $$f$$ by 
<div>
    \begin{equation}
    f \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x'}))
    \end{equation}
</div>
where $$m: \mathbb{R}^n \to \mathbb{R}$$ is the mean function and $$k$$ is the covariance kernel, which returns a covariance matrix

<div>
    \begin{equation}
    k: \mathbb{R}^n \times \mathbb{R}^n \to \{\Sigma \in GL_n(\mathbb{R}) \, | \, \Sigma \succeq 0 \land \Sigma^{\mathsf{T}} = \Sigma\}.
    \end{equation}
</div>

These functions satisfy 
<div>
    \begin{align}
    m(\mathbf{x}) &= \mathrm{E}[f(\mathbf{x})] \\ 
    k(\mathbf{x}, \mathbf{x}') &= \mathrm{cov}[f(\mathbf{x}), f(\mathbf{x}')].
    \end{align}
</div>

Evaluating $$f$$ at $$\mathbf{x}$$ gives a Gaussian distribution

<div>
    \begin{equation}
    \mathbf{f} \sim \mathcal{N}(\mathbf{m}, \Sigma)
    \end{equation}
</div>
where $$\mathbf{m} = m(\mathbf{x})$$ and $$\Sigma_{ij} = k(x_i, x_j)$$. As such, we can think of a Gaussian process (GP) as providing a distribution over functions via the evaluation functional. 

Note that we have specified the GP in terms of a function that generates a covariance matrix, rather than a precision matrix. This is because the GP needs to satisfy the same marginalisation properties as the Gaussian distribution. 

## The Big Picture
The idea is that a GP gives a way of specifying a prior distribution over functions (in the Bayesian sense), by choosing the mean and covariance functions. 

Then, using some training data, we can calculate a posterior distribution over functions for our regression problem. This allows us to make predictions and gives confidence intervals in a natural way. 

Next we will talk about specifying the prior mean and covariance functions, then we will discuss how to a make predictions using GPs.

## The Mean Function
As we saw above, the mean function specifies the mean of the draws from the GP. 

After this section, we will assume that our GP is specified to have prior mean of zero. There are a couple reasons for this.

First, if we wanted to model using a deterministic mean function, we could always just model $$y - m = f - m + \varepsilon$$ and put the mean back in after prediction.

Second, the mean can be marginalised out. That is 
<div>
    \begin{align}
     f|a &\sim \mathcal{GP}(a\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')) \quad\textrm{with}\quad a \sim \mathcal{N}(0, 1) \\ 
    \implies\quad f &\sim \mathcal{GP}(0, \mu(\mathbf{x})\mu(\mathbf{x}') +  k(\mathbf{x}, \mathbf{x}')).
    \end{align}
</div>

If you use a local kernel (one that decays quickly to zero as the covariates move apart in $$\mathbb{R}^n$$), then away from the training data you will predict the prior mean of your GP. This can provide a downside to modelling a trivial prior mean $$m = 0$$, since you lose structure in your predictions away from the training data.


## The Covariance Kernel
When drawing from a GP it is the kernel that determines how these random variables are correlated. The covariance kernel is what gives structure to the GP and the majority of modelling effort will generally go into choosing the kernel.

#### Definition
The kernel is the key determinant of the structure of the GP. As such, this choice encodes your assumptions about the data generating process. Below we will list a few of the common choices for kernels, but you are by all means allowed to define your own kernel, the only requirement is that your function generates a valid covariance matrix.

> A function $$k(x, x')$$ is a kernel if it is symmetric in its arguments and is positive semi-definite

By positive semi-definite we mean that for all $$f$$ in some (square-normed function) space we have 
<div>
    \begin{equation}
    \int k(x, x') f(x)f(x') \, \mathrm{d}\mu(x) \mathrm{d}\mu(x') \geq 0.
    \end{equation}
</div>

####  Isotropy and Stationarity

- We say that a kernel is isotropic if it is a function of $$\|\mathbf{x} - \mathbf{x}'\|$$. That is, the kernel is invariant with respect to rotations.

- We say that a kernel is stationary if it is a function of $$\mathbf{x} - \mathbf{x}'$$. That is, the kernel is invariant with respect to translations.

You may have already heard of stationarity in the context of stochastic processes. A GP is weakly stationary if its mean function is constant and its kernel is stationary in the sense defined above. It is strictly stationary if all of its finite dimensional distributions are invariant to translation.


#### Examples of Kernels
We list a few common examples of kernels. For more information see [*Gaussian Processes for Machine Learning*](http://www.gaussianprocess.org/gpml/). 

- Radial Basis Function (RBF):  $$k(\mathbf{x}, \mathbf{x'}) = \mathrm{exp}\left(- \frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2 l^2}\right)$$

- Ornstein-Uhlenbeck (OU):  $$k(\mathbf{x}, \mathbf{x'}) = \mathrm{exp}\left(- \frac{\|\mathbf{x} - \mathbf{x}'\|}{l}\right)$$

- Periodic (Per):  $$k(\mathbf{x}, \mathbf{x'}) = \mathrm{exp}\left(- \frac{2}{l^2} \mathrm{sin}^2\left(\frac12\|\mathbf{x} - \mathbf{x}'\|^2 \right)\right)$$

- Rational Quadratic (RQ):  $$\left(1 + \frac{r^2}{2\alpha l^2} \right) ^ {-\alpha}$$

Note that all of these examples are stationary. The RBF and Per kernels both give rise to processes that are infinitely differentiable (in mean square sense). 

![kernel_examples](/images/Gaussian-Process-Regression/PriorDraws.png)


You may have noticed that there are additional parameters in these kernel functions. These are known as hyperparameters and they form part of the model selection problem for GPs, we will talk about them later.


#### Kernel Algebra
It is straightforward to show that the sum of two kernels is also a kernel and the product of two kernels is again a kernel. This means that it is possible to build structure in your models hierarchically by composing structure from various kernels. For more on this, see the first few chapters of [David Duvenaud's PhD thesis](https://www.cs.toronto.edu/~duvenaud/thesis.pdf).

## Regression
We will now explore how, given some training data $$\mathcal{D} = (X, \mathbf{y})$$, we can make predictions at test points $$X_*$$ for the problem defined at the beginning
<div>
    \begin{equation}
    y = f(\mathbf{x}) + \varepsilon \quad \mathrm{with} \quad \varepsilon \sim \mathcal{N}(0, \sigma^2).
    \end{equation}
</div>
We will follow the convention of Rasmussen and Williams and stack our training $$\mathbf{x}$$ *horizontally* so that each columns of $$X$$ is a training data point. Note that this is the transpose of how you will often see the design matrix!

Denote the stacked predicted values for $$f$$ as $$\mathbf{f}_*$$, then by the marginalisation property we have the following joint distribution of the training and test data
<div>
    \begin{align}
     \begin{bmatrix}
             \mathbf{y} \\
             \mathbf{f}_* \\
     \end{bmatrix}
      & \sim \mathcal{N}\left(0,
       \begin{bmatrix}
       K(X, X) + \sigma^2 I &  K(X, X_*)\\
       K(X_*, X) & K(X_*, X_*)
       \end{bmatrix}
       \right).
    \end{align}
</div>
Now all we need to do to get the posterior predictive distribution for $$\mathbf{f}_*$$ is to use the Guassian conditioning formula to arrive at
<div>
    \begin{equation}
    \mathbf{f}_* | \mathbf{y}, X, X_*
    \sim 
    \mathcal{N} \left(
        K(X_*, X)\left[K(X, X) + \sigma^2 I\right]^{-1}\mathbf{y}, \,\,
        K(X_*, X_*) - K(X_*, X)\left[K(X, X) + \sigma^2 I\right]^{-1}K(X, X_*)
    \right).
    \end{equation}
</div>

In the Bayesian formulation we choose our predictions to minimize the expected value of some loss function, with the expectation taken against the distribution of the prediction points. Typically, one would choose a squared error loss function, which would result in predicting the mean of this distribution
<div>
    \begin{equation}
    \mathbf{y}_* = K(X_*, X)\left[K(X, X) + \sigma^2 I\right]^{-1}\mathbf{y}.
    \end{equation}
</div>

The final point to note here is that the prediction at a single test point is a linear combination of kernel evaluations at the test point and on the training set, reminiscent of the representor theorem.

## Model Selection
#### Hyperparameters
As stated above, kernel covariance functions often come in families parameterised by some vector of hyperparameters $$\mathbf{\theta}$$
<div>
    \begin{equation}
    k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}, \mathbf{x}'; \mathbf{\theta}).
    \end{equation}
</div>
In the case of the RBF kernel, this was the length scale $$l$$. As we will see, the choice of hyperparameters forms a key part of the model selection process for GPs.


#### Bayesian Model Selection
Given some class of hypotheses for our problem $$\{\mathcal{H}_i\}$$, we find our distribution for $$\mathbf{y}$$ as

<div>
    \begin{equation}
    \mathsf{P}(\mathbf{y} | X, \mathbf{\theta}, \mathcal{H}_i)
    = \int \mathsf{P}(\mathbf{y} | X, \mathbf{\theta}, \mathcal{H}_i, \mathbf{f}) \mathsf{P}(\mathbf{f} | \mathbf{\theta}, \mathcal{H}_i) \, \mathrm{d}\mathbf{f}
    \end{equation}
</div>
where we have integrated over the function values $$\mathbf{f}$$ according to the measure given by the GP. This term is known as the marginal likelihood or the model evidence.

Ideally, we would proceed by first integrating out the hyperparameters
<div>
    \begin{equation}
    \mathsf{P}(\mathbf{y} | X, \mathcal{H}_i)
    = \int \mathsf{P}(\mathbf{y} | X, \mathbf{\theta}, \mathcal{H}_i) \mathsf{P}(\mathbf{\theta} | \mathcal{H}_i) \, \mathrm{d}\mathbf{\theta}
    \end{equation}
</div>
and then find the distribution for each hypothesis using Bayes' rule
<div>
    \begin{equation}
    \mathsf{P}(\mathcal{H}_i | \mathbf{y}, X)
    = \frac{\mathsf{P}(\mathbf{y} | X, \mathcal{H}_i) \mathsf{P}(\mathcal{H}_i)}{\mathsf{P}(\mathbf{y} | X)}
    \end{equation}
</div>
with $$\mathsf{P}(\mathbf{y}|X) = \sum_i \mathsf{P}(\mathbf{y}, X, \mathcal{H}_i)\mathsf{P}(\mathcal{H}_i)$$.

Unfortunately, the integral over the hyperparameters is often intractable. In general, people sidestep this issue by approximating the integral by Laplace's method. In turn, this means maximising the log marginal likelihood w.r.t the hyperparameters
<div>
    \begin{equation}
    l(\mathbf{\theta}) = \mathrm{log}(\mathsf{P}(\mathbf{y} | X, \mathbf{\theta}, \mathcal{H}_i)).
    \end{equation}
</div>
One would then use $$\theta^* = \underset{\mathbf{\theta}}{\mathrm{argmax}} \, l$$ in the steps that followed. 

Finally, one should note that at the training points
<div>
    \begin{equation}
    \mathbf{y}| X, \mathbf{\theta}, \mathcal{H}_i \sim \mathcal{N}(0, K(X, X) + \sigma^2 I)
    \end{equation}
</div>
so you don't need to do any fiddling to find $$l(\mathbf{\theta})$$.


<!--
## An Example Application

## Where to look next
- Duvenaud paper
- Duvenaud's thesis
- GPML book
- etc.
-->

## Acknowledgements
These notes borrow heavily from the bible on GPs, *Gaussian Processes for Machine Learning* by Rasmussen and Williams, see [here](http://www.gaussianprocess.org/gpml/).

Any errors are mine. (Obviously.)
