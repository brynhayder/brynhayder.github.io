---
layout: post
title:  "Deriving the Neural Tangent Kernel"
date:   2019-04-2 00:00:00 +0000
categories: jekyll update
usemathjax: true
comments: true
---
## Introduction
I recently came across the paper [Neural Tangent Kernel: Covergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572) published at NeurIPS 2018. 
I was super excited by this paper and I think it is a great execution of a very natural idea.
I thought I would try to reproduce the ideas of the paper myself. 
Hence, this post is a reconstruction of the general idea of the paper in a way that seems natural to me.
\[
\newcommand{\params}{\vec{\theta}}
\newcommand{\net}{f^\params}
\newcommand{\F}{\mathcal{F}}
\newcommand{\X}{\mathcal{X}}
\]

## Roadmap
We have a neural network that we will train using gradient descent on some loss function.
We are interested in characterising the behaviour of the optimisation, in particular we want to know what the cost does under gradient descent.

We will explore this question directly and find an object (a kernel on a space of functions in which our network lives), the neural tangent kernel (NTK), and see that our optimisation procedure will result in a network that performs well on our task if the NTK satisfies a certain condition (is positive definite).
The central contributions of the paper are to provide the insight into the dynamics of the cost and prove the aforementioned property.
I'm not going to go through the proofs (at least in the first version of this post) but I will hopefully give some insight into the general idea.

## Networks as Functions
Neural networks are a special class of parameterised functions. 
As we vary the parameters $\params$, the network function $\net$ varies on a family of functions defined by the architecture. 
The choice of $\params$ determines the behaviour of the network and we will typically choose $\params \in \R^P$ to (approximately) minimise some _cost_ function $c(\params) = C[\net]$ (here we say cost rather than loss for consistency with the paper).
Note that we have defined the cost in terms of a functional $C$, we will be interested in the behaviour of $C$ on the space of functions $\F = \\{\net | \params \in \R^P\\}$ attainable by our network. (There are some technical requirements for $\F$ that we will ignore.)
For the purposes of this post, a functional is just a mapping from functions (so from a space like $\F$) to $\R$.
If we have a functional $J$ and a function $g$ then $J$ evaluated on $g$ is a real number written $J[g]$ with square brackets.

## The Cost Functional
We have made a change in perspective from thinking of our network in terms of parameter space to thinking of it living in the function space $\F$.
Correspondingly, we are going to study the dynamics of $C$ on $\F$ rather than of $\params$ on $\R^P$.

Suppose we perform gradient descent on the parameters, so that they vary on the trajectory
\[
\diff{\params}{t} = - \pdiff{C[\net]}{\params},
\]
where $t$ plays the role of "time".
From now on we will have $\params = \params(t)$, but leave time time dependence implicit for notational ease.
We want to analyse the dynamics of the cost in this scheme, and in order to do that we will need the functional derivative of $C$.

## Functional Derivatives
We want to measure the variation of a functional as we move a function in our function space.
To do this we will define a notion of a derivative analogous to the $f: \R \to \R$ case.

Given a (suitably nice) space of functions $\X \to \R^k$ which we'll call $\mathcal{G}$, $g \in \mathcal{G}$ and a functional $J: \mathcal{G} \to \R$, the functional derivative of $J$ at $g$ is written
\[
    \fdiff{J}{g}.
\]
This is the element in $\mathcal{G}$ such that for _any_ $\phi \in \mathcal{G}$
\[
   \int_\X \fdiff{J}{g}(x)^{T} \phi(x) \d x = \lim_{\epsilon \to 0} \frac{J[g + \epsilon \phi] - J[g]}{\epsilon}.
\]
(This exists and is unique by [Riesz-Markov-Kakutani Representation Theorem](https://en.wikipedia.org/wiki/Riesz-Markov-Kakutani_representation_theorem).)
You can think of this as the change in $J$ from moving infinitesimally at $g$ in the direction of $\phi$, analogous to the familiar fact that the directional derivative of $f$ in the direction of $\vec{n}$ is $\vec{n}\cdot\nabla f$.

### How Does Changing the Parameters Change the Cost?
Suppose we vary the parameters of our network $\params \to \params + \epsilon \vec{\eta}$, how does the cost change?
<div>
\begin{align*}
   \lim_{\epsilon \to 0} \frac{C[f^{\params + \epsilon \eta}] - C[\net]}{\epsilon} 
   &= \lim_{\epsilon \to 0} \frac{C[\net + \epsilon \eta \cdot \pdiff{\net}{\params} + O(\epsilon^2)] - C[\net]}{\epsilon} \\
   &= \sum_{i,j} \eta_i \int_\X \fdiff{C}{\net}(x)_j \pdiff{\net_j}{\theta_i}(x) \d x
\end{align*}
</div>
Note that we have just calculated $\vec{\eta} \cdot \pdiff{C[\net]}{\params}$.
We can calculate $\diff{C[\net]}{t}$ (recalling that $\params = \params(t)$) by setting $\eta = \diff{\params}{t}$.

## The Neural Tangent Kernel
We can now turn back to our original question: what are the dynamics of the cost under gradient descent?
First recall that under gradient descent we have
\[
    \diff{\params}{t} = - \pdiff{C[\net]}{\params}.
\]
Now we can calculate

<div>
\begin{align*}
   \diff{C[\net]}{t} 
   &= \sum_{i, j} \diff{\theta_j}{t} \int_\X \fdiff{C}{\net}(x)_i\pdiff{\net}{\theta_j}(x)_i \d x \\
   &= -\sum_{i, j, k} \int_\X \fdiff{C}{\net}(x')_k  \pdiff{\net}{\theta_i}(x')_k \d x' \int_\X \fdiff{C}{\net}(x)_j\pdiff{\net}{\theta_i}(x)_j \d x \\
   &= - \sum_{j, k = 1}^{M} \int_\X \fdiff{C}{\net}(x)_j \left( \sum_{i=1}^P \pdiff{\net}{\theta_i}(x)_j \pdiff{\net}{\theta_i}(x')_k\right) \fdiff{C}{\net}(x')_k \d x \d x' \\
   &= - \int_\X \fdiff{C}{\net}(x)^T K_{\text{NTK}}(x, x') \fdiff{C}{\net}(x') \d x \d x' \\
   &= - \left\lVert{} \fdiff{C}{\net} \right\rVert{}^2_{K_\text{NTK}}.
\end{align*}
</div>
Where we have introduced the _neural tangent kernel_
\[
    K_{\text{NTK}} = \sum_{i=1}^{P} \pdiff{\net}{\theta_i} \otimes \pdiff{\net}{\theta_i}
\]
and the final line is the correspondingly induced norm.

We see that if this kernel is positive definite, then the cost will converge to a global optima on $\F$.
In the paper it is shown that at (Gaussian) initialisation the kernel is indeed positive definite in the infinite width limit, and, also in the infinite width limit, that it remains approximately constant throughout training.

