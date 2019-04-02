---
layout: post
title:  "Deriving the Neural Tangent Kernel"
date:   2019-04-1 00:00:00 +0000
categories: jekyll update
usemathjax: true
comments: true
---
\[
\newcommand{\params}{\vec{\theta}}
\newcommand{\net}{f_\params}
\newcommand{\F}{\mathcal{F}}
\newcommand{\X}{\mathcal{X}}
\]

## Introduction
I recently came across the paper [Neural Tangent Kernel: Covergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572) published at NeurIPS 2018. I was super excited by this paper and I think it is a great execution of a very natural idea. However, although the paper is structured logically, I struggled to get a feel for how I might have arrived at their ideas myself. This post is a reconstruction of the ideas of the paper in a way that seems natural to me.

## Roadmap
We have a neural network that we will train using gradient descent on some cost (loss) function.
We are interested in characterising the behaviour of the optimisation, in particular we want to know what the cost does under gradient descent.

We will explore this question directly and find an object (a kernel on a space of functions in which our network lives), the neural tangent kernel (NTK), and see that our optimisation procedure will result in a network that performs well on our task if the NTK satisfies a certain condition (is positive definite). The central contributions of the paper are to provide the insight into the dynamics of the cost and prove the aforementioned property. I'm not going to go through the proofs (at least in the first version of this post) but I will hopefully give you some insight into the general idea.

## The General Idea
### Networks as Functions
Neural networks are a special class of parameterised functions. 
As we vary the parameters $\params$, the network function $\net$ varies on a family of functions defined by the architecture. 
The choice of $\params$ determines the behaviour of the network and we will typically choose $\params \in \R^P$ to (approximately) minimise some _cost_ function $c(\params) = C[\net]$ (here we say cost rather than loss for consistency with the paper).
Note that we have defined the cost in terms of a functional $C$, we will be interested in the behaviour of $C$ on the space of functions $\F = \\{\net | \params \in \R^P\\}$ attainable by our network. (There are some technical requirements for $\F$ that we will ignore.)
For the purposes of this post, a functional is just a mapping from functions (so from a space like $\F$) to $\R$.
If we have a functional $J$ and a function $g$ then $J$ evaluated on $g$ is a real number written $J[g]$ with square brackets.

### The Cost Functional
We have made a change in perspective from thinking of our network in terms of parameter space to thinking of it living in the function space $\F$.
Correspondingly, we are going to study the dynamics of $C$ on $\F$ rather than of $\params$ on $\R^P$.

Suppose we perform gradient descent on the parameters, so that they vary on the trajectory
\[
\diff{\params}{t} = - \pdiff{C[\net]}{\params},
\]
where $t$ plays the role of "time".
From now on we will have $\params = \params(t)$, but leave time time dependence implicit for notational ease.
We want to analyse the dynamics of the cost in this scheme, that is
\[
    \diff{C[\net]}{t} = \pdiff{C[\net]}{\params} \diff{\params}{t}.
\]
In order to calculate the first term, we will need the functional derivative.

### Functional Derivatives
We want to measure the variation of a functional as we move a function in our function space.
To do this we will define a notion of a derivative analogous to the $f: \R \to \R$ case.

Given a (suitably nice) space of functions $\X \to \R^k$, $\mathcal{G}$, $g \in \mathcal{G}$ and a functional $J: \mathcal{G} \to \R$, the functional derivative of $J$ at $g$ is written
\[
    \fdiff{J}{g}.
\]
This is the element in $\mathcal{G}$ that for _any_ $\phi \in \mathcal{G}$ is such that
\[
   \int_\X \fdiff{J}{g}(x)^{T} \phi(x) \d x = \lim_{\epsilon \to 0} \frac{J[g + \epsilon \phi] - J[g]}{\epsilon}.
\]
(This exists and is unique by [Riesz-Markov-Kakutani Representation Theorem](https://en.wikipedia.org/wiki/Riesz-Markov-Kakutani_representation_theorem) and a change of variables.)
You can think of this as the change in $J$ from moving infinitesimally in the direction of $\phi$, analogous to the familiar fact that the directional derivative of $f$ in the direction of $\vec{n}$ is $\vec{n}\cdot\nabla f$.





### Dynamics of Cost Under Gradient Descent
### The Neural Tangent Kernel








## Contents
1. [Introduction](#introduction)
2. Motivating Example
3. A General Construction(#general-construction)
