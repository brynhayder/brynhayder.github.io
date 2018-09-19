---
layout: post
title:  "Some Thoughts on the Central Limit Theorem"
date:   2018-09-18 00:00:00 +0000
categories: jekyll update
usemathjax: true
---

{% include mathjax.html %}

The Central Limit Theorem (CLT) is a famous and surprising theorem in probability.

>Let $$X_1, X_2, \dots$$ be a sequence of iid random variables with $$\E{}[X_i] = \mu$$ and finite and non-zero variance $$\Var{}[X_i] = \sigma^2 < \infty$$. Define $$\bar{X}_n = \frac1n \sum_{i=1}^n X_i$$. Let $$G_n(x)$$ denote the cdf of $$\sqrt{n}(\bar{X}_n - \mu)/\sigma$$. Then $$\forall x \in \R$$,
><div>
    \[
        \lim_{n\to\infty} G_n(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-y^2/2}\d{}y.
    \]
></div>
>That is, $$\sqrt{n}(\bar{X}_n - \mu)/\sigma$$ converges in distribution to the standard normal.

This says that a centred sample average from a population converges to a normal distribution as the sample becomes infinitely large. Note how little we assume about the distribution of the things we are averaging! This is one of the reasons why the normal distribution is so common in statistics. Intuition behind this result is not easy to come by, so I'll note two thoughts that might be helpful.

##  Stability
The theorem assumes that $$X_i$$ are iid with finite mean and variance. If the sequence of sample averages $$\bar{X}_n$$ converges to anything, we should expect the limiting distribution to a [stable distribution](https://en.wikipedia.org/wiki/Stable_distribution) with finite variance.

> $$X$$ is a stable distribution if, for any $$X_1, X_2 \sim X$$ and constants $$a, b > 0$$ there exists constants $$c>0, d$$ with
><div>
    \[
    aX_1 + bX_2 \sim cX + d.
    \]
></div>

The normal distribution is the *only* stable distribution with finite variance. So if the sample averages converge, we might expect them to converge to a normal distribution.

(still need to show why limit should be stable, it is clear that it is stable if a = b = 1, but what about otherwise?)

## Entropy
The normal distribution is the maximum entropy distribution of fixed mean and variance.

(still need to show why the limit would be maximum entropy, is entropy of a sum bigger than entropy of the summands?)

## Acknowledgements
I don't claim any originality of anything here.

- I took the statement of the CLT from Statistical Inference (2e) by Casella and Berger (which is a great book).
- I adapted the stability definition from the linked wikipedia page.

