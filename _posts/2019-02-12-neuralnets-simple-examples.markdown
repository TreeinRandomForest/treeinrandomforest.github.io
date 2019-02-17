---
layout: post
title:  "Neural Networks on Toy Examples: An Aid to Understanding"
date:   2019-02-12
categories: [deep-learning]
tags: [deep-learning]
mathjax: true
---

Continuing our trend of understanding neural networks from first principles, we would like to get an intuition for how exactly a simple network learns to predict. We'll look at two simple problems - a binary classification problem and a 1-dimensional regression problem.

# Classification

Consider the simple dataset shown below. Each point is described by a 2-dimensional coordinate $(x,y)$. Points in the red center are labeled as class 0 and points in the green ring are labeled as class 1. We would like to train a **classifier** i.e. a statistical model that takes the $(x,y)$ point as an input and predict the probability $p$ of belonging to class 1. We can convert these probabilities $p$ to discrete classes 0 and 1 by picking a threshold $\delta$. If $p > \delta$, we predict the class to be 1 and if $p \leq \delta$, we predict the class to be 0.
