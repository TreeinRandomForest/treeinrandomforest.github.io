---
layout: post
title:  "A detailed look at backpropagation in feedforward neural networks"
date:   2018-10-30
categories: [deep-learning]
tags: [deep-learning]
mathjax: true
---

The last few years have shown an enormous rise in the use of neural networks for supervised-learning tasks. This growth has been driven by multiple factors - exponentially more labeled data, faster and cheaper GPUs (graphics processing units), as well as better understanding of neural network training.

At the same time, the core training algorithm used to train neural networks is still backpropagation and gradient descent. While there are many excellent frameworks like TensorFlow and PyTorch that takes care of the details for the modern machine learning practitioner, it is crucial to understand what they do under the hood. The first step in that journey is understanding what backpropagation actually is.

# Global View of the Training Process

One can view a neural network as a black box that maps certain input vectors $\vec{x}$ to output vectors $\vec{y}$. More formally, the neural network is a function, $f$:

$$\vec{y} = f(\vec{x})$$

$f$ also depends on several underlying parameters, also known as weights, denoted by $\vec{w}$:

$$\vec{y} = f(\vec{x}; \vec{w})$$

The situation isn't unlike linear regression where the output $y$:

$$y = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n = \vec{w}.\vec{x}$$

is a function of the input $\vec{x}$ with some parameters $\vec{w}$.

The central problem then is the discovery of the correct $\vec{w}$. What does that even mean? Well, given a dataset with input vectors and the corresponding outputs (labels), one uses $f$ with random weights $\vec{w}$ to make predictions, measures the deviation between the predictions and labels and tweaks the weights to minimize the deviation.

As an example, one commonly used measure of deviation is **mean-squared error** which is especially useful for regression problems. Another one commonly used is **cross-entropy** (or **negative-log-likelihood**) which is used for classification problems. For simplicity, we'll use mean-squared error below but the discussion is minimally changed if one uses a different error metric.

The mean-squared error measures the deviation between the label $y$ and the prediction $\hat{y}$ as:

$$error = \frac{(\hat{y}-y)^2}{2}$$

Dividing by 2 is purely a convention that will become clear later on. If the label and prediction agree, this error is 0 and the more they disagree, the higher the error. For a dataset, one would just average the errors:

$$C = \frac{1}{n} \Sigma_{i=0}^{n} \frac{(\hat{y}_i-y_i)^2}{2}$$

where we introduced the symbol $C$ which stands for **cost**. The terms **cost**, **loss**, **error**, **deviance** are often used interchangeably but we'll stick to cost from now on. $n$ is the number of examples in the dataset and the symbol $\Sigma$ (capital "Sigma") denotes a sum over all the errors.

Since the predictions are functions of $\vec{w}$, $C$ actually depends on $\vec{w}$ as well:

$$C[\vec{w}] = \frac{1}{n} \Sigma_{i=0}^{n} \frac{(f(\vec{x}_i; \vec{w})-y_i)^2}{2}$$

where we made $C[\vec{w}]$ denotes $C$'s dependence on $\vec{w}$ .

We would now pick some random $\vec{w}$, make predictions $f(\vec{x}_{i}; \vec{w})$ and compute the cost $C[\vec{w}]$. Our next task is to tweak $\vec{w}$ and repeat the procedure so that $C[\vec{w}]$ decreases. Our end goal is to minimize the cost, $C[\vec{w}]$ and the set of weights $\vec{w}$ that would do that would define our final model.

The big question here is two-fold:
* How should we choose the initial weights, $\vec{w}_0$ (the "0" denotes "initial")?

* Once we compute $C[\vec{w}\_{i}]$ with a given $\vec{w}\_{i}$, how should we choose the next $\vec{w}\_{i+1}$ so that **in the long run**, we decrease $C[\vec{w}]$?

There's some new notation above so let's take a moment to clearly define what we mean:

Think of the process of updating the weights $\vec{w}$ as a process that occurs once every second. At time $t=0$, we start with a randomly generated $\vec{w}\_0$. At time $t$, the weights will be $\vec{w}\_{t}$. We want a rule to go from $\vec{w}\_{i}$ to $\vec{w}\_{i+1}$.

One way to minimize $C[\vec{w}]$ is a "greedy" approach. Let's look at a simple example that is one-dimensional i.e. there's only one element in $\vec{w}$ called $w$, which is a real number:

!PICTURE

This is a nice situation where there is exactly one minimum at $w\_{\*}$. Let's suppose, we are $w\_{R}$ ($R$ denotes "right" and $L$ denotes "left"). We know we need to move to the left or in the negative direction. We also know the slope of the cost curve is positive ("pointing up") at $w\_R$. On the other, suppose we are $w_L$. We need to move to the right or in the positive direction while the slope is negative ("pointing down") at $w_L$.

In other words:
* When the slope of the cost function is positive, we need to move the weight to the negative direction i.e. decrease the weight.
* When the slope is negative, we need to move the weight in the positive direction i.e. increase the weight.

Mathematically,

$$w_{i+1} = w_{i} - \text{(something)} \text{(sign of slope)}$$

where $\text{something}$ is a positive number (so it won't change the sign of the term) which signifies the magnitude of the change in $w_{i}$.

When the slope is positive, we get:

$w_{i+1} = w_{i} - \text{(positive)} \text{(positive)} = w_{i} - \text{positive}$

i.e. $w_{i+1} < w_{i}$ so we moved in the negative direction.

When the slope is negative, we get:

$w_{i+1} = w_{i} - \text{(positive)} \text{(negative)} = w_{i} + \text{positive}$

i.e. $w_{i+1} > w_{i}$ so we moved in the positive direction.

We still need to decide what $\text{something}$ is. It's usually taken to be proportional to the magnitude of the slope:

$$w_{i+1} = w_{i} - \eta \mid{\frac{dC[w_i]}{dw}}\mid \text{(sign of slope)}$$

where $\eta$ is a constant of proportionality called the **learning rate**, $\mid\frac{dC[w_i]}{dw}\mid$ is the absolute value of the slope (or derivative) at the point $w_i$. We don't need to separate out the magnitude of the slope and the sign of the slope and we can simply write:

$$w_{i+1} = w_{i} - \eta \frac{dC[w_i]}{dw}$$

In the discussion below, we'll assume mean-squared error and exactly one data point.

## Backpropagation I - linear activations


We will be working with a very simple network architecture in this section. The architecture is shown in figure X. 

There is an input node taking a vector $x_0$, two internal nodes with values $x_1$ and $x_2$ respectively and an output node with value $x_3$ (also denoted as $\hat{y}$).

There are three weights: $w_{01}$, $w_{12}$, and $w_{23}$ respectively. You should read $w_{ij}$ as the weight "transforming the value at node i to the value at node j".

More precisely, the forward propagation in this architecture is:

$x_1 = w_{01} x_0$

$x_2 = w_{12} x_1$

$\hat{y} \equiv x_3 = w_{23} x_2$

We can substitude the values iteratively to get:

$x_3 = w_{23} x_2 = w_{23} w_{12} x_1 = w_{23} w_{12} w_{01} x_0$

There is something silly going on here. Why would we have all these weights when we can define a new weight, say $w_{c} \equiv w_{23} w_{12} w_{01}$ (the "c" stands for combined) and define the following architecture going straight from the input to the output

$x_3 = w_c x_0$

You are absolutely right if you made that observation and it's a very important point. Just combining these "linear" nodes doesn't do anything. We need to add non-linearities to be able to learn arbitrarily complicated functions $f$. But for now, bear with me since this sections lays the groundwork for the next section where we introduce non-linearities into the network architecture.

Going back to our network, to implement backpropagation, we need to calculate all the derivatives of the cost function with respect to the weights.

The cost is

$C[\vec{w}] = C[w_{01}, w_{12}, w_{23}] = \frac{(x_3 - y)^2}{2}$

where $y$ is the actual value/label in our dataset and is a constant (independent of $\vec{w})$.

Expanding, we get

$C[\vec{w}] = \frac{1}{2} (w_{23} w_{12} w_{01} x_0 - y)^2$

We can now calculate the derivatives by using some basic calculus:

$\frac{\partial C}{\partial w_{23}} = \frac{1}{2} 2 (x_3 - y) \frac{\partial x_3}{\partial w_{23}}\\
= (x_3 - y) w_{12} w_{01} x_0\\
$

$\frac{\partial C}{\partial w_{12}} = (x_3-y) w_{23} w_{01} x_0$

$\frac{\partial C}{\partial w_{01}} = (x_3-y) w_{23} w_{12} x_0$

We see a couple of patterns here:

* There is one derivative for each weight.
* Each derivative is proportional to $(x_3 - y)$ or the deviation between the prediction and the target/label. If the deviation is 0, then all the derivatives are 0 and there are no corrections to the weights during gradient descent, as should be the case.
* One can think of two "chains" - a forward chain and a backward chain.
	* Forward chains look like:
		* $x_0$
		* $w_{01} x_0$ (same as $x_1$)
		* $w_{12} w_{01} x_0$ (same as $x_2$)
		* $w_{23} w_{12} w_{01} x_0$ (same as $x_3$)
	* Backward chains look like:
		* $(x_3 - y)$
		* $w_{23} (x_3 - y)$
		* $w_{12} w_{23} (x_3 - y)$
		* $w_{01} w_{12} w_{23} (x_3 - y)$
	* Both forward and backward chains show up in the derivatives.

To make the above point clearer, let's rewrite the derivatives with the weights in order from left to right and any **missing weight is colored in red**.

$\frac{\partial C}{\partial w_{23}} = (x_3 - y) {\color{red} {w_{23}}} w_{12} w_{01} x_0\\
$

$\frac{\partial C}{\partial w_{12}} = (x_3-y) w_{23} {\color{red} {w_{12}}} w_{01} x_0$

$\frac{\partial C}{\partial w_{01}} = (x_3-y) w_{23} w_{12} {\color{red} {w_{01}}} x_0$


Let's introduce some more notation for the backward chains. We will use the Greek symbol "delta", $\delta$ since it stands for the English "d" for "difference" or "deviance" and is a conventional symbol used for $x_3-y$ which measures the error.

Define:

$\delta_0 = x_3-y$

$\delta_1 = w_{23} (x_3 - y)$

$\delta_2 = w_{12} w_{23} (x_3 - y)$

$\delta_3 = w_{01} w_{12} w_{23} (x_3 - y)$

Using these new symbols, we can write the derivatives in a very simple manner:

$\frac{\partial C}{\partial w_{23}} = \underbrace{(x_3 - y)}\_{\delta_0} {\color{red} {w_{23}}} \underbrace{w_{12} w_{01} x_0}\_{x_2} = \delta_0 x_2\\
$

$\frac{\partial C}{\partial w_{12}} = \underbrace{(x_3-y) w_{23}}\_{\delta_1} {\color{red} {w_{12}}} \underbrace{w_{01} x_0}\_{x_1} = \delta_1 x_1$

$\frac{\partial C}{\partial w_{01}} = \underbrace{(x_3-y) w_{23} w_{12}}\_{\delta_2} {\color{red} {w_{01}}} \underbrace{x_0}\_{x_0} = \delta_2 x_0$

In other words, we always get the combination $\delta_{A} x_{B}$ where $A+B=2$ and $B$ (subscript of $x$) is the source of the weight with respect to which we are taking the derivative. So,

$\frac{\partial C}{\partial w_{i,i+1}} = \delta_{2-i} x_i$

There is no magic about the "2". It is the number of hidden layers.

The main advantage here is that as one makes predictions, one has to calculate the forward chains - $x_1, x_2, x_3$ and once one calculates the deviation $(x_3 - y)$, calculating the backward chains is just an iterative multiplication by the weights but going in the reverse direction. So far so good but this is still just a linear neural network with nothing interesting going on. Let's add non-linearities into the mix.

## Backpropagation II - non-linear activations

We will still maintain the same architecture consisting of a single node for each of the 4 layers. The new addition is an extra operation at each node except for the input node.

A **linear** function $g(\vec{x})$ is any function with the following property:

$g(a\vec{x} + b\vec{y}) = ag(\vec{x}) + bg(\vec{y})$

where $a, b$ are constant real numbers and $\vec{x}, \vec{y}$ are $n$-dimensional vectors.

Linear functions can be very useful and the whole subject of **linear algebra** studies vector spaces and linear functions between them. But, most real systems in the world are not linear. We want our neural network to be able to learn arbitrary non-linear functions. To enable this, we need to add some non-linear function at various points.

At each node, we now define two numbers:

$p_i$ is the value **before** the non-linear function is applied (p is for "pre")

$q_i$ is the value **after** the non-linear function is applied (q since it comes after p)

and $i$ denotes which layer/node we are talking about. The non-linear function, also commonly known as the **activation function** or just **activation** is denoted by $\sigma_i$. This can potentially be different at every single layer/node.

Forward propagation is now modified where every equation in REFERENCE is now split into two parts as shown below:

$p_0 = x_0 \text(input) \rightarrow q_0 = \sigma_0(p_0)$

$p_1 = w_{01} q_0 \rightarrow q_1 = \sigma_1(p_1)$

$p_2 = w_{12} q_1 \rightarrow q_2 = \sigma_2(p_2)$

$p_3 = w_{23} q_2 \rightarrow q_3 = \sigma_3(p_3) \text(output)$

The input $x_0$ is now denoted by $p_0$ for notational consistency. $p_0$ is now fed to the activation function $\sigma_0$ to get $q_0$. $q_0$ is now the input to the second layer. This process repeats till we get $q_3$ at the end which is the output of the model.

As a special case, consider $\sigma_i(x) = id(x) = x$ where $id$ denotes the identity function that maps every input to itself - $id(x) = x$. In that case, $p_i = q_i$ and we get our old linear neural network back.

To be more explicity about the output $q_3$'s dependence on the weights, we can combine equations REFERENCE:

$q_3 = \sigma_3(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0))))$

As before, we compute the cost function to compare the output to the label:

$C[\vec{w}] = C[w_{01}, w_{12}, w_{23}] = \frac{(q_3-y)^2}{2}$

or more explicitly:

$C[\vec{w}] = \frac{(\sigma_3(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0))))-y)^2}{2}$

Let's take a step back and realize that we really haven't done anything very different. All we did was add 4 activations to our neural network, compute the output and evaluate the cost to see how well we did. As before what we really care about are the derivatives of the cost with respect to the weights so we can do gradient descent.

Using calculus, we can explicitly compute the derivatives (and write all the terms explicitly for clarity):

$\frac{\partial C}{\partial w_{23}} = (q_3-y) \sigma_3'(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))) \sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))$

$\frac{\partial C}{\partial w_{12}} = (q_3-y) \sigma_3'(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))) w_{23} \sigma_2'(w_{12}\sigma_1(w_{01}\sigma_0(p_0))) \sigma_1(w_{01}\sigma_0(p_0))$

$\frac{\partial C}{\partial w_{01}} = (q_3-y) \sigma_3'(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))) w_{23} \sigma_2'(w_{12}\sigma_1(w_{01}\sigma_0(p_0))) w_{12} \sigma_1'(w_{01}\sigma_0(p_0)) \sigma_0(p_0) $

This is great! We can already do some sanity checks and make a few observations:

Sanity checks:
* All the derivatives are proportional to $(q_3-y)$. In other words, if your prediction is exactly equal to the label, there are no derivatives and hence no gradient descent to do which is precisely what one would expect.
* If we replace all the activations by the identity function, $id(x) = x$ with $id'(x) = 1$, then we can replace the derivatives with 1, all the $q_i = p_i = x_i$ and we recover the derivatives for Case I (REFERENCE).

Observations:
* We still get both forward chains and backward chains:
	* Forward chains now look like:
		* ${\color{blue} {p_0}}$
		* $\sigma_0(p_0) = {\color{blue} {q_0}}$
		* $w_{01} \sigma_0(p_0) = {\color{blue} {p_1}}$
		* $\sigma_1(w_{01} \sigma_0(p_0)) = {\color{blue} {q_1}}$
		* $w_{12} \sigma_1(w_{01} \sigma_0(p_0)) = {\color{blue} {p_2}}$
		* $\sigma_2(w_{12} \sigma_1(w_{01} \sigma_0(p_0))) = {\color{blue} {q_2}}$
		* $w_{23} \sigma_2(w_{12} \sigma_1(w_{01} \sigma_0(p_0))) = {\color{blue} {p_3}}$
		* $\sigma_3(w_{23} \sigma_2(w_{12} \sigma_1(w_{01} \sigma_0(p_0)))) = {\color{blue} {q_3}}$
		* These are just the terms forward propagation generates.
	* Backward chains:
		* $(q_3-y)$
		* $(q_3 - y) \sigma_3'(p_3)$
		* $(q_3 - y) \sigma_3'(p_3) w_{23}$
		* $(q_3 - y) \sigma_3'(p_3) w_{23} \sigma_2'(p_2)$
		* $(q_3 - y) \sigma_3'(p_3) w_{23} \sigma_2'(p_2) w_{12}$
		* $(q_3 - y) \sigma_3'(p_3) w_{23} \sigma_2'(p_2) w_{12} \sigma_1'(p_1)$
		* $(q_3 - y) \sigma_3'(p_3) w_{23} \sigma_2'(p_2) w_{12} \sigma_1'(p_1) w_{01}$
		* $(q_3 - y) \sigma_3'(p_3) w_{23} \sigma_2'(p_2) w_{12} \sigma_1'(p_1) w_{01} \sigma_0'(p_0)$
		* These now have derivatives and if $\sigma_i(x) = id(x)$, we can replace the derivatives by 1 and recover the backward chains from CASE I.

As before, let's rewrite the derivatives with missing terms highlighted in ${\color{red} {red}}$.

$\frac{\partial C}{\partial w_{23}} = (q_3-y) \sigma_3'(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))) {\color{red} {w_{23}}} \sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))$

$\frac{\partial C}{\partial w_{12}} = (q_3-y) \sigma_3'(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))) w_{23} \sigma_2'(w_{12}\sigma_1(w_{01}\sigma_0(p_0))) {\color{red} {w_{12}}} \sigma_1(w_{01}\sigma_0(p_0))$

$\frac{\partial C}{\partial w_{01}} = (q_3-y) \sigma_3'(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))) w_{23} \sigma_2'(w_{12}\sigma_1(w_{01}\sigma_0(p_0))) w_{12} \sigma_1'(w_{01}\sigma_0(p_0)) {\color{red} {w_{01}}} \sigma_0(p_0) $

Define:

$\delta_0 = (q_3-y) \sigma_3'(p_3)$

$\delta_1 = (q_3-y) \sigma_3'(p_3) w_{23} \sigma_2'(p_2)$

$\delta_2 = (q_3-y) \sigma_3'(p_3) w_{23}) \sigma_2'(p_2) w_{12} \sigma_1'(p_1)$

Using these, we can rewrite the derivatives:

$\frac{\partial C}{\partial w_{23}} = \underbrace{(q_3-y) \sigma_3'(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0))))}\_{\delta_0} \space {\color{red} {w_{23}}} \space \underbrace{\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))}\_{q_2} = \delta_0 q_2$

$\frac{\partial C}{\partial w_{12}} = \underbrace{(q_3-y) \sigma_3'(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))) w_{23} \sigma_2'(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))}\_{\delta_1} \space {\color{red} {w_{12}}} \space \underbrace{\sigma_1(w_{01}\sigma_0(p_0))}\_{q_1} = \delta_1 q_1$

$\frac{\partial C}{\partial w_{01}} = \underbrace{(q_3-y) \sigma_3'(w_{23}\sigma_2(w_{12}\sigma_1(w_{01}\sigma_0(p_0)))) w_{23} \sigma_2'(w_{12}\sigma_1(w_{01}\sigma_0(p_0))) w_{12} \sigma_1'(w_{01}\sigma_0(p_0))}\_{\delta_2} \space {\color{red} {w_{01}}} \space \underbrace{q_0}\_{q_0} = \delta_2 q_0$

As before, we get the same pattern:

$\frac{\partial C}{\partial w_{i,i+1}} = \delta_{2-i} q_i$

## Backpropagation III - linear activations + multi-node layers