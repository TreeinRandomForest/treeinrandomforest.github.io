---
layout: post
title:  "A detailed look at backpropagation in feedforward neural networks"
date:   2018-10-30
categories: [deep-learning]
tags: [deep-learning]
mathjax: true
---

Some assumptions/prerequisites/notes before we start:

* You'll need some familiarity with matrices and matrix multiplication as well as differentiation from calculus (but no integration at all).
* Ideally, get a few sheets of paper, a pen and a quiet space and work through the calculations as you go alone. Writing something out cements the material exponentially more than just reading it.
* Unfortunately I don't know how to show the details without mathematics. Please don't be turned off by unusual symbols - they are just strokes on a piece of paper or pixels on a screen.
* What you'll hopefully take away is that after all the fog clears, the simple act of calculating derivatives for this problem results in simple, iterative equations that let us train neural networks very efficiently.

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

which is gratifying. 

## Backpropagation III - linear activations + multi-node layers

In practice, neural networks with one node per layer are not very helpful. What we really want is to put multiple nodes at each layer to get the classic feedforward neural network shown below. For now, as in Section I, we won't include non-linear activations.

SOME EXPLANATION OF MATRIX MULTIPLICATION

Forward propagation in this architecture is:

$x_1 = W_{01} x_0$

$x_2 = W_{12} x_1$

$x_3 = W_{23} x_2$

where the $W_{ij}$ are matrices of weights! We used $w_{ij}$ to refer to an individual weight in sections I and II and we'll use $W_{ij}$ to refer to the **weight matrix** that takes us from layer $i$ to layer $j$.

We can combine these equations to write:

$x_3 = W_{23}W_{12}W_{01}x_0$

As in section I, there's still the same silliness going on. Why not define $W_c = W_{23}W_{12}W_{01}$ which is just another matrix and do gradient descent on the elements of $W_c$ directly. As before though, we intend on introducing non-linear activations eventually.

In principle, we haven't done anything radically new. We just need to compute a cost and then find the derivatives with respect to each individual weight. Recall that we were using the mean-squared error metric as a cost function. The only difference is that now the output itself might be a vector:

$y = (y_1, y_2, \ldots, y_n)$

i.e. there are $n$ labels and the output vector $x_3$ also has $n$ dimensions. So the cost would just be a sum of mean-squared errors for every element in $y$ and $x_3$:

$C = \frac{1}{n} [(x_{3,1}-y_1)^2 + (x_{3,2}-y_2)^2 + \ldots + (x_{3,n}-y_n)^2]$

where $x_3 = (x_{3,1}, x_{3,2}, \ldots, x_{3,n})$

so $x_{3,i}$ denotes the $i$th element of $x_3$.

A more concise way of writing this is as follows:

$C[W_{01}, W_{12}, W_{23}] = \frac{(x_3-y)^T(x_3-y)}{2}$

where $x^T$ denotes the transpose of a vector. More generally, given a matrix $A$ with elements $a_{ij}$, the transpose of a matrix, denoted by $A^T$ has elements where the rows and columns are flipped. So

$(A^T)\_{ij} = a_{ji}$

Note that the indices on the right-hand side are flipped. An example will make this clear:

$A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\\
a_{21} & a_{22} & a_{23} \\\
\end{bmatrix}
\implies
A^T = \begin{bmatrix}
a_{11} & a_{21}\\\
a_{12} & a_{22}\\\
a_{13} & a_{23}\\\
\end{bmatrix}
$

So, the $ij$-th element of $A^T$ is the $ji$-th element of A. In other words, $A^T$ takes every row of $A$ and makes it into a column. Moreover, transposing a matrix changes its dimensions. If $A$ has $n$ rows and $m$ columns, also written as $dim(A) = (n,m)$, then $A^T$ has $m$ rows and $n$ dimensions i.e. $dim(A^T) = (m,n)$.

INTRODUCE MATRIX DERIVATIVES

$\frac{\delta C}{\delta W_{ij}} \equiv 
\begin{bmatrix} 
 \frac{\partial C}{\partial w^{(ij)}\_{11}} & \frac{\partial C}{\partial w^{(ij)}\_{12}} & \ldots \frac{\partial C}{\partial w^{(ij)}\_{1m}} \\\
 \frac{\partial C}{\partial w^{(ij)}\_{21}} & \frac{\partial C}{\partial w^{(ij)}\_{22}} & \ldots \frac{\partial C}{\partial w^{(ij)}\_{2m}} \\\
 \vdots & & \\\
 \frac{\partial C}{\partial w^{(ij)}\_{n1}} & \frac{\partial C}{\partial w^{(ij)}\_{n2}} & \ldots \frac{\partial C}{\partial w^{(ij)}\_{nm}}
\end{bmatrix}$ 

Expanding the cost function, we get:

$C[W] = \frac{(x_3-y)^T(x_3-y)}{2} = \frac{1}{2}\[x_3^Tx_3 - x_3^Ty - y^Tx_3 + y^Ty\]$

The only term that doesn't depend on the weights matrices is $y^Ty$ and is a constant once the dataset is fixed (i.e. the labels are fixed). So we can neglect this term from here on since it'll never contribute to our derivatives.

Also, $x_3^Ty = y^Tx_3$ since they are both just dot products between the same vectors. More explicitly, if $x_3 = (a_1 a_2 \ldots a_n)$ and $y = (b_1 b_2 \ldots b_n)$, then

$x_3^Ty = [a_1 a_2 \ldots a_n]
\begin{bmatrix}
b_1 \\\
b_2 \\\
\vdots \\\
b_n
\end{bmatrix}
= a_1 b_1 + a_2 b_2 + \ldots a_n b_n$

and 

$y^T x_3 = [b_1 b_2 \ldots b_n]
\begin{bmatrix}
a_1 \\\
a_2 \\\
\vdots \\\
a_n
\end{bmatrix}
= b_1 a_1 + b_2 a_2 + \ldots b_n a_n$

The two values are the same.

So, we can rewrite the cost 

$$C[W] = \frac{1}{2}[x_3^Tx_3 - 2 y^Tx_3] = \frac{x_3^Ty}{2} - y^Tx_3$$

To reiterate:
* "$=$" is being misused here since we completely dropped the term $y^Ty$ BUT since we are only using $C$ to find the derivatives for gradient descent and the dropped term doesn't contribute, it doesn't matter. If it makes more comfortable, you could define a new cost $C' = C - y^Ty$ and since minimizing a function $f$ is equivalent to minimizing $f + \text{constant}$, minimizing $C'$ and $C$ is equivalent in the sense that they will result in the same set of minimizing weights.
* We combined $x_3^Ty$ and $y^T x_3$ since they are equal (hence the factor of 2).

Good progress! We are no minimizing (\REF to above cost). Now, we can compute the derivative with respect to every matrix element of every matrix $W_{ij}$ and do gradient descent on each one:

$w_{ab}^{(ij), t+1} = w_{ab}^{(ij), t} - \eta \frac{\partial C}{\partial w_{ab}^{(ij), t+1}}$

Some notes on notation. In $w_{ab}^{(ij), t+1}$, the $t$ refers to the step in gradient descent, $(ij)$ refers to the matrix $W_{ij}$ that the weight comes from and $ab$ refers to the matrix element, i.e. row $a$ and column $b$. This horrible tragedy of notational burden is 1) very annoying, 2) absolutely devoid of any insight. Sure we can compute this mess and maybe even elegantly but unlike sections I and II, there seem to be no nice backward chains here. To prepare a nice meal, one has to sometimes do a lot of "prep" i.e. preparation of ingredients and "pre-processing" them. Using mathematics to understand and gain insights is no different. So we'll take a nice de-tour to introduce the idea of **matrix derivatives**.

### Aside: Matrix derivatives or "I don't like a million indices"

#### Cost linear in weights

Let's start with a simple cost function:

$C[A] = y^TAx$

where $x, y$ are vectors and $A$ is a matrix. More precisely,

$x: (n, 1)$

$A: (m, n)$

$y: (m, 1)$

where we define the dimensions of each object.

$x: (n,1)$ means $x$ has $n$ rows and 1 column.

$A: (m,n)$ means $A$ has m rows and n columns.

$y: (m,1)$ means $y$ has m rows and 1 column or $y^T: (1,m)$ i.e. $y^T$ has 1 row and m columns as one would expect from transposing it.

Why is all this important? Because the product $O_1O_2$ is only defined when

$\text{number of columns of } O_1 = \text{number of rows of } O_2$

and the dimensions of $O_1O_2$ will be:

$(\text{number of rows of }O_1, \text{number of columns of }O_2)$

A more concise way of writing this is:

$O_1: (n_1, k)$
$O_2: (k, m_2)$

$O_1O_2: (n_1, m_2)$

For our case,

$C[A] = \underbrace{y^T}\_{(1,m)}\underbrace{A}\_{(m,n)}\underbrace{x}\_{(n,1)}$

and $C[A]: (1,1)$ i.e. it's just a number which is what we expected to get for the cost.

Our notation for the cost:

$C[A]$ 

betrays our intention to keep $x, y$ fixed and minimize $C$ as a function of the elements of $A$. We'll still use gradient descent which requires that we compute the derivatives of $C$ with respect to the elements of $A$.

$A = \begin{bmatrix}
	a_{11} & a_{12} & \ldots & a_{1n} \\\
	a_{21} & a_{22} & \ldots & a_{2n} \\\
	\vdots \\\
	a_{m1} & a_{m2} & \ldots & a_{mn} \\\
\end{bmatrix}$

Once we have the derivatives:

$\frac{\partial C}{\partial a_{ij}}$

we can update the elements of $A$:

$a_{ij}^{t+1} = a_{ij}^{t} - \eta \frac{\partial C}{\partial a_{ij}^{t}}$

Instead let's try and combine the derivatives in a matrix:

$\frac{\delta C}{\delta A} \equiv \begin{bmatrix}
	\frac{\partial C}{\partial a_{11}} & \frac{\partial C}{\partial a_{12}} & \ldots & \frac{\partial C}{a_{1n}} \\\
	\frac{\partial C}{\partial a_{21}} & \frac{\partial C}{\partial a_{22}} & \ldots & \frac{\partial C}{a_{2n}} \\\
	\vdots \\\
	\frac{\partial C}{\partial a_{m1}} & \frac{\partial C}{\partial a_{m2}} & \ldots & \frac{\partial C}{a_{mn}} \\\
\end{bmatrix}$

We can then write:

$A^{t+1} = A^{t} - \eta \frac{\delta C}{\delta A^t}$

i.e. update the whole matrix in one go!

For \REF above to be define, we require $dim(A) = dim(\frac{\delta C}{\delta A}) = (m,n)$.

We also know that $C$ is linear in the elements of $A$ (more below) and so the derivatives should not depend on $A$ - just like the derivative of $f(x) = ax + b$ with respect to x, $\frac{df}{dx} = a$ doesn't depend on $x$. So, $\frac{\delta C}{\delta A}$ can only depend on $x,y$ and the only way to construct a matrix of dimension $(m,n)$ from $x$ and $y$ is 

$\underbrace{y}\_{(m,1)}\underbrace{x^T}\_{(1,n)} = \begin{bmatrix}
y_1 \\\
y_2 \\\
\vdots \\\
y_m
\end{bmatrix}
\begin{bmatrix}
x_1 & x_2 & \ldots & x_n \\\
\end{bmatrix} = \begin{bmatrix}
y_1 x_1 & y_1 x_2 & \ldots y_1 x_n \\\
y_2 x_1 & y_2 x_2 & \ldots y_2 x_n \\\
\vdots \\\
y_m x_1 & y_m x_2 & \ldots y_md x_n \\\
\end{bmatrix}$

Maybe all this is just too general and hand-wavy. After all, couldn't we multiply $yx^T$ by a constant and still get something with dimension $(m,n)$. That's true! So, let's compute the derivative matrix explicitly to convince ourselves.

$C[A] = \begin{bmatrix}
	y_1 & y_2 & \ldots & y_m \\\
\end{bmatrix}
\begin{bmatrix}
	a_{11} & a_{12} & \ldots & a_{1n} \\\
	a_{21} & a_{22} & \ldots & a_{2n} \\\
	\vdots \\\
	a_{m1} & a_{m2} & \ldots & a_{mn} \\\
\end{bmatrix}
\begin{bmatrix}
	x_1 \\\
	x_2 \\\
	\vdots \\\
	x_n
\end{bmatrix}$

$C[A] = \begin{bmatrix}
	y_1 & y_2 & \ldots & y_m \\\
\end{bmatrix}
\begin{bmatrix}
	a_{11} x_1 + a_{12} x_2 + \ldots + a_{1n} x_n \\\
	a_{21} x_1 + a_{22} x_2 + \ldots + a_{2n} x_n \\\
	\ldots \\\
	a_{m1} x_1 + a_{m2} x_2 + \ldots + a_{mn} x_n \\\
\end{bmatrix} \\\ = y_1 a_{11} x_1 + y_1 a_{12} x_2 + \ldots y_1 a_{1n} x_n + \\\ \space\space y_2 a_{21} x_1 + y_2 a_{22} x_2 + \ldots y_2 a_{2n} x_n + \\\ \space\space y_m a_{m1} x_1 + y_m a_{m2} x_2 + \ldots y_m a_{mn} x_n$

If we look closely at the last line, all the terms are of the form $y_i a_{ij} x_j$ (which is exactly how one writes matrix multiplication). So, we could write this as:

$C[A] = \Sigma_{i=1}^{m}\Sigma_{j=1}^{n} y_i a_{ij} x_j$

We also introduce the so-called Einstein (yes, the same Einstein you are thinking about) notation here now. We drop the summation sign, $\Sigma$ and write:

$C[A] = y_i a_{ij} x_j$

with the convention that any index that repeats twice is to be summed over. Since i appears twice - once with $y$ and once in $a_{ij}$ and j appears twice - once with $a_{ij}$ and once with $x_j$, they both get summed over the appropriate range. This way we don't have to write the summation sign each way.

To be clear, $y_i a_{ij} x_j$ is the same as $\Sigma_{i=1}^{m}\Sigma_{j=1}^{n} y_i a_{ij} x_j$ using the Einstein notation. Also, it doesn't matter what we call the repeated index so:

$y_i a_{ij} x_j = y_{bob} a_{bob,nancy} x_{nancy}$

It doesn't matter at all what we can the indices.

Great! so we computed an explicit form of $C$ and now we want derivatives with respect to $a_{kl}$ where $k,l$ are just indices denoting row k and column l. 

$\frac{\partial C}{\partial a_{kl}} = \frac{\partial}{\partial a_{kl}} [y_i a_{ij} x_j]$

Some more notation. We define:

$\delta_{a,b} = \begin{cases}
1, \text{if } a=b \\\
0, \text{otherwise} \\\
\end{cases}$

Then, 

$\frac{\partial C}{\partial a_{kl}} = y_i x_j \frac{\partial a_{ij}}{\partial a_{kl}}$

since $y_i, x_j$ don't depend on $a_{kl}$.

Now, 

$\frac{\partial a_{ij}}{\partial a_{kl}} = \begin{cases}
1, \text{if } i=k, j=l \\\
0, \text{otherwise}
\end{cases}$

Another way of writing this is:

$\frac{\partial a_{ij}}{\partial a_{kl}} = \delta_{i,k}\delta_{j,l}$

So,

$\frac{\partial C}{\partial a_{kl}} = y_i x_j \frac{\partial a_{ij}}{\partial a_{kl}} = y_i x_j \delta_{i,k}\delta_{j,l}$

But since repeated indices are summed over, when $i=k$ and when $j=l$, we get:

$\frac{\partial C}{\partial a_{kl}} = y_k x_l$

which is exactly the $(k,l)$ element of $yx^T$. So we just showed through explicit calculation that:

$\frac{\delta C}{\delta A} = y x^T$

the same result we got earlier by looking at various dimensions.

Phew! So all this work just to show that if:

$C[A] = y^T A x$

then, 

$\frac{\delta C}{\delta A} = y x^T$

which can be used in gradient descent as:

$A^{t+1} = A^{t} - \eta \frac{\delta C}{\delta A^t}$

Now (anticipating future use), what if 

$C[A, B] = y^T A B x$

is our cost function. Is there an easy way to calculate $\frac{\delta C}{\delta A}$ and $\frac{\delta C}{\delta B}$? You bet there is!

Let's start with $\frac{\delta C}{\delta A}$.

We can define $x' = Bx$ to get $C = y^T A x'$.

We know, $\frac{\delta C}{\delta A} = y x'^T$ from our earlier result and we can just replace $x'$ to get:

$\frac{\delta C}{\delta A} = y (Bx)^T = y x^T B^T$ using the $(AB)^T = B^TA^T$.

On to $\frac{\delta C}{\delta B}$. We can use a similar trick.

$C = (A^Ty)^T B x$ since $(A^Ty)^T = y^T A$

Let's define $y' = A^T y$ to get $C = y'^T B x$. From our previous result:

$\frac{\delta C}{\delta B} = y' x^T = A^T y x^T$

#### Cost quadratic in weights

What if the cost has a different form now?

$C = \frac{1}{2} x^TA^TAx$

Now we have two $A$ matrices multiplying the terms hence it's quadratic in the weights/elements of $A$.

The dimensions are:

$dim(x) = (n,1) \implies dim(x^T) = (1,n)$

$dim(A) = (m,n) \implies dim(A^T) = (n,m)$

Can we still guess what $\frac{\delta C}{\delta A}$ should be from the dimensions alone?

We expect $dim(A) = dim(\frac{\delta C}{\delta A}) = (m,n)$ and also since $C$ is quadratic in $A$, we expect the derivative to be linear in $A$.

Let's take a few guesses:

$\frac{\delta C}{\delta A} = A (x^Tx)$ which works dimensionally since $x^Tx$ is just a number.

$\frac{\delta C}{\delta A} = A (xx^T)$ which works dimensionally since $xx^T$ has dimension $(n,n)$ so we still get something linear in $A$ and with dimension $(m,n)$.

Technically, $\frac{\delta C}{\delta A} = A (xx^T) (xx^T)$ also works. But if we follow our intuition from calculus, $C$ is quadratic in $x$ and the $x$ terms just come along for the ride as constants. Taking derivatives can't change its order. So the final answer also needs to be quadratic in $x$ which rules out $A (xx^T) (xx^T)$ or $A (xx^T)^n$ for $n>1$.

Let's see if we can convince ourselves by doing an explicit calculation. We'll happly use our new index notation to cut through the calculation:

$C = \frac{1}{2} x^TA^TAx = \frac{1}{2} x_i (A^T)\_{ij} (A)\_{jk} x_k = \frac{1}{2} x_i a_{ji} a_{jk} x_k$

where as before repeated indices mean an implicit sum. Now, using the chain rule:

$\frac{\partial C}{\partial a_{cd}} = \frac{1}{2} [x_i \frac{\partial a_{ji}}{\partial a_{cd}} a_{jk} x_k + x_i a_{ji} \frac{\partial a_{jk}}{\partial a_{cd}} x_k]$

$\frac{\partial C}{\partial a_{cd}} = \frac{1}{2} [x_i \delta_{j,c}\delta_{i,d} a_{jk} x_k + x_i a_{ji} \delta_{j,c}\delta_{k,d} x_k] = \frac{1}{2} [x_d a_{ck} x_k + x_i a_{ci}x_d]$

These two terms are exactly the same:

$x_d a_{ck} x_k = x_d (Ax)\_{c}$

$x_i a_{ci} x_d = x_d a_{ci} x_i = x_d (Ax)\_{c}$

and add up to kill the factor of $\frac{1}{2}$.

So,

$\frac{\partial C}{\partial a_{cd}} = (Ax)\_{c} x_d = (Axx^T)\_{cd}$

In other words, we just showed that:

$\frac{\delta C}{\delta A} = Ax x^T$.

We still have one more calculation to do that will be crucial for doing back-propagation on our multi-node neural network.

Now,

$C = \frac{1}{2} x^T B^T A^T A B x$

Calculating $\frac{\delta C}{\delta A}$ is easy given what we just calculated and we just need to replace $x \rightarrow B x$. So,

$\frac{\delta C}{\delta A} = (ABx) (Bx)^T = (ABx)x^TB^T$

But what about $\frac{\delta C}{\delta B}$? If we define $D = A^T A$ then we have

$C = \frac{1}{2} x^T B^T D B x$

Again we'll guess our solution base on dimensions and then explicitly compute it.

We are given the sizes:

$dim(x) = (n,1) \implies dim(x^T) = (1,n)$

$dim(B) = (m, n) \implies dim(B^T) = (n, m)$

$dim(A) = (l, m) \implies dim(A^T) = (m,l)$

We also expect $dim(\frac{\delta C}{\delta B}) = dim(B) = (m,n)$. As before, the derivative should be linear in $B$, quadratic in $A$ and $x$ since they are for all practical purposes, constants for us.

So, let's see what modular pieces we have to work with:

Quadratic in $x$:

$dim(x^Tx) = (1,1)$

$dim(xx^T) = (n,n)$

Quadratic in $A$:

$dim(A A^T) = (l,l)$

$dim(A^T A) = (m,m)$

Linear in $B$:

$dim(B) = (m,n)$

So we can multiply $B$ on the right by something that is $(1,1)$ i.e. $x^Tx$ or $(n,n)$ i.e. $xx^T$ and on the left by something that is $(m,m)$ i.e. $A^TA$.

Guess is:

$\frac{\delta C}{\delta B} = (A^T A) B \begin{cases} 
xx^T \\\
x^Tx \\\
\end{cases}$

We also know that if we replace $D = A^T A$ by the $(m,m)$ identity matrix, we recover our previous example $C = \frac{1}{2} x^T B^T B x$ which gave us $\frac{\delta C}{\delta B} = B (xx^T)$ so we know $xx^T$ is the wrong choice to make.

So, to summarize, if 

$C = \frac{1}{2} x^T B^T A^T A B x$

then 

$\frac{\delta C}{\delta A} = (ABx) (Bx)^T = (ABx)x^TB^T$

and 

$\frac{\delta C}{\delta B} = (A^T A) B (xx^T)$

Of course, let's prove this by doing the explicit calculation using our powerful index notation:

We defined $D = A^TA$ which is a symmetric matrix i.e. $D^T = (A^TA)^T = A^T A = D$ or in terms of elements of $D$, $d_{ij} = d_{ji}$.

$C = \frac{1}{2} x^T B^T D B x = \frac{1}{2} x_i (B^T)\_{ij} (D)\_{jk} (B)\_{kl} x_l = \frac{1}{2} x_i b_{ji} d_{jk} b_{kl} x_l$

Then,

$[\frac{\delta C}{\delta B}]\_{cd} = \frac{1}{2} [x_i \frac{\partial b_{ji}}{\partial b_{cd}} d_{jk} b_{kl} x_l + x_i b_{ji} d_{jk} \frac{\partial b_{kl}}{\partial b_{cd}} x_l]$

We now the derivatives above can only be $1$ when the indices match and otherwise they are $0$:

$\frac{\partial b_{kl}}{\partial b_{cd}} = \delta_{k,c} \delta_{l,d}$

So,

$[\frac{\delta C}{\delta B}]\_{cd} = \frac{1}{2} [x_i \delta_{j,c}\delta_{i,d} d_{jk} b_{kl} x_l + x_i b_{ji} d_{jk} \delta_{k,c}\delta_{l,d} x_l]$

All repeated indices are summed over and the $\delta$s pick out the correct index. As an example:

$\delta_{a,b} x_b = \Sigma_{b=0}^{n} \delta_{a,b} x_b = \underbrace{\Sigma_{b\neq a} \underbrace{\delta_{a,b}}\_{= 0} x_b + \underbrace{\delta_{a,a}}\_{= 1} x_a}\_{\text{Separating terms where the index is a and not a}} = x_a$

In other words if you see something like

$\delta{a,b} x_b$

read it as "wherever you see a $b$, replace it with an $a$ and remove the deltas"

and if you see 

$\delta(a,b)\delta(c,d) x_b y_d$

read it as "wherever you see a $b$, replace it with $a and wherever you see $d$, replace it with $c$ and remove the deltas".

Using this, we get

$[\frac{\delta C}{\delta B}]\_{cd} = \frac{1}{2} [x_d d_{ck} b_{kl} x_l + x_i b_{ji} d_{jc} x_d]$

These are basically the same terms:

$[\frac{\delta C}{\delta B}]\_{cd} = \frac{1}{2} x_d [d_{ck} b_{kl} x_l + d_{jc} b_{ji} x_i]$

where we have just rearranged the factors in the second term and factored out $x_d$. Recall that $D$ was symmetric, i.e. $d_{jc} = d_{cj}$. Then we get

$[\frac{\delta C}{\delta B}]\_{cd} = \frac{1}{2} x_d [d_{ck} b_{kl} x_l + d_{cj} b_{ji} x_i]$

So the two terms are exactly the same since the only non-repeated index is $c$. In other words

$[\frac{\delta C}{\delta B}]\_{cd} = x_d d_{ck} b_{kl} x_l = (DBx)\_{c}x_d = (DBxx^T)\_{cd}$

confirming our suspicion that:

$\frac{\delta C}{\delta B}] = (DBxx^T) = (A^TA)B(xx^T)$

That's it! I promise that's the end of index manipulation exercises. We'll now collect all our results and use them to show that we still get backward chains as before.

$C = y^TAx$:

$\frac{\delta C}{\delta A} = yx^T$

$C = y^TABx$:

$\frac{\delta C}{\delta A} = yx^TB^T$

$\frac{\delta C}{\delta B} = A^Tyx^T$

$C = \frac{1}{2} x^TA^TAx$:

$\frac{\delta C}{\delta A} = A(xx^T)$

$C = \frac{1}{2} x^TB^TA^TABx$:

$\frac{\delta C}{\delta A} = AB(xx^T)B^T$

$\frac{\delta C}{\delta B} = (A^TA) B (xx^T)$

It's time to get back to our neural network and put all this together. To recap, our forward propagation was defined as:

$x_1 = W_{01} x_0$

$x_2 = W_{12} x_1$

$x_3 = W_{23} x_2$

or if we combine the equations:

$x_3 = W_{23}W_{12}W_{01}x_0$

and the cost is:

$$C[W_{01}, W_{12}, W_{23}] = \frac{1}{2}[x_3^Tx_3 - 2 y^Tx_3] = \frac{x_3^Ty}{2} - y^Tx_3 = \frac{1}{2}x_0^TW_{01}^TW_{12}^TW_{23}^TW_{23}W_{12}W_{01}x_0 - y^TW_{23}W_{12}W_{01}x_0$$

We can now use our catalog of matrix derivatives to calculate the 3 derivatives needed for gradient descent:

$\frac{\delta C}{\delta W_{01}}, \frac{\delta C}{\delta W_{12}}, \frac{\delta C}{\delta W_{23}}$

$\frac{\delta C}{\delta W_{01}}$:

Let's define $D \equiv W_{23}W_{12}$ to give:

$C = \frac{1}{2} x_0^T W_{01}^T D^T D W_{01} x_0 - y^T E W_{01} x_0$

Then, using identities REF AND REF (tool tips?):

$\frac{\delta C}{\delta W_{01}} = \frac{\delta}{\delta W_{01}} \frac{1}{2} x_0^T W_{01}^T D^T D W_{01} x_0 - \frac{\delta}{\delta W_{01}} y^T D W_{01} x_0 = (D^TD)W_{01}(x_0x_0^T) - D^T y x_0^T$

So,

$\frac{\delta C}{\delta W_{01}} = W_{12}^TW_{23}^T(W_{23}W_{12}W_{01}x_0)x_0^T - W_{12}^TW_{23}^Tyx_0^T$

But, $W_{23}W_{12}W_{01}x_0$ is precisely $x_3$, the result of forward propagation. So, we get a very nice result:

$\frac{\delta C}{\delta W_{01}} = W_{12}^TW_{23}^T(x_3-y)x_0^T$

NOTE about dependence on $x_3-y$

$\frac{\delta C}{\delta W_{12}}$:

Define $u \equiv W_{01}x_0$ to get:

$C = \frac{1}{2}u^TW_{12}^TW_{23}^TW_{23}W_{12}u - y^TW_{23}W_{12}u$

Using identities REF and REF (tooltips), we get:

$\frac{\delta C}{\delta W_{01}} = W_{23}^TW_{23}W_{12}uu^T - W_{23}^Tyu^T$

Replacing $u = W_{01}x_0$,

$\frac{\delta C}{\delta W_{12}} = W_{23}^TW_{23}W_{12}W_{01}x_0x_0^TW_{01}^T - W_{23}^Tyx_0^TW_{01}^T = W_{23}^Tx_3x_1^T - W_{23}^Tyx_1^T = W_{23}^T(x_3-y)x_1^T$

$\frac{\delta C}{\delta W_{23}}$:

Define $D \equiv W_{12}W_{01}$ to get:

$C = \frac{1}{2}x_0^TW_{01}^TW_{12}^TW_{23}^TW_{23}W_{12}W_{01}x_0 - y^TW_{23}W_{12}W_{01}x_0$

$C = \frac{1}{2}x_0^TD^TW_{23}^TW_{23}Dx_0 - y^TW_{23}Dx_0$

Using identities REF and REF (tooltips), we get:

$\frac{\delta C}{\delta W_{23}} = W_{23}D(x_0x_0^T)D^T - y^Tx_0^TD^T$

Replacing $D = W_{12}W_{01}$:

$\frac{\delta C}{\delta W_{23}} = W_{23}W_{12}W_{01}x_0x_0^TW_{01}^TW_{12}^T - y^Tx_0^TW_{01}^TW_{12}^T = (x_3-y)x_2^T$

In summary:

$\frac{\delta C}{\delta W_{01}} = W_{12}^TW_{23}^T(x_3-y)x_0^T$

$\frac{\delta C}{\delta W_{12}} = W_{23}^T(x_3-y)x_1^T$

$\frac{\delta C}{\delta W_{23}} = (x_3-y)x_2^T$


Presto!!! We again see forward and backward chains.

Forward chains:
* ${\color{blue} x_0}$
* $W_{01} x_0 = {\color{blue} x_1}$
* $W_{12} W_{01} x_0 = {\color{blue} x_2}$
* $W_{23} W_{12} W_{01} x_0 = {\color{blue} x_3}$

Backward chains:
* $x_3-y \equiv {\color{red} {\Delta_0}}$
* $W_{23}^T(x_3-y) \equiv {\color{red} {\Delta_1}}$
* $W_{12}^TW_{23}^T(x_3-y) \equiv {\color{red} {\Delta_2}}$
* $W_{01}^TW_{12}^TW_{23}^T(x_3-y) \equiv {\color{red} {\Delta_3}}$

where we now use capital deltas $\Delta$ instead of small deltas $\delta$, to signify that the backward chains are matrices.

As before, we can succinctly write the derivatives as:

$\frac{\delta C}{\delta W_{i,i+1}} = \Delta_{2-i} x_i^T$

In this notation, this is essentially the same as the results from sections I and II except for the fact that $x_i$ is now a vector and $\Delta_i$ is a matrix.

## Backpropagation IV - non-linear activations + multi-node layers

Finally! We can now start building up the full backpropagation for a realistic feedforward neural network with multiple layers, each with multiple nodes and non-linear activations.

We'll use notation similar to section II.

Forward propagation is:

$p_0 = x_0$

$q_0 = \sigma_0(x_0)$

$p_1 = W_{01} q_0$

$q_1 = \sigma_1(p_1)$

$p_2 = W_{12} q_1$

$q_2 = \sigma_2(p_2)$

$p_3 = W_{23} q_2$

$q_3 = \sigma_3(p_3)$

To summarize:
* $p_i$ and $q_i$ are always vectors.
* $p_i$ is the "pre-activation" ("p" for "pre") input to a layer. 
* $q_i$ is the "post-activation" ("q" since it comes after "p") output of a layer.
* $W_{i,j}$ always takes $q_i \rightarrow \p_j$.
* $\sigma_i$ always takes $p_i \rightarrow \q_i$.

The last two rules pop out of our equations and while it's just notation, it serves as a powerful guide to ensure that we are not making mistakes. At any point in the calculation if you see the combination $W_{ij} p_i$, something is probably wrong. If we see, $W_{ij}p_k$ where $k\neq i,j$, again something is probably wrong.

We'll use the mean-squared cost which is:

$C[W_{01}, W_{12}, W_{23}] = \frac{1}{2} (q_3^-y)^2$

As before, we need matrix derivatives but with a twist introduced by the activation functions.

Let's start with a simpler cost that mimics the term linear in $q_3$:

$C[A] = y^T \sigma(A) x$

where the dimensions are:

$\text{dim}(x) = (n,1)$

$\text{dim}(y) = (m,1)$

$\text{dim}(A) = (m,n)$

In other words:


$C[A] = \underbrace{y^T}\_{(1,m)} \underbrace{\sigma(A)}\_{(m,n)} \underbrace{x}\_{(n,1)}$

We want to compute 

$\frac{\delta C}{\delta A}$

We can try guessing what this should be based on the dimensions of the matrix. 

* $\frac{\delta C}{\delta A}$ should be linear in $x$ and $y$ 
* It should also depend on $\sigma'(A)$ where 

$\sigma'(A) = \begin{bmatrix}
\sigma'(a_{11}) & \sigma'(a_{12}) & \ldots & \sigma'(a_{1n}) \\\
\sigma'(a_{21}) & \sigma'(a_{22}) & \ldots & \sigma'(a_{2n}) \\\
\ldots \\\
\sigma'(a_{m1}) & \sigma'(a_{m2}) & \ldots & \sigma'(a_{mn}) \\\
\end{bmatrix}$

* $\text{dim}(\frac{\delta C}{\delta A}) = \text{dim}(A) = (m,n)$

* Any guess we come up with should reduce to the special case $\frac{\delta C}{\delta A} = yx^T$ when $\sigma = id$, the identity function, $id(x) = x$.

So we know that:

$\frac{\delta C}{\delta A} = yx^T \bigodot \sigma'(A)$.

where $\bigodot$ is a stand-in for something we need to do to combine $yx^T$ and $\sigma'(A)$. We can't just multiply the two matrices since the dimensions aren't consistent for multiplication, i.e. both these terms have dimension $(m,n)$.

Also, if $\sigma = id$, then $\sigma(x) = x$ and $\sigma'(x) = 1$, then 

$\sigma'(A) = \begin{bmatrix}
1 & 1 & \ldots & 1 \\\
1 & 1 & \ldots & 1 \\\
\ldots \\\
1 & 1 & \ldots & 1 \\\
\end{bmatrix}$

i.e. the matrix consisting of only $1$s denoted by $U$.

In this case, we know that the derivative should reduce to:

$\frac{\delta C}{\delta A} = yx^T \bigodot \sigma'(A) = yx^T \bigodot U = yx^T$.

So, for an arbitrary matrix $M$, we know now

$M \bigodot U = A$

where $\text{dim}(M) = \text{dim}(U) = 1$.

Now, we can guess what the operation $\bigodot$ does and then confirm our guess:

Given two matrices $A, B$ with the same dimensions, it seems

$(A\bigodot B)\_{ij} = a_{ij} b_{ij}$

or more explicitly

$\begin{bmatrix}
a_{11} & a_{12} & \ldots & a_{1n} \\\
a_{21} & a_{22} & \ldots & a_{2n} \\\
\ldots \\\
a_{m1} & a_{m2} & \ldots & a_{mn} \\\
\end{bmatrix}
\bigodot
\begin{bmatrix}
b_{11} & b_{12} & \ldots & b_{1n} \\\
b_{21} & b_{22} & \ldots & b_{2n} \\\
\ldots \\\
b_{m1} & b_{m2} & \ldots & b_{mn} \\\
\end{bmatrix}
= \begin{bmatrix}
a_{11}b_{11} & a_{12}b_{12} & \ldots & a_{1n}b_{1n} \\\
a_{21}b_{21} & a_{22}b_{22} & \ldots & a_{2n}b_{2n} \\\
\ldots \\\
a_{m1}b_{m1} & a_{m2}b_{m2} & \ldots & a_{mn}b_{mn} \\\
\end{bmatrix}$

So if all $b_{ij}=$, then we recover $A$ as expected. While we are still not quite sure if this is correct, our guess is:

$\frac{\delta C}{\delta A} = yx^T \bigodot \sigma'(A)$

where $(A \bigodot B)\_{ij} = a_{ij} b_{ij}$.

Let's confirm this with a more detailed calculation using the Einstein index notation:

$C = y_i \sigma(a_{ij}) x_j$

NEED TO CLEAR NOTATION

$\frac{\partial C}{\partial a_{kl}} = y_i x_j \sigma'(a_{ij}) \frac{\partial a_{ij}}{\partial a_{kl}} = y_k x_l \sigma'(a_{kl})$

$\frac{\partial C}{\partial a_{kl}} = [\frac{\delta C}{\delta A}]\_{kl} = y_k x_l \sigma'(a_{kl}) = (yx^T)\_{kl} \sigma'(a_{kl}) = [yx^T \bigodot \sigma'(A)]\_{kl}$

i.e.

$\frac{\delta C}{\delta A} = yx^T \bigodot \sigma'(A)$

as claimed before.

Anticipating future use, consider

$C = y^T \sigma_1(A \sigma_2(B)) x$

and the derivatives

$\frac{\delta C}{\delta A}, \frac{\delta C}{\delta B}$

$\frac{\delta C}{\delta A}$:


Testing collapsible markdown

<details>
	<summary>Expand</summary>
	
	Stuff goes here
	</details>