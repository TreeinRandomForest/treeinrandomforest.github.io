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

which is gratifying. 

## Backpropagation III - linear activations + multi-node layers

In practice, neural networks with one node per layer are not very helpful. What we really want is to put multiple nodes at each layer to get the classic feedforward neural network shown below. For now, as in Section I, we don't include non-linear activations.

SOME EXPLANATION OF MATRIX MULTIPLICATION

Forward propagation in this architecture is:

$x_1 = W_{01} x_0$

$x_2 = W_{12} x_1$

$x_3 = W_{23} x_2$

where the $W_{ij}$ are matrices of weights! We used $w_{ij}$ to refer to an individual weight in sections I and II and we'll use $W_{ij}$ to refer to the **weight matrix** that takes us from layer $i$ to layer $j$.

We can combine these equations to write:

$x_3 = W_{23}W_{12}W_{01}x_0$

As in section I, there's still the same silliness going on. Why not define $W_c = W_{23}W_{12}W_{01}$ which is just another matrix and do gradient descent on the elements of $W_c$ directly. As before though, we intend on introducing non-linear activations eventually.

In principle, we haven't done anything radically new. We just need to compute a cost and then find the derivatives with respect to each individual weight. Recall that 

(NEED TO introduce the idea of vector-valued outputs and notation for transpose)

$C[W_{01}, W_{12}, W_{23}] = \frac{(x_3-y)^T(x_3-y)}{2}$

where $x_3$ is a vector of $N$ outputs obtained by forward propagation the input $x_0$ and $y$ are the $N$ targets/labels.

INTRODUCE MATRIX DERIVATIVES

$\frac{\delta C}{\delta W_{ij}} \equiv 
\begin{bmatrix} 
 \frac{\partial C}{\partial w^{(ij)}\_{11}} & \frac{\partial C}{\partial w^{(ij)}\_{12}} & \ldots \frac{\partial C}{\partial w^{(ij)}\_{1m}} \\\
 \frac{\partial C}{\partial w^{(ij)}\_{21}} & \frac{\partial C}{\partial w^{(ij)}\_{22}} & \ldots \frac{\partial C}{\partial w^{(ij)}\_{2m}} \\\
 \vdots & & \\\
 \frac{\partial C}{\partial w^{(ij)}\_{n1}} & \frac{\partial C}{\partial w^{(ij)}\_{n2}} & \ldots \frac{\partial C}{\partial w^{(ij)}\_{nm}}
\end{bmatrix}$ 

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

Two observations:
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

$C = \frac{1}{2} x^TA^TAx = \frac{1}{2} x_i (A^T)\_{ij} (A)\_{jk} x_k = x_i a_{ji} a_{jk} x_k$

where as before repeated indices mean an implicit sum. Now, using the chain rule:

$\frac{\partial C}{\partial a_{cd}} = \frac{1}{2} [x_i \frac{\partial a_{ji}}{\partial a_{cd}} a_{jk} x_k + x_i a_{ji} \frac{\partial a_{jk}}{\partial a_{cd}} x_k]$

$\frac{\partial C}{\partial a_{cd}} = \frac{1}{2} [x_i \delta_{j,c}\delta_{i,d} a_{jk} x_k + x_i a_{ij} \delta_{j,c}\delta_{k,d} x_k] = \frac{1}{2} [x_d a_{ck} x_k + x_i a_{ic}x_d]$
