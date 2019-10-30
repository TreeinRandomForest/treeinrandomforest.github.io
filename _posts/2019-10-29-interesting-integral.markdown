---
layout: post
title:  "An Interesting Integral"
date:   2019-10-29 13:50:49 -0400
categories: jekyll update
mathjax: true
---

I was recently reading an old physics textbook and came across an interesting integral:

$$\int_0^\infty \frac{y^3}{e^y - 1} dy = \frac{\pi^4}{15}$$

This is pretty enough that one has to do the integral.

On first glance, one could expand $\frac{1}{e^y - 1}$ in a geometric series:

$$-\frac{1}{1-e^y} = -(1 + e^y + e^{2y} + e^{3y} + \ldots)$$

and then integrate term-by-term. The obvious problem is that this doesn't converge:

$$\int_0^\infty y^3 e^{ky} dy$$

with $k\geq0$

since the integrad blows up as $y \rightarrow \infty$.

Of course, the answer is to get decaying exponentials in the integral. 

$$\frac{1}{e^y-1} = \frac{e^{-y}}{1-e^{-y}} = e^{-y}(1 + e^{-y} + e^{-2y} + e^{-3y} + \ldots)$$

(multiply both numerator and denominator by $e^{-y}$).

So,

$$\int_0^\infty \frac{y^3}{e^y - 1} dy = \int_0^\infty y^3 (e^{-y} + e^{-2y} + e^{-3y} + \ldots) dy$$

We can easily solve each term by using integration by parts or by differentiating under the integral.

$$\int_0^\infty y^3 e^{-ky} dy = \big( \frac{d}{d(-k)}\big)^3 \int e^{-ky} dy$$

$$\implies I(k) \equiv \int_0^\infty y^3 e^{-ky} dy = -\big( \frac{d}{dk}\big)^3 \frac{1}{k} = \frac{6}{k^4}$$

So, 

$$\int_0^\infty \frac{y^3}{e^y - 1} dy = I(1) + I(2) + \ldots = \Sigma_{k=1}^{\infty} \frac{6}{k^4} = 6\Sigma_{k=1}^{\infty} \frac{1}{k^4}$$

We now need to evaluate this sum. The rigorous way of doing this is to use Fourier series. But years ago, while taking a break from research, I was playing with the function $\sin(x)$ and found a cute (but non-rigorous trick).

We know $\sin(x) = 0$ for all $x = n\pi$ where $n$ is an integer i.e. $n \in \mathbb{Z}$. This would lead to a guess:

$$\sin(x) = \ldots(x-3\pi)(x-2\pi)(x-\pi)x(x+\pi)(x+2\pi)(x+3\pi)\ldots$$

We know $\sin(0) = 0$ and this formula does give us that since it is proportional to $x$. We also know $\frac{\sin(x)}{x} \rightarrow_{x\rightarrow 0} 1$.

Following our formula, we get:

$$\frac{\sin(x)}{x} \rightarrow \ldots(0-3\pi)(0-2\pi)(0-\pi)(0+\pi)(0+2\pi)(0+3\pi)\ldots$$

This is a problem. We are multiplying infinite factors of $\pi$ and we also have alternating signs. So maybe we should rewrite each term $x - n\pi$ is a different way. The obvious way is to write $1 - \frac{x}{n\pi}$ since we are still analytic in $x$ (we wouldn't be if we wrote $1 - \frac{n\pi}{x}$) to get:

$$\frac{\sin(x)}{x} = \ldots(1-\frac{x}{3\pi})(1-\frac{x}{2\pi})(1-\frac{x}{\pi})(1+\frac{x}{\pi})(1+\frac{x}{2\pi})(1+\frac{x}{3\pi})\ldots$$

Now, when $x\rightarrow 0$, the right-hand side goes to 1. We can combine every pair of terms:

$$(1-\frac{x}{n\pi})(1+\frac{x}{n\pi}) = (1-\frac{x^2}{n^2\pi^2})$$

to finally get:

$$\sin(x) = x (1-\frac{x^2}{\pi^2})(1-\frac{x^2}{2^2\pi^2})(1-\frac{x^2}{3^2\pi^2})\ldots$$

Now, we know that the Taylor expansion of $\sin(x)$ is:

$$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \ldots$$

If we have two series representations of the same function, we should compare them term-by-term! The original motivation for doing so was to see if my guess for the infinite product representation was correct.

Let's compare the coefficients for $x$ first. In the product representation, the only way to get $x$ is to multiply the leading $x$ by 1s in all the terms. This matches with the $x$ in the Taylor expansion.

What about $x^2$? Clearly, there's no $x^2$ in the Taylor expansion. What about the product representation? If we choose even one single factor of $x^2$ from one of the factors, it would multiply with the leading $x$ and give $x^3$. So good news: no second-order term from the product.

What about $x^3$? We get an infinite number of such terms by multiplying the leading $x$ by one of the $x^2$ terms to get:

$$-x^3 (\frac{1}{\pi^2} + \frac{1}{2^2\pi^2} + \frac{1}{3^2\pi^2} + \ldots)$$

If our representation was correct, this infinite sum should equal $\frac{1}{3!}$.

$$1 + \frac{1}{4} + \frac{1}{9} + \frac{1}{16} + \ldots = \frac{\pi^2}{6}$$

after some re-arrangements. Wow! if you have played with Fourier series before, this should be familiar, hinting that the product representation might be correct.

This leads to the next question. What if we compare the coefficient for $x^5$ (for $x^4$, it's 0 in both representations)? For the Taylor expansion, the coefficient can just be read off. It's $\frac{1}{5!} = \frac{1}{120}$.

For the product representation, we can get $x^5$ if we multiply the leading factor of $x$ by two $x^2$ factors. For very such choice of pairs, we get a coefficient of $\frac{1}{\pi^2}\frac{1}{m^2}\frac{1}{n^2}$ where $m,n = 1,2,\ldots$.

So, we get:

$$\frac{1}{120} = \frac{1}{1^2\pi^2} (\frac{1}{2^2\pi^2} + \frac{1}{3^2\pi^2} + \ldots) + \frac{1}{2^2\pi^2} (\frac{1}{3^2\pi^2} + \frac{1}{4^2\pi^2} + \ldots) + \ldots$$

Every pair occurs once and we can write this more concisely as:

$$\frac{\pi^4}{120} = \Sigma_{m < n} \frac{1}{m^2} \frac{1}{n^2}$$

where it is implicit that $m,n$ range over the positive integers.

Now,

$$\Sigma_{m < n} \frac{1}{m^2} \frac{1}{n^2} = \frac{1}{2}\Sigma_{m\neq n} \frac{1}{m^2} \frac{1}{n^2} = \frac{1}{2}\big(\Sigma_{m, n} \frac{1}{m^2} \frac{1}{n^2} - \Sigma_{m = n} \frac{1}{m^2} \frac{1}{n^2}\big)$$

Putting this together, we get

$$\frac{\pi^4}{120} = \frac{1}{2}\big(\Sigma_{m, n} \frac{1}{m^2} \frac{1}{n^2} - \Sigma_{m} \frac{1}{m^4}\big)$$

But $\Sigma_m \frac{1}{m^4}$ is exactly what we want! Also note that

$$\Sigma_{m,n} \frac{1}{m^2} \frac{1}{n^2} = \Sigma_{m} \frac{1}{m^2}\Sigma_{n} \frac{1}{n^2} = \big( \frac{\pi^2}{6}\big)^2 = \frac{\pi^4}{36}$$

So,

$$\Sigma_{m} \frac{1}{m^4} = \frac{\pi^4}{36} - \frac{2\pi^4}{120} = \boxed{\frac{\pi^4}{90}}$$

If we go back to our original integral:

$$\int_0^\infty \frac{y^3}{e^y - 1} dy = 6\Sigma_{k=1}^{\infty} \frac{1}{k^4} = 6 \frac{\pi^4}{90} = \boxed{\frac{\pi^4}{15}}$$

This is not a mathematically rigorous demonstration but fun nonetheless.