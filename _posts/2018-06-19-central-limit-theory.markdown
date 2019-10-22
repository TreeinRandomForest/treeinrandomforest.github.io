---
layout: post
title:  "A Physicist's Proof of the Central Limit Theorem"
date:   2018-06-18 16:59:43 -0400
categories: jekyll update
mathjax: true
---

Suppose we are given $$n$$ independent, identically distributed (i.i.d.) random variables $$Y_1, Y_2, \ldots, Y_n$$. We are interested in the distribution of 

$$
\bar{Y} = \frac{Y_1 + Y_2 + \ldots + Y_n}{n}
$$

Suppose, the probability distribution function (p.d.f.) of $$Y_i$$ is $$f: \mathbb{R} \rightarrow \mathbb{R}$$ and the p.d.f of $$\bar{Y}$$ is $$g: \mathbb{R} \rightarrow \mathbb{R}$$. Then,

\begin{align} \label{master}
g(\bar{y}; n) = \mathbb{P}[\bar{Y}=\bar{y}] = \int_{-\infty}^{\infty}{dy_1dy_2\ldots dy_{n}\, f(y_1) f(y_2)\ldots f(y_n)\delta\big({\bar{y}-\frac{y_1 + y_2 + \ldots + y_n}{n}\big)}}
\end{align}

where $$\delta$$ is the Dirac delta function defined as:
$$
\delta(x-x') = \begin{cases}
\infty &\text{x = x'}\\
0 & \text{else}
\end{cases}
$$

such that $$\int_{x'\in S}{dx\, \delta(x-x')} = 1$$

A Dirac delta exists to be integrated over i.e.

\begin{align}
\int_{x' \in S} {dx\, \delta(x-x')f(x)} = f(x')
\end{align}

If we were to use this property to integrate in equation \ref{master}, then we would lose one of the integrals, say $$y_1$$ and lose the symmetry between the $$y_i$$ terms. To preserve that symmetry, we'll use another representation of the delta function.

As motivation, consider the Fourier transform of a function $$f$$:
\begin{align} \label{ft}
\tilde{f}(k) = \int_{-\infty}^{\infty}\frac{dx}{\sqrt{2\pi}}\, e^{-ikx}f(x)
\end{align}

The inverse Fourier transform is given by:
\begin{align} \label{ift}
f(x) = \int_{-\infty}^{\infty} \frac{dk}{\sqrt{2\pi}} e^{ikx} \tilde{f}(k)
\end{align}

We can plug in the definition of the Fourier transform in equation \ref{ft} into equation \ref{ift} to get:

$$
f(x) = \int_{-\infty}^{\infty} \frac{dk}{\sqrt{2\pi}} e^{ikx} \tilde{f}(k)
= \int\int \frac{dk}{\sqrt{2\pi}} \frac{dx'}{\sqrt{2\pi}} e^{ikx}e^{-ikx'}f(x')
= \int dx'\, \delta(x-x') f(x')
$$

where
\begin{align}
\delta(x-x') = \int_{-\infty}^{\infty} \frac{dk}{2\pi} e^{ik(x-x')}
\end{align}

is exactly the Dirac delta we defined earlier.

We will now use this representation of the delta function in equation \ref{master}. Limits on integrals are from $$-\infty$$ to $$\infty$$ and won't be explicitly written.

$$
\begin{split}
g(\bar{y}; n) & = \mathbb{P}[\bar{Y}=\bar{y}] \\
& = \int{dy_1dy_2\ldots dy_{n}\, f(y_1) f(y_2)\ldots f(y_n)\delta\big({\bar{y}-\frac{y_1 + y_2 + \ldots + y_n}{n}\big)}} \\
& = \int dy_1dy_2\ldots dy_n\, \frac{dk}{2\pi} f(y_1)f(y_2)\ldots f(y_n) e^{ik(\bar{y}-\frac{y_1+y_2+\ldots+y_n}{n})} \\
& = \int\frac{dk}{2\pi} \, e^{ik\bar{y}} \int{dy_1\, f(y_1) e^{-iky_1/n}} \int{dy_2\, f(y_2) e^{-iky_2/n}} \ldots \int{dy_n\, f(y_n) e^{-iky_n/n}} \\
& = \int\frac{dk}{2\pi} \, e^{ik\bar{y}} \big[\int{dy_1\, f(y_1) e^{-iky_1/n}}\big]^n
\end{split}
$$

Effectively, apart from scalings and some normalization terms, this states that the p.d.f of the distribution of the mean is the inverse Fourier transform of the $$nth$$ power of the Fourier transform of the individual p.d.f. of the $$Y_i$$s.

Of course, one can choose explicit functional forms for $$f(y)$$ i.e. the p.d.f. of the $$Y_i$$s but we want a more general solution. One possible solution is to expand $$f(y)$$ in a series that can then be truncated for large n.

In general, given $$f$$, we can compute all the moments:
$$ \label{moments}
\big<X^n\big> = \int dx\, x^n f(x)
$$

as well as the central moments:
$$
\big<X^n\big>\_c = \int dx\, (x-\mu)^n f(x)
$$

where $$\mu = \big<X\big>$$ is the first moment and assumed to be finite. Note that $$\big<X\big>\_c = 0$$.

On the other hand, suppose we are given all the moments. Can we compute $$f$$ then? The answer to this question is positive as shown below.

Suppose, we are given all moments defined by equation \ref {moments}. We can rewrite those as:
$$
\begin{split}
\big<X^n\big> & = \int dx\, x^n f(x) \\
& = \int dx\, \frac{d^n}{d(-ik)^n}\rvert_{k=0} e^{-ikx} f(x) \\
& = \frac{d^n}{d(-ik)^n}\rvert_{k=0} \int dx\, e^{-ikx} f(x) \\
& = \frac{d^n}{d(-ik)^n}\rvert_{k=0} \sqrt{2\pi} \tilde{f}(k)
\end{split}
$$

In other words, knowing the moments is equivalent to knowing the derivatives of the Fourier transform of the p.d.f. at $$k=0$$.
$$
\frac{(-i)^n}{\sqrt{2\pi}}\big<X^n\big> = \frac{d^n\tilde{f}}{dk^n}(0)
$$

So, we know the Taylor expansion of $$\tilde{f}$$:
$$
\begin{split}
\tilde{f}(k) & = \tilde{f}(0) + \frac{d\tilde{f}(0)}{dk}k + \frac{d^2\tilde{f}(0)}{dk^2}\frac{k^2}{2!} + \ldots + \frac{d^m\tilde{f}(0)}{dk^m}\frac{k^m}{m!} + \ldots \\
& = \frac{1}{\sqrt{2\pi}} - \frac{i\big<X\big>}{\sqrt{2\pi}} k - \frac{\big<X^2\big>}{\sqrt{2\pi}}\frac{k^2}{2!} + \ldots + \frac{(-i)^m\big<X^m\big>}{\sqrt{2\pi}}\frac{k^m}{m!} + \ldots
\end{split}
$$

which can be inverted to give the original p.d.f. $$f$$. We have used the fact that $$\tilde{f}(0) = \int \frac{dx}{\sqrt{2\pi}} f(x) = \frac{1}{\sqrt{2\pi}}$$ since $$f$$ is a p.d.f.

We can also do the same expansion using the central moments instead:
$$
\begin{split}
\big<X^n\big>_c & = \int dx\, (x-\mu)^n f(x)\\
& = \int dx\, \frac{d^n}{d(-ik)^n}\rvert_{k=0} e^{-ik(x-\mu)} f(x) \\
& = \frac{d^n}{d(-ik)^n}\rvert_{k=0} \int dx\, e^{-ik(x-\mu)} f(x) \\
& = \frac{d^n}{d(-ik)^n}\rvert_{k=0} e^{ik\mu}\int dx\, e^{-ikx} f(x) \\
& = \frac{d^n}{d(-ik)^n}\rvert_{k=0} e^{ik\mu} \sqrt{2\pi}\tilde{f}(k)
\end{split}
$$

In other words, the moments give us the Taylor series terms in the expansion of $$e^{ik\mu} \tilde{f}(k)$$ instead of just $$\tilde{f}(k)$$ before. So,

Define $$\tilde{g}(k) = e^{ik\mu}\tilde{f}(k)$$.

$$
\begin{split}
\tilde{g}(k) & = \tilde{g}(0) + \frac{d\tilde{g}(0)}{dk}k + \frac{d^2\tilde{g}(0)}{dk^2}\frac{k^2}{2!} + \ldots + \frac{d^m\tilde{g}(0)}{dk^m}\frac{k^m}{m!} + \ldots \\
& = \frac{1}{\sqrt{2\pi}} - \frac{i\big<X\big>\_c}{\sqrt{2\pi}} k - \frac{\big<X^2\big>\_c}{\sqrt{2\pi}}\frac{k^2}{2!} + \ldots + \frac{(-i)^m\big<X^m\big>\_c}{\sqrt{2\pi}}\frac{k^m}{m!} + \ldots
\end{split}
$$

using $$\tilde{g}(0) = \tilde{f}(0) = \frac{1}{\sqrt{2\pi}}$$.

Finally,

$$
\tilde{f}(k) = e^{-ik\mu} \big( \frac{1}{\sqrt{2\pi}} - \frac{i\big<X\big>\_c}{\sqrt{2\pi}} k - \frac{\big<X^2\big>\_c}{\sqrt{2\pi}}\frac{k^2}{2!} + \ldots + \frac{(-i)^m\big<X^m\big>\_c}{\sqrt{2\pi}}\frac{k^m}{m!} + \ldots \big)
$$

We will use central moments from here on but the same result can be derived by using the non-central moments. We can use this result in equation \ref{master}:

$$
\begin{split}
g(\bar{y}; n) & = \int\frac{dk}{2\pi} \, e^{ik\bar{y}} \big[\int{dy_1\, f(y_1) e^{-iky_1/n}}\big]^n \\
& = \int\frac{dk}{2\pi} \, e^{ik\bar{y}} \big[\sqrt{2\pi}\tilde{f}(\frac{k}{n})\big]^n\\
& = \int\frac{dk}{2\pi} \, e^{ik\bar{y}} \big[ e^{-ik\mu/n} \sqrt{2\pi}^n \big( \frac{1}{\sqrt{2\pi}} - \frac{i\big<X\big>\_c}{\sqrt{2\pi}} \frac{k}{n} - \frac{\big<X^2\big>\_c}{\sqrt{2\pi}}\frac{1}{2!}(\frac{k}{n})^2 + \ldots + \frac{(-i)^m\big<X^m\big>\_c}{\sqrt{2\pi}}\frac{1}{m!}\big(\frac{k}{n}\big)^m + \ldots \big)\big]^n \\
& = \frac{1}{2\pi}\int dk\, e^{ik(\bar{y}-\mu)} \big(1 - i\big<X\big>\_c \frac{k}{n} - \frac{\big<X^2\big>\_c}{2} (\frac{k}{n})^2 + \frac{i\big<X^3\big>\_c}{3!} (\frac{k}{n})^3 + \frac{\big<X^4\big>\_c}{4!} (\frac{k}{n})^4 \ldots\big)^n
\end{split}
$$

More precisely, we want the limit

$$
\begin{split}
g(\bar{y}; n) & = \lim_{n\rightarrow\infty}\lim_{\Lambda\rightarrow\infty}\frac{1}{2\pi}\int_{-\Lambda}^{\Lambda} dk\, e^{ik(\bar{y}-\mu)} \big(1 - i\big<X\big>\_c \frac{k}{n} - \frac{\big<X^2\big>\_c}{2} (\frac{k}{n})^2 + \frac{i\big<X^3\big>\_c}{3!} (\frac{k}{n})^3 + \frac{\big<X^4\big>\_c}{4!} (\frac{k}{n})^4 \ldots\big)^n \\
& = \lim_{\Lambda\rightarrow\infty}\lim_{n\rightarrow\infty}\frac{1}{2\pi}\int_{-\Lambda}^{\Lambda} dk\, e^{ik(\bar{y}-\mu)} \big(1 - i\big<X\big>\_c \frac{k}{n} - \frac{\big<X^2\big>\_c}{2} (\frac{k}{n})^2 + \frac{i\big<X^3\big>\_c}{3!} (\frac{k}{n})^3 + \frac{\big<X^4\big>\_c}{4!} (\frac{k}{n})^4 \ldots\big)^n \\
& = \lim_{\Lambda\rightarrow\infty}\frac{1}{2\pi}\int_{-\Lambda}^{\Lambda} \lim_{n\rightarrow\infty} dk\, e^{ik(\bar{y}-\mu)} \big(1 - i\big<X\big>\_c \frac{k}{n} - \frac{\big<X^2\big>\_c}{2} (\frac{k}{n})^2 + \frac{i\big<X^3\big>\_c}{3!} (\frac{k}{n})^3 + \frac{\big<X^4\big>\_c}{4!} (\frac{k}{n})^4 \ldots\big)^n \\
\end{split}
$$

In particular, $$\frac{k}{n}$$ can be made arbitrarily small this way. For large enough n, $$\frac{\|k\|}{n} < \frac{\Lambda}{n} \equiv \epsilon$$.

Note, $$\big<X\big>_c = 0$$ and defining, $$\big<X^2\big>_c \equiv \sigma^2$$.
$$
g(\bar{y}) = \frac{1}{2\pi} \int_{-\infty}^{\infty} dk\, e^{ik(\bar{y}-\mu)}\big[1 - \frac{\sigma^2}{2} (\frac{k}{n})^2+ \mathcal{O}((\frac{k}{n})^3)\big]^n 
$$

Using $$1 - x \approx e^x$$ for small $$x$$, we get
$$
g(\bar{y}) = \frac{1}{2\pi} \int_{-\infty}^{\infty} dk\, e^{ik\bar{y}}e^{-ik\mu}e^{-\frac{\sigma^2 k^2}{2n}} 
$$

which is exactly the inverse Fourier transform of the Fourier transform of a Gaussian with mean $$\mu$$ and variance $$\frac{\sigma^2}{n}$$.

The argument of the exponential in the integral is:
$$
\begin{split}
\text{arg} &= ik\bar{y}-ik\mu-\frac{\sigma^2 k^2}{2n} \\
& = -\frac{\sigma^2 k^2}{2n} + ik(\bar{y}-\mu) \\
& = -\frac{\sigma^2}{2n} \big[k^2 + \frac{i2n(\mu-\bar{y})}{\sigma^2}k\big] \\
& = -\frac{\sigma^2}{2n} \big[\big(k + \frac{in(\mu-\bar{y})}{\sigma^2}\big)^2 - \frac{i^2n^2(\mu-\bar{y})^2}{\sigma^4}\big] \\
& = -\frac{\sigma^2}{2n} \big(k + \frac{in(\mu-\bar{y})}{\sigma^2}\big)^2 -\frac{n(\bar{y}-\mu)^2}{2\sigma^2} 
\end{split}
$$

So, the integral is:
$$
\begin{split}
g(\bar{y}) &= \frac{1}{2\pi} \int dk\, e^{ik\bar{y}}e^{-ik\mu}e^{-\frac{\sigma^2 k^2}{2n}} \\
& = \frac{1}{2\pi} \int dk\, e^{-\frac{\sigma^2}{2n} \big(k + \frac{in(\mu-\bar{y})}{\sigma^2}\big)^2} e^{-\frac{n(\bar{y}-\mu)^2}{2\sigma^2}} \\
& = e^{-\frac{n(\bar{y}-\mu)^2}{2\sigma^2}} \frac{1}{2\pi} \int dk\, e^{-\frac{\sigma^2}{2n} \big(k + \frac{in(\mu-\bar{y})}{\sigma^2}\big)^2} \\
\end{split}
$$

The integral is a simple Gaussian integral that integrates to $$\sqrt{\frac{\pi}{\sigma^2 / 2n}}$$. So, we get
$$
g(\bar{y}) = e^{-\frac{n(\bar{y}-\mu)^2}{2\sigma^2}} \frac{1}{2\pi} \sqrt{\frac{\pi}{\sigma^2 / 2n}}
= \frac{1}{\sqrt{2\pi\sigma^2 / n}} e^{-\frac{(\bar{y}-\mu)^2}{2\sigma^2 / n}}
$$

is exactly a Gaussian p.d.f. with mean = $$\mu$$ and standard deviation = $$\frac{\sigma}{\sqrt{n}}$$.
