---
layout: post
title:  "Symmetric Matrix Block Inverse"
date:   2021-06-22
categories: jekyll update
mathjax: true
---

Consider the symmetric block matrix

$$M = \begin{pmatrix}
A & C \\
C^T & B \\ 
\end{pmatrix}
$$

with $A = A^T$ and $B = B^T$ and $A, B$ are invertible.

We want to calculate the inverse:

$$M^{-1} =
\begin{pmatrix}
E & G \\
G^T & F \\ 
\end{pmatrix}
$$

Note that the inverse of an invertible symmetric matrix is symmetric. This is used in the general form for $M^{-1}$ above. Quick proof: $A = A^T$. So (multiplying on the left and right by $A^{-1}$), $A^{-1}A = A^{-1}A^T = I$ and $AA^{-1} = A^T A^{-1} = I$. By uniquess of inverses, $(A^T)^{-1} = A^{-1}$. Since $(B^T)^{-1} = (B^{-1})^T$ for any general invertible $B$, $(A^{-1})^T = A^{-1}$ i.e. $A^{-1}$ is symmetric.

$$
\begin{pmatrix}
A & C \\
C^T & B \\ 
\end{pmatrix}
\begin{pmatrix}
E & G \\
G^T & F \\ 
\end{pmatrix} =
\begin{pmatrix}
I_1 & 0 \\
0 & I_2 \\
\end{pmatrix}
$$

where the two identity matrices will be of different sizes generally to be consistent with the block structure. This results in 4 identities for the three unknowns: $E, F, G$.

$$\begin{equation}
AE + CG^T = I_1 \\
C^T G + BF = I_2 \\
AG + CF = 0 \\
C^T E + B G^T = 0 \\
\end{equation}
$$

$$AG + CF = 0$$: Solve for $G$ to get $G = -A^{-1}CF$.

$$C^T G + BF = I_2$$: 

Substituting for $G$, we get

$$-C^TA^{-1}CF + BF = I_2$$

$$\implies (B-C^TA^{-1}C)F = I_2$$

to get

$$\boxed{F = (B-C^TA^{-1}C)^{-1}}$$

Using $G = -A^{-1}CF$, we get

$$\boxed{G = -A^{-1}C(B-C^TA^{-1}C)^{-1}}$$

$$AE + CG^T = I_1$$:

Solving for $E$,

$$E = A^{-1}(I_1 - CG^T) = A^{-1}(I_1 - C[-A^{-1}C(B-C^TA^{-1}C)^{-1}]^T)$$

Simplifying, $\boxed{E = A^{-1}(I_1 + C[B-C^TA^{-1}C]^{-1}C^TA^{-1})}$

where we have freely used the symmetry of $A, B$ and the identity $(A^T)^{-1} = (A^{-1})^T$.

So, 

$$\boxed{M^{-1} =
\begin{pmatrix}
A^{-1}(I_1 + C[B-C^TA^{-1}C]^{-1}C^TA^{-1}) & -A^{-1}C(B-C^TA^{-1}C)^{-1} \\
-(B-C^TA^{-1}C)^{-1}C^TA^{-1} & (B-C^TA^{-1}C)^{-1} \\ 
\end{pmatrix}}
$$

**Sanity Check 1**:

If $M$ is block diagonal ($C = 0$), we get,

$$M^{-1} =
\begin{pmatrix}
A^{-1}(I_1 + C[B-C^TA^{-1}C]^{-1}C^TA^{-1}) & -A^{-1}C(B-C^TA^{-1}C)^{-1} \\
-(B-C^TA^{-1}C)^{-1}C^TA^{-1} & (B-C^TA^{-1}C)^{-1} \\ 
\end{pmatrix}
= \begin{pmatrix}
A^{-1} & 0 \\
0 & B^{-1} \\
\end{pmatrix}
$$

as expected.

**Sanity Check 2**:

We used three equations to derive $M^{-1}$. Check to see our solution is consistent with the fourth equation:

$$C^T E + B G^T = 0$$

$$\begin{equation}
\begin{split}
LHS &= C^T E + B G^T \\ 
&= C^T A^{-1}(I_1 + C[B-C^TA^{-1}C]^{-1}C^TA^{-1})  + B [-A^{-1}C(B-C^TA^{-1}C)^{-1}]^T \\
&= C^T A^{-1} + C^T A^{-1}C[B-C^TA^{-1}C]^{-1}C^TA^{-1} - B(B-C^TA^{-1}C)^{-1}]C^TA^{-1}\\
&= C^T A^{-1} + [C^T A^{-1} C - B][B-C^TA^{-1}C]^{-1}C^TA^{-1}\\
&= C^T A^{-1} - C^T A^{-1} \\
&= 0 \\
&= RHS
\end{split}
\end{equation}$$
