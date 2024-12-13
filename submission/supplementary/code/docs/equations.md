# Mathematical Framework

This document describes the mathematical foundations of the fractal field theory framework.

\begin{document}

## 1. Field Equations

### 1.1 Basic Field Definition

The fractal field $\phi(x)$ is defined on a self-similar space with dimension $D$. The field satisfies:

\begin{equation}
\label{eq:lagrangian}
\mathcal{L} = \frac{1}{2}(\partial_\mu \phi)(\partial^\mu \phi) - V(\phi)
\end{equation}

where $V(\phi)$ is the potential term with fractal scaling properties.

### 1.2 Scaling Properties

The field exhibits fractal scaling under the transformation:

\begin{equation}
\label{eq:scaling}
\phi(x) \rightarrow \lambda^{\Delta} \phi(\lambda x)
\end{equation}

where $\Delta$ is the scaling dimension and $\lambda$ is the scale factor.

### 1.3 Coupling Evolution

The coupling constants evolve according to:

\begin{equation}
\label{eq:beta}
\beta(g) = \mu \frac{\partial g}{\partial \mu} = -b_0 g^3 - b_1 g^5 + \mathcal{O}(g^7)
\end{equation}

where $b_0$ and $b_1$ are the first two coefficients of the beta function.

## 2. Observable Predictions

### 2.1 Cross Sections

The general form for cross sections is:

\begin{equation}
\label{eq:cross_section}
\sigma(E) = \frac{1}{s} \int |\mathcal{M}|^2 d\Phi_n
\end{equation}

where $\mathcal{M}$ is the matrix element and $d\Phi_n$ is the n-body phase space.

### 2.2 Correlation Functions

The n-point correlation function is defined as:

\begin{equation}
\label{eq:correlation}
G^{(n)}(x_1,\ldots,x_n) = \langle \phi(x_1)\cdots\phi(x_n) \rangle
\end{equation}

## 3. Error Analysis

### 3.1 Statistical Uncertainties

The statistical uncertainty propagation follows:

\begin{equation}
\label{eq:error_prop}
\sigma_f^2 = \sum_{i,j} \frac{\partial f}{\partial x_i} \frac{\partial f}{\partial x_j} \text{Cov}(x_i, x_j)
\end{equation}

### 3.2 Systematic Effects

Systematic uncertainties are incorporated through:

\begin{equation}
\label{eq:total_error}
\sigma_{\text{total}}^2 = \sigma_{\text{stat}}^2 + \sigma_{\text{syst}}^2 + 2\rho\sigma_{\text{stat}}\sigma_{\text{syst}}
\end{equation}

\end{document}