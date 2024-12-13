# Mathematical Relationships in Fractal Field Theory

## Core Equations

### Field Equations
The fundamental field equation (Eq. 2.3 in main.tex):
```
(-∂²/∂t² + ∂²/∂x² - m²)ψ = λF[ψ]
```
relates to the fractal basis functions through:

### Basis Functions
The fractal basis functions (Eq. 2.7):
```
ψₙ(x) = αⁿ exp(-αⁿx²/2) Hₙ(αⁿx)
```
where Hₙ are Hermite polynomials.

### Coupling Evolution
The gauge coupling evolution (Eq. 3.12):
```
gᵢ(E) = g₀ᵢ(1 + α ln(E/E₀))^(-γᵢ)
```
with γᵢ being the fractal scaling exponents.

## Relationships

### Energy Scale Dependence
The mass term m² and coupling λ are dynamically generated:
```
m² = α² ∫|ψ|² dx
λ = α⁴
```

### Fractal Interaction
The fractal interaction term F[ψ] (Eq. 2.15):
```
F[ψ] = α Σₙ (ψₙ ∫ψ*ψₙdx)
```
couples different fractal levels.

### Normalization Conditions
The basis functions satisfy (Eq. 2.9):
```
∫ψₙ*(x)ψₘ(x)dx = δₙₘ
```

## Physical Observables

### Energy Density
The energy density (Eq. 4.3):
```
ε = |∂ₜψ|² + |∂ₓψ|² + m²|ψ|² + λ/4|ψ|⁴
```

### Fractal Dimension
The fractal dimension D_f (Eq. 5.7):
```
D_f = 2 + lim_{n→∞} ln(Σ αᵏ h(k))/ln(n)
```
where h(k) is the scaling function. 