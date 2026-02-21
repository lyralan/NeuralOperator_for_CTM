# Hybrid Neural Operator Surrogates for Differentiable Atmospheric Transport Modeling

------------------------------------------------------------------------

# Part 1 --- Mini-Project Proposal

## Motivation

Atmospheric chemistry transport models (CTMs) solve high-dimensional PDE
systems governing pollutant transport and diffusion. These models are:

-   Computationally expensive\
-   Used in inverse modeling and data assimilation\
-   Embedded in PDE-constrained optimization frameworks\
-   Differentiated via adjoint methods

Recent advances in neural operators (e.g., Fourier Neural Operators)
enable learning mappings between function spaces and have shown strong
performance in fluid dynamics and climate emulation.

This project explores:

Can neural operators serve as differentiable, stable surrogates for
atmospheric transport PDE components while preserving gradient structure
relevant to inverse modeling?

------------------------------------------------------------------------

## Problem Setup

We consider a simplified 2D advection-diffusion PDE:

∂c/∂t + u·∇c = D∇²c + S

where: - c(x,y,t) = pollutant concentration\
- u = wind field\
- D = diffusion coefficient\
- S = emission source term

We aim to learn an operator:

G: (c₀, u, D, S) → c_T

using a Fourier Neural Operator (FNO).

------------------------------------------------------------------------

## Research Questions

1.  Can neural operators approximate atmospheric transport across
    varying regimes?
2.  Does the surrogate preserve physically meaningful gradient
    structure?
3.  Can hybrid physics--ML decomposition improve stability?
4.  How does surrogate accuracy degrade under long rollout horizons?

------------------------------------------------------------------------

## Proposed Contributions

1.  Implementation of a neural operator surrogate for atmospheric
    transport PDE.
2.  Evaluation of gradient fidelity compared to PDE-based gradients.
3.  Hybrid modeling: physical advection + learned diffusion module.
4.  Stability and generalization analysis across parameter regimes.

------------------------------------------------------------------------

# Part 2 --- Experimental Plan and Metrics

## Step 1: Baseline PDE Simulation

-   2D periodic domain
-   Vary wind fields, diffusion coefficients, emission patterns
-   Generate 5k--20k trajectories

------------------------------------------------------------------------

## Step 2: Train Models

Models: - Fourier Neural Operator (FNO) - U-Net baseline - CNN
baseline - Hybrid physics--ML model

------------------------------------------------------------------------

## Evaluation Metrics

### Field Reconstruction

-   L2 error
-   Relative L2
-   MAE
-   Spectral error

### Physical Consistency

-   Mass conservation error
-   Long-rollout stability
-   Error growth rate

### Generalization

-   Unseen wind regimes
-   Different diffusion ranges

### Gradient Fidelity (Key Contribution)

-   Cosine similarity between true and surrogate gradients
-   Relative gradient L2 error
-   Inverse problem recovery performance

### Hybrid Model Comparison

-   Stability vs full surrogate
-   Gradient fidelity preservation

------------------------------------------------------------------------

# Part 3 --- Mapping to WeatherNext2 / DeepMind Research

Relevant research directions:

-   GraphCast (structured operator learning)
-   NeuralGCM (hybrid physics--ML climate modeling)
-   FourCastNet (spectral operator learning for weather)

This project aligns through:

-   Operator learning for atmospheric dynamics
-   Hybrid physics--ML modeling
-   Differentiable modeling for inverse problems
-   Structured spatiotemporal systems

------------------------------------------------------------------------

# Part 4 --- Structured Reading List

## Neural Operators

1.  Fourier Neural Operator (Li et al., 2020)
2.  DeepONet (Lu et al., 2021)
3.  Neural Operator Survey (Kovachki et al., 2023)

## Scientific ML Foundations

4.  Physics-Informed Neural Networks (Karniadakis et al.)
5.  Integrating Physics and ML (Willard et al.)
6.  Scientific Machine Learning overview (Rackauckas et al.)

## Weather / Climate ML

7.  GraphCast (DeepMind)
8.  NeuralGCM
9.  FourCastNet
10. Pangu-Weather

## Differentiable Modeling & Data Assimilation

11. 4D-Var tutorials
12. Hybrid DA + ML methods
13. Differentiable physics simulation surveys

------------------------------------------------------------------------

# Week-by-Week Execution Roadmap (6 Months)

## Weeks 1--2

-   Implement 2D advection-diffusion solver
-   Add unit tests and visualization tools

## Weeks 3--4

-   Build dataset generator
-   Train CNN and U-Net baselines

## Weeks 5--6

-   Implement Fourier Neural Operator
-   Compare one-step prediction performance

## Weeks 7--8

-   Test generalization across regimes
-   Evaluate rollout stability

## Weeks 9--10

-   Implement hybrid physics--ML diffusion surrogate

## Weeks 11--12

-   Implement inverse problem setup
-   Compare true PDE gradients vs surrogate gradients

## Weeks 13--14

-   Conduct ablation studies
-   Evaluate resolution scaling

## Weeks 15--16

-   Finalize figures and experiment design
-   Draft workshop outline

## Weeks 17--18

-   Write full paper draft
-   Run final experiment sweeps

## Weeks 19--20

-   Clean repository and documentation
-   Prepare reproducibility package

## Weeks 21--24

-   Final revisions
-   Workshop submission
-   Prepare internship-facing project summary

------------------------------------------------------------------------

End of Document
