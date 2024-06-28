# Binary signal recovery

## Binary signal recovery via maximum likelihood estimate

Let $\mathbf{X} \in \mathbb{R}^{m \times d}$ be a random sensing matrix with i.i.d. entries sampled from $\mathcal{N}(0, 1)$. Let $\boldsymbol{\xi} \in \mathbb{R}^{m}$ be a noise vector, independent of $\mathbf{X}$, with i.i.d. entries sampled from $\mathcal{N}(0, 1)$.

Take $\Theta = \{0, 1\}^{d}$ (signal space) and let $\boldsymbol{\theta} \in \Theta$ (signal) be chosen uniformly at random and be independent of the pair $(\mathbf{X}, \boldsymbol{\xi})$.

The measurement vector $\mathbf{y} \in \mathbb{R}^{m}$ is generated as
$\mathbf{y} = \mathbf{X}\boldsymbol{\theta} + \boldsymbol{\xi}.$

We want to recover the unknown vector $\boldsymbol{\theta}$ using Markov Chain Monte Carlo techniques, given the observations $(\mathbf{X}, \mathbf{y})$. We are interested in the case when $d$ is large. We recover $\boldsymbol{\theta}$ by finding the maximum likelihood estimate.

In the present setting, the maximum likelihood estimate of $\boldsymbol{\theta}$ is given by the value $\widehat{\boldsymbol{\theta}} \in \Theta$ that maximizes the likelihood function

$$
L(\mathbf{X}, \mathbf{y}; \boldsymbol{\theta}) = \frac{\exp\left(-\frac{1}{2}(\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^{\top}(\mathbf{y} - \mathbf{X}\boldsymbol{\theta})\right)}{(2\pi)^{m/2}},
$$


given the observations $(\mathbf{X}, \mathbf{y})$. Here the superscript $\top$ represents the transpose operation. We can equivalently cast the question in the form of a minimization problem. Indeed, the maximum likelihood estimate of $\boldsymbol{\theta}$ is given by the value $\widehat{\boldsymbol{\theta}} \in \Theta$ that minimizes the function

$$H(\mathbf{X}, \mathbf{y}; \boldsymbol{\theta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^{\top}(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}),$$

given the observations $(\mathbf{X}, \mathbf{y})$.

## Metropolis-Hastings algorithm

Let $\beta > 0$ be a fixed real parameter. We construct the Metropolis-Hastings (discrete-time) Markov chain on the state space $\Theta$, with stationary distribution

$$
\pi_{\beta}(\boldsymbol{\theta}) = \frac{\exp\{-\beta H(\mathbf{X}, \mathbf{y}; \boldsymbol{\theta})\}}{Z_{\beta}}
$$

$$
Z_{\beta} = \sum_{\boldsymbol{\theta} \in \Theta} \exp\{-\beta H(\mathbf{X}, \mathbf{y}; \boldsymbol{\theta})\}.
$$

Observe that the probability distribution $\pi_{\beta}$ concentrates on the maximum likelihood estimate as $\beta \to +\infty$. Therefore, if we choose $\beta$ sufficiently large and we run the chain for a large number $N$ of steps, we can take the state visited at time $N$ as the maximum likelihood estimate $\widehat{\boldsymbol{\theta}}$.

The following algorithm produces the first $N$ steps $\boldsymbol{\theta}_1, \ldots, \boldsymbol{\theta}_N$ of the Metropolis-Hasting chain on $\Theta$.

**Input:**
- Value of the parameter $\beta$
- Number of steps $N$; 
- Initial state $\bar{\boldsymbol{\theta}} \in \Theta$;

**Output:** trajectory of the Metropolis-Hastings chain starting at $\bar{\boldsymbol{\theta}}$;

**Procedure**

*Step 1.* Set $\boldsymbol{\theta}_0 = \bar{\boldsymbol{\theta}}$.

*Step 2.* For $t = 1, 2, \ldots, N - 1$:
   1. Pick $i$ uniformly at random in $\{1, 2, \ldots, d\}$.
   2. Let the proposed state be $\boldsymbol{\theta}^* \in \Theta$, with entries
      $$
      \boldsymbol{\theta}^*(j) = 
      \begin{cases}
          \boldsymbol{\theta}^{t-1}(j) & \text{if } j \neq i \\
          1 - \boldsymbol{\theta}^{t-1}(j) & \text{if } j = i
      \end{cases} \text{ for } j = 1, 2, \ldots, d.
      $$
   3. Set
      $$
      \boldsymbol{\theta}^t = 
      \begin{cases}
          \boldsymbol{\theta}^* & \text{with probability } \min\left\{1, \frac{\exp\{-\beta H(\mathbf{X}, \mathbf{y}; \boldsymbol{\theta}^*)\}}{\exp\{-\beta H(\mathbf{X}, \mathbf{y}; \boldsymbol{\theta}^{t-1})\}}\right\} \\
          \boldsymbol{\theta}^{t-1} & \text{with probability } 1 - \min\left\{1, \frac{\exp\{-\beta H(\mathbf{X}, \mathbf{y}; \boldsymbol{\theta}^*)\}}{\exp\{-\beta H(\mathbf{X}, \mathbf{y}; \boldsymbol{\theta}^{t-1})\}}\right\}
      \end{cases}.
      $$

## Project

By implementing the Metropolis-Hastings algorithm above, we determine an estimate $\hat{\theta}$ of a signal $\theta \in \Theta$ for any given realization of $(X, y)$. To check the quality of our estimate, we analyze the mean squared error

$$
\mathcal{E}=E\left((\hat{\theta}-\theta)^{\top}(\hat{\theta}-\theta)\right)
$$

where the expectation is over $\theta$ and $(X, y)$, for different values of $m$ (number of measurements).

Fix $d=10$. For every $1 \leq m \leq 15$, compute the mean squared error. Plot $\mathcal{E}$ as a function of $m$ and comment on the characteristics of your plot. What is the minimum value of $\frac{m}{d}$ required to reliably recover $\theta$ ?

Remark. The mean squared error $\mathcal{E}$ can be estimated by exploiting the law of large numbers. Let $M$ denote the number of independent realizations of $(\theta, X, y)$. Moreover, let $\hat{\theta}^{(j)}$ be the maximum likelihood estimate of the $j$-th signal $\theta^{(j)}$, obtained by the $j$-th run of the Metropolis-Hastings algorithm, given $\left(X^{(j)}, y^{(j)}\right)$. If $M$ is sufficiently large (use $M$ of order $10^{4}$ ), then we have the approximation

$$
\mathcal{E} \approx \frac{1}{M} \sum_{j=1}^{M}\left(\hat{\theta}^{(j)}-\theta^{(j)}\right)^{\top}\left(\hat{\theta}^{(j)}-\theta^{(j)}\right)
$$

## References

[1] Levin D.A. and Peres Y., Markov chains and mixing times, Volume 107, American Mathematical Society, 2017

[2] Ross S.M., Simulation, Academic Press, 2006