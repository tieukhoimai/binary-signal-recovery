# Binary signal recovery

## I. Generate data

- $\mathbf{X} \in \mathbb{R}^{m \times d}$ be a random sensing matrix with i.i.d. entries sampled from $\mathcal{N}(0, 1)$.
- $\boldsymbol{\xi} \in \mathbb{R}^{m}$ be a noise vector, independent of $\mathbf{X}$, with i.i.d. entries sampled from $\mathcal{N}(0, 1)$
- $\Theta = \{0, 1\}^{d}$ (signal space) and let $\boldsymbol{\theta} \in \Theta$ (signal) be chosen uniformly at random and be independent of the pair $(\mathbf{X}, \boldsymbol{\xi})$.

The measurement vector $\mathbf{y} \in \mathbb{R}^{m}$ is generated as

$$\mathbf{y} = \mathbf{X}\boldsymbol{\theta} + \boldsymbol{\xi}.$$

```python
def generate_data(m, d):
  X = np.random.randn(m, d)  # Sensing matrix
  xi = np.random.randn(m)   # Noise
  theta = np.random.randint(2, size=d)  # Signal
  y = X @ theta + xi  # Measurements
  return X, y, theta
```

## II. Define the likelihood and energy functions

**Objective:** We need to recover the unknown vector $\boldsymbol{\theta}$ using Markov Chain Monte Carlo techniques, given the observations $(\mathbf{X}, \mathbf{y})$. We are interested in the case when $d$ is large. We recover $\boldsymbol{\theta}$ by finding the maximum likelihood estimate.

### 1. Likelihood Function ($\mathcal{L}$):

- **Assumption:** The noise vector $\boldsymbol{\xi}$ follows a multivariate normal distribution with zero mean and identity covariance matrix, denoted as $\boldsymbol{\xi} \sim \mathcal{N}(0,I)$.
- **Measurement Model:** $\mathbf{y} = \mathbf{X} \boldsymbol{\theta} + \boldsymbol{\xi}$

Since $\boldsymbol{\xi}$ is a multivariate normal, its probability density function (pdf) is:

$$\mathcal{L}(\xi) = (2\pi)^{-m/2} \times exp\{-\frac{1}{2} \xi^T \xi\}$$

Now, we want to find the probability of observing $y$ given $X$ and $\theta$, which is equivalent to finding the probability of the noise ξ being equal to $\mathbf{y} - \mathbf{X} \boldsymbol{\theta}$:

$$\mathcal{L}(y | X, \theta) = p(\xi = y - X\theta)
                   = (2\pi)^{-m/2} \times exp\{-\frac{1}{2} (y - X\theta)^T (y - X\theta)\}$$


This is the likelihood function $\mathcal{L}(X, y; \theta)$:

$$\mathcal{L}(X, y; \theta) = \frac{exp\{-\frac{1}{2} (y - X \theta)^T (y - X\theta)\}}{(2\pi)^{m/2}}$$

### 2. Minimization Function ($H$):

- To maximize this function, we can maximize the logarithm of this function. Take the logarithm, we have:
  
$$
\log\mathcal{L}(X, y ; \theta)=-\frac{1}{2}(y-X \theta)^{\top}(y-X \theta)-\dfrac{m}{2}\log(2 \pi),
$$

- We can equivalently cast the question in the form of a minimization problem. Indeed, the maximum likelihood estimate of $\theta$ is given by the value $\hat{\theta} \in \Theta$ that minimizes the function

$$
H(X, y ; \theta)=-\left [-(y-X\theta)^{\top}(y-X\theta)\right ]=(y-X\theta)^{\top}(y-X\theta),
$$

given the observations $(X, y)$, and $\dfrac{1}{2}, \dfrac{m}{2}\log(2 \pi)$ are constants, so we can vanish them.

- To simplify the optimization problem, we define a new function $H(X, y; \theta)$ as the exponent:

$$H(X, y; \theta) = (y - X\theta)^T (y - X\theta)$$

By minimizing $H(X, y; \theta)$, we essentially maximize the likelihood function $\mathcal{L}(X, y; \theta)$ and find the maximum likelihood estimate of $\theta$.

**In summary:**

- **Maximize:** $\mathcal{L}(\xi) = (2\pi)^{-m/2} \times exp\{-\frac{1}{2} \xi^T \xi\}$
- **Minimize:** $H(X, y; \theta) = (y - X\theta)^T (y - X\theta)$


## III. Metropolis-Hastings algorithm

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

<img src="img/thumbnail.png" alt="thumbnail" width="750"/>

<!-- **Procedure**

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
      $$ -->

```python
def metropolis_hastings(X, y, beta, M, initial_theta):
  d = len(initial_theta)
  theta = initial_theta.copy()

  for _ in range(M):
    i = np.random.randint(d)
    theta_star = theta.copy()
    theta_star[i] = 1 - theta_star[i]

    acceptance_prob = min(1, np.exp(-beta * (H(X, y, theta_star) - H(X, y, theta))))

    if np.random.rand() < acceptance_prob:
      theta = theta_star

  return theta
```

## IV. Experiment

By implementing the Metropolis-Hastings algorithm above, we determine an estimate $\hat{\theta}$ of a signal $\theta \in \Theta$ for any given realization of $(X, y)$. To check the quality of our estimate, we analyze the mean squared error

$$
\mathcal{E}=E\left((\hat{\theta}-\theta)^{\top}(\hat{\theta}-\theta)\right)
$$

where the expectation is over $\theta$ and $(X, y)$, for different values of $m$ (number of measurements).

Fix $d=10$. For every $1 \leq m \leq 15$, compute the mean squared error. Note that the mean squared error $\mathcal{E}$ is estimated by exploiting the law of large numbers. Let $M$ denote the number of independent realizations of $(\theta, X, y)$. Moreover, $\hat{\theta}^{(j)}$ be the maximum likelihood estimate of the $j$-th signal $\theta^{(j)}$, obtained by the $j$-th run of the Metropolis-Hastings algorithm, given $\left(X^{(j)}, y^{(j)}\right)$. If $M$ is sufficiently large (use $M$ of order $10^{4}$ ), then we have the approximation

$$
\mathcal{E} \approx \frac{1}{M} \sum_{j=1}^{M}\left(\hat{\theta}^{(j)}-\theta^{(j)}\right)^{\top}\left(\hat{\theta}^{(j)}-\theta^{(j)}\right)
$$

```python
def experiment(d=10, m_max=15, N=100, M=100, beta=1.0):
  start_time = time.time()
  ms = range(1, m_max + 1)
  MSEs = []

  for m in ms:
      mse = 0
      for _ in range(N):
          X, y, theta_true = generate_data(m, d)
          initial_theta = np.random.randint(2, size=d)
          theta_hat = metropolis_hastings(X, y, beta, M, initial_theta)
          mse += ((theta_hat - theta_true).T @ (theta_hat - theta_true)) / M
      MSEs.append(mse)

  end_time = time.time()
  avg_time = (end_time - start_time)

  return ms, MSEs, avg_time, N, M, beta
```

### 1. Experiment with $\beta$

```python
results = []

# Loop through the beta values
beta_values = np.linspace(0.1, 2.0, 5)

for beta_i in beta_values:
  ms, MSEs, avg_time, N, M, beta = experiment(d=10, m_max=15, N=1000, M=1000, beta=beta_i)

  # Store the results in the list
  results.append({
        'M': M,
        'N': N,
        'beta': beta,
        'm': list(ms),
        'MSE': MSEs,
        'Time': avg_time
    })

# Create a pandas DataFrame from the results list
df = pd.DataFrame(results)

def convert_str_to_arr(str_value):
  return [float(x.strip()) for x in str_value.strip('[]').split(',')]

# Plot the result
fig, ax = plt.subplots(figsize=(12, 6))

for i in range(0,len(df)):
  m = convert_str_to_arr(df['m'][i])
  mse = convert_str_to_arr(df['MSE'][i])
  label = r'$\beta = {0:.2f}$ , $M = {1:.0f}$, $N = {2:.0f}$, $Minimum MSE = {3:.2f}$'.format(df['beta'][i], df['M'][i], df['N'][i], min(mse))
  sns.lineplot(x=m, y=mse, marker='o',label=label)

ax.set_xlabel('m')
ax.set_ylabel('MSE')
plt.show()
```

![beta Experiment](img/beta-exp-result.png)


```python
# Plot time complexity

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.lineplot(data=df, x='beta', y='Time', marker='s')
ax.set_xticks(beta_values)
ax.set_xticklabels([r'$\beta = {:.2f}$'.format(beta) for beta in beta_values])
ax.set_xlabel(r'$\beta$')
ax.set_ylabel("Average Time (s)")
ax.grid(False)
```

![Time Complexity of Beta](img/beta-exp-time.png)

### 2. Experiment with $M$

```python
# Create an empty list to store the results
m_results = []

# Loop through the beta values
M_values = np.linspace(1000, 10000, 5, dtype=int)

for M_i in M_values:
  ms, MSEs, avg_time, N, M, beta = experiment(d=10, m_max=15, N=1000, M=M_i, beta=1.0)

  # Store the results in the list
  m_results.append({
        'M': M,
        'N': N,
        'beta': beta,
        'm': list(ms),
        'MSE': MSEs,
        'Time': avg_time
    })

# Create a pandas DataFrame from the results list
df_m = pd.DataFrame(m_results)

# Plot the result
ms = range(1, 16)

fig, ax = plt.subplots(figsize=(12, 6))

for i in range(0,len(df)):
  label = r'$\beta = {0:.2f}$ , $M = {1:.0f}$, $N = {2:.0f}$, $Minimum MSE = {3:.2f}$'.format(df_m['beta'][i], df_m['M'][i], df_m['N'][i], min(df_m['MSE'][i]))
  sns.lineplot(x=df_m['m'][i], y=df_m['MSE'][i], marker='o',label=label)

ax.set_xlabel('m')
ax.set_ylabel('MSE')
plt.show()
```

![M Experiment](img/M-exp-result.png)

```python
# Plot time complexity

fig, ax = plt.subplots(figsize=(12, 6))
ax = sns.lineplot(data=df, x='beta', y='Time', marker='s')
ax.set_xticks(beta_values)
ax.set_xticklabels([r'$\beta = {:.2f}$'.format(beta) for beta in beta_values])
ax.set_xlabel(r'$\beta$')
ax.set_ylabel("Average Time (s)")
ax.grid(False)
```

![Time Complexity of M](img/M-exp-time.png)

In conclusion, the plot highlights the trade-off between the number of measurements, the parameter  $\beta$, $M$, the accuracy of the recovered signal and time complexity of algorithm. While a higher $m$, an appropriate  $\beta$ and sufficiently large $M$ can improve the recovery, defining a universal minimum $m/d$ for reliable recovery depends on the specific application requirements and acceptable error tolerance.
