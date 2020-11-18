import numpy as np
import matplotlib.pyplot as plt

num_sims = 5  # Display five runs

t_init = 3
t_end = 7
N = 1000  # Compute 1000 grid points
dt = float(t_end - t_init) / N
y_init = 0

c_theta = 0.7
c_mu = 1.5
c_sigma = 1


def mu(y, t):
    """Implement the Ornstein–Uhlenbeck mu."""  # = \theta (\mu-Y_t)
    return c_theta * (c_mu - y)


def sigma(y, t):
    """Implement the Ornstein–Uhlenbeck sigma."""  # = \sigma
    return c_sigma


def dW(delta_t):
    """Sample a random number at each call."""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))


ts = np.arange(t_init, t_end + dt, dt)
ys = np.zeros(N + 1)

ys[0] = y_init

for _ in range(num_sims):
    for i in range(1, ts.size):
        t = (i - 1) * dt
        y = ys[i - 1]
        # 직전 값 y_{i-1}에서 dt 만큼
        ys[i] = y + mu(y, t) * dt + sigma(y, t) * dW(dt)
    plt.plot(ts, ys)

plt.xlabel("time (s)")
h = plt.ylabel("y")
plt.show()
