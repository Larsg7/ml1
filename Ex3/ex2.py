import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


class Gaussian:
    def __init__(self, mu, tau):
        self.mu = mu
        self.tau = tau

    def gen(self):
        return np.random.normal(self.mu, 1/self.tau)


def plot_gaussian(x, mu, tau, label=None):
    plt.plot(x, mlab.normpdf(x, mu, 1 / tau), label=label)


def plot_hist(data, bins, label=None):
    plt.hist(data, bins, label=label, histtype="step")


g = Gaussian(-1, 2)
prior = Gaussian(1, 0.2)


def get_posterior(N, tau, tau_0, data, mu_0):
    tau_N = N * tau + tau_0
    mu_N = 1 / tau_N * (tau * sum(data) + tau_0 * mu_0)
    return mu_N, tau_N


Ns = [1, 5, 10, 50, 100]

x = np.linspace(-2, 4, 200)
plot_gaussian(x, 1, 0.2, label="prior")

for N in Ns:
    data = [g.gen() for _ in range(N)]
    mu_N, tau_N = get_posterior(N, g.tau, prior.tau, data, prior.mu)
    plot_gaussian(x, mu_N, tau_N, label=f"posterior, N = {N}")
    print(f"N = {N}")
    print(f"mu_N = {mu_N}")
    print(f"tau_N = {tau_N}")
    print()
    plt.legend()

plt.ylim(top=20, bottom=0)
plt.show()
