from collections import namedtuple

gaussian = namedtuple("Gaussian", ["mean", "var"])
gaussian.__repr__ = lambda s: f"N=(mean {s[0]:.3f}, var= {s[1]:.3f})"


def gaussian_add(g1, g2):
    return gaussian(g1.mean + g2.mean, g1.var + g2.var)


def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)


def update(likelihood, prior):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior


predicted_pos = gaussian(10.0, 5**2)
measurement_pos = gaussian(11.0, 10**2)

estimated_pos = update(predicted_pos, measurement_pos)
print(estimated_pos)