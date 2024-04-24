from collections import namedtuple

gaussian = namedtuple("Gaussian", ["mean", "var"])
gaussian.__repr__ = lambda s: f"N=(mean {s[0]:.3f}, var= {s[1]:.3f})"


def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)


pos = gaussian(10.0, 0.2**2)
movement = gaussian(15.0, 0.7**2)

g = predict(pos, movement)
print(g)
