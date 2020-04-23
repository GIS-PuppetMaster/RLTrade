import numpy as np

x = np.array([i for i in range(-5, 6)])
print(x)

print(x + np.sign(x))
y = np.sign(x) * np.log10(np.abs(x + np.sign(x)))
y[np.isnan(y)] = 0
y[np.isinf(y)] = 0
print(y)
