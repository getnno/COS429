import numpy as np
import scipy.signal
a = 2 * np.ones((5,5))
c = np.stack((a, -a))
b = np.ones()
d = scipy.signal.convolve(a, c, type = valid)
print(d.shape)
print(d)