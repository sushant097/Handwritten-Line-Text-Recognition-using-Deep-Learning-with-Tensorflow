import numpy as np

batch_size = 10
img_size = (800, 64)
random_matrix = np.random.random((batch_size, img_size[0], img_size[1]))

print(random_matrix)
print(random_matrix.shape)