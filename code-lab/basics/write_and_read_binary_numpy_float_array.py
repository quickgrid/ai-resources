## Save binary file using tofile and read using fromfile
## Used for sending numpy array to C++

import numpy as np


## Save numpy array to file
fname = 'temp/chosen.dat'
a = [[1.23, 34.5677], [32.5, 235.56], [12.4365756436, 3474.2366236322398]]
print(a)
a = np.array(a, dtype=np.float32)
a.tofile(fname)


## Read the saved numpy array binary file
val = np.fromfile(fname, dtype=np.float32)
print(val)


## print the values one by one
mat_len = len(a[0]) * len(a)
for i in range(0, mat_len):
    print(val[i], end=' ')
