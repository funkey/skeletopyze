# skeletpyze

A simple python module to skeletonize volumes given as numpy arrays.

# Install

```
python setup.py install
```

# Example usage

```python
import skeletopyze
import numpy as np

a = np.zeros((100,100,100), dtype=np.int32)
a[50,50,:] = 1
a[48:52,48:52,48:52] = 1
a[46:54,46:54,70:78] = 1

params = skeletopyze.Parameters()

print("Skeletonizing")
b = skeletopyze.get_skeleton_graph(a, params)

print("Skeleton contains nodes:")
for n in b.nodes():
    print str(n) + ": " + "(%d, %d, %d), diameter %f"%(b.locations(n).x(), b.locations(n).y(), b.locations(n).z(), b.diameters(n))

print("Skeleton contains edges:")
for e in b.edges():
    print (e.u, e.v)
```
