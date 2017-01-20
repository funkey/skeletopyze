# skeletopyze

A simple python module to skeletonize volumes given as numpy arrays.

# Install

## Using `conda`

```
conda config --add channels ukoethe
conda install -c funkey skeletopyze
```

## Build from source

Get this repository (and submodules):
```
git clone https://github.com/funkey/skeletopyze
cd skeletopyze
git submodule update --init
```

See the `meta.yaml` for build dependencies. Once fulfilled, build an install
with:
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

# Development

## Building the `conda` package

```
conda install conda-build\<2.0
conda config --add channels ukoethe
conda build --python 2.7 .
conda build --python 3.5 .
```
