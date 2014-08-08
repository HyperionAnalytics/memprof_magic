### IPython Notebook magic for function line-by-line memory profiling

memprof IPython Notebook magic is based on [memory_profiler](https://github.com/fabianp/memory_profiler) by Fabian Pedregosa.

### Installation

```python
%install_ext https://raw.githubusercontent.com/HyperionAnalytics/memprof_magic/master/memprof.py
```

### Load the magic into the notebook

```python
%load_ext memprof
```

### Usage

```python
%memprof?
```

### Example

[IPython Notebook](http://nbviewer.ipython.org/github/HyperionAnalytics/memprof_magic/blob/master/line_mem_prof_ipnb.ipynb)