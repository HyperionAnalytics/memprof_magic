"""
Milos Miljkovic 2014

memprof.py
version 1.0.0

IPython magic to memory profile a function line by line.

Installation:

  %install_ext https://raw.githubusercontent.com//HyperionAnalytics/memprof_magic/master/memprof.py

Usage:

  %load_ext memprof

  %memprof

arguments:

  -f FUNCTION,  --function FUNCTION      string to evaluate function call
  -p PRECISION, --precision PRECISION    integer to set number of decimal places in resuts

Examples:

  %memprof -f 'my_function(arg1, arg2)' -p 2
"""
import time
import sys
import os
import warnings
import linecache
import inspect
import subprocess
import inspect
from io import StringIO

import IPython
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

try:
    from multiprocessing import Process, Pipe
except ImportError:
    from multiprocessing.dummy import Process, Pipe

has_psutil = False
try:
    import psutil
    has_psutil = True
except ImportError:
    pass

TWO_20 = float(2 ** 20)

def get_memory(pid, include_children=False):
    if pid == -1:
        pid = os.getpid()

    if has_psutil:
        process = psutil.Process(pid)
        try:
            mem_info = getattr(process, 'memory_info', process.get_memory_info)
            mem = mem_info()[0] / TWO_20
            if include_children:
                for p in process.get_children(recursive=True):
                    mem_info = getattr(p, 'memory_info', p.get_memory_info)
                    mem += mem_info()[0] / TWO_20
            else:
                return mem
        except psutil.AccessDenied:
            print('Access denied to psutil.')
            pass
    else:
        msg = 'The psutil module is required for memory profiling.'
        raise NotImplementedError(msg)

def format_results(prof, fcn_eval, precision=1):
    stream = sys.stdout
    template = '{0:>6} {1:>12} {2:>12}   {3:<}'

    for code in prof.code_map:
        lines = prof.code_map[code]
        if not lines:
            print('Measurements are empty.')
            continue

        b = inspect.getsource(fcn_eval)
        all_lines = b.splitlines(keepends=True)
        sub_lines = inspect.getblock(all_lines[code.co_firstlineno - 1:])
        linenos = range(code.co_firstlineno,
                        code.co_firstlineno + len(sub_lines))

        header = template.format('Line #', 'Mem usage', 'Increment', 'Line Contents')
        stream.write(header + '\n')
        stream.write('=' * len(header) + '\n')

        mem_old = lines[min(lines.keys())]
        float_format = '{0}.{1}f'.format(precision + 4, precision)
        template_mem = '{0:' + float_format + '} MiB'
        for line in linenos:
            mem = ''
            inc = ''
            if line in lines:
                mem = lines[line]
                inc = mem - mem_old
                mem_old = mem
                mem = template_mem.format(mem)
                inc = template_mem.format(inc)
            stream.write(template.format(line, mem, inc, all_lines[line - 1]))
        stream.write('\n\n')


class LineProfiler(object):
    """ A profiler that records the amount of memory for each line."""
    def __init__(self, **kw):
        self.code_map = {}
        self.enable_count = 0
        self.prevline = None
        self.include_children = kw.get('include_children', False)

    def __call__(self, func):
        self.add_function(func)
        f = self.wrap_function(func)
        f.__module__ = func.__module__
        f.__name__ = func.__name__
        f.__doc__ = func.__doc__
        f.__dict__.update(getattr(func, '__dict__', {}))
        return f

    def add_code(self, code, toplevel_code=None):
        if code not in self.code_map:
            self.code_map[code] = {}
            for subcode in filter(inspect.iscode, code.co_consts):
                self.add_code(subcode)

    def add_function(self, func):
        """Record line profiling information for the given Python function."""
        try:
            # func_code does not exist in Python3
            code = func.__code__
        except AttributeError:
            warnings.warn("Could not extract a code object for the object %r" % func)
        else:
            self.add_code(code)

    def wrap_function(self, func):
        """Wrap a function to profile it."""
        def f(*args, **kwds):
            self.enable_by_count()
            try:
                result = func(*args, **kwds)
            finally:
                self.disable_by_count()
            return result
        return f

    def runctx(self, cmd, globals, locals):
        """Profile a single executable statement in the given namespaces."""
        self.enable_by_count()
        try:
            exec(cmd, globals, locals)
        finally:
            self.disable_by_count()
        return self

    def enable_by_count(self):
        """ Enable the profiler if it hasn't been enabled before."""
        if self.enable_count == 0:
            self.enable()
        self.enable_count += 1

    def disable_by_count(self):
        """Disable the profiler if the number of disable requests matches the
           number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def trace_memory_usage(self, frame, event, arg):
        """ Callback for sys.settrace."""
        if (event in ('call', 'line', 'return')
                and frame.f_code in self.code_map):
            if event != 'call':
                # "call" event just saves the lineno but not the memory
                mem = get_memory(-1, include_children=self.include_children)
                # if there is already a measurement for that line get the max
                old_mem = self.code_map[frame.f_code].get(self.prevline, 0)
                self.code_map[frame.f_code][self.prevline] = max(mem, old_mem)
            self.prevline = frame.f_lineno
        if self._original_trace_function is not None:
            (self._original_trace_function)(frame, event, arg)
        return self.trace_memory_usage

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()

    def enable(self):
        self._original_trace_function = sys.gettrace()
        sys.settrace(self.trace_memory_usage)

    def disable(self):
        sys.settrace(self._original_trace_function)


@magics_class
class MemProf(Magics):
    """IPython magic to to memory profile a function line by line."""
    @magic_arguments()
    @argument('-f', '--function', type=str, help='function to evaluate: string')
    @argument('-p', '--precision', type=int, help='set number of decimal places: integer')
    @line_magic
    def memprof(self, line):
        args = parse_argstring(self.memprof, line)
        if args.function:
            fcn_eval_string = args.function.strip('\'"')
            fcn_name_string = fcn_eval_string.split('(')[0]
        if args.precision:
            kwv1 = args.precision
        else:
            kwv1 = 1

        profile = LineProfiler()
        global_ns = self.shell.user_global_ns
        local_ns = self.shell.user_ns
        fcn_eval = global_ns[fcn_name_string]
        profile(fcn_eval)

        try:
            import builtins
        except ImportError:
            import __builtin__ as builtins

        if 'profile' in builtins.__dict__:
            had_profile = True
            old_profile = builtins.__dict__['profile']
        else:
            had_profile = False
            old_profile = None
        builtins.__dict__['profile'] = profile

        try:
            try:
                profile.runctx(fcn_eval_string, global_ns, local_ns)
            except SystemExit:
                print('SystemExit exception caught in code being profiled.')
            except KeyboardInterrupt:
                print('KeyboardInterrupt exception caught in code being profiled.')
        finally:
            if had_profile:
                builtins.__dict__['profile'] = old_profile

        format_results(profile, fcn_eval, precision=kwv1)

def load_ipython_extension(ipython):
    ipython.register_magics(MemProf)
