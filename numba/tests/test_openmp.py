import math
import time
import dis
import numbers
import os
import platform
import sys
import subprocess
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp

from numba import njit, typeof
from numba.core import (types, utils, typing, errors, ir, rewrites,
                        typed_passes, inline_closurecall, config, compiler, cpu)
from numba.extending import (overload_method, register_model,
                             typeof_impl, unbox, NativeValue, models)
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
                            get_definition, is_getitem, is_setitem,
                            index_var_of_get_setitem)
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.bytecode import ByteCodeIter
from numba.core.compiler import (compile_isolated, Flags, CompilerBase,
                                 DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
                      override_env_config, linux_only, tag, _32bit, needs_blas,
                      needs_lapack, disabled_test, skip_unless_scipy,
                      needs_subprocess, override_config)
from numba.openmp import openmp_context as openmp
from numba.openmp import (omp_set_num_threads, omp_get_thread_num,
                    omp_get_num_threads, omp_get_wtime, omp_set_nested,
                    omp_set_max_active_levels, omp_set_dynamic,
                    omp_get_max_active_levels, omp_get_max_threads,
                    omp_get_num_procs, UnspecifiedVarInDefaultNone,
                    NonconstantOpenmpSpecification,
                    NonStringOpenmpSpecification,
                    ParallelForExtraCode, ParallelForWrongLoopCount,
                    omp_in_parallel)
import cmath
import unittest

# NOTE: Each OpenMP test class is run in separate subprocess, this is to reduce
# memory pressure in CI settings. The environment variable "SUBPROC_TEST" is
# used to determine whether a test is skipped or not, such that if you want to
# run any OpenMP test directly this environment variable can be set. The
# subprocesses running the test classes set this environment variable as the new
# process starts which enables the tests within the process. The decorator
# @needs_subprocess is used to ensure the appropriate test skips are made.

@linux_only
class TestOpenmpRunner(TestCase):
    _numba_parallel_test_ = False

    # Each test class can run for 30 minutes before time out.
    _TIMEOUT = 1800

    """This is the test runner for all the parfors tests, it runs them in
    subprocesses as described above. The convention for the test method naming
    is: `test_<TestClass>` where <TestClass> is the name of the test class in
    this module.
    """
    def runner(self):
        themod = self.__module__
        test_clazz_name = self.id().split('.')[-1].split('_')[-1]
        # don't specify a given test, it's an entire class that needs running
        self.subprocess_test_runner(test_module=themod,
                                    test_class=test_clazz_name,
                                    timeout=self._TIMEOUT)

    """
    def test_TestOpenmpBasic(self):
        self.runner()
    """

    def test_TestOpenmpRuntimeFunctions(self):
        self.runner()

    def test_TestOpenmpParallelForResults(self):
        self.runner()

    def test_TestOpenmpForSchedule(self):
        self.runner()

    def test_TestOpenmpParallelClauses(self):
        self.runner()

    def test_TestOpenmpDataClauses(self):
        self.runner()

    def test_TestOpenmpConstraints(self):
        self.runner()

    def test_TestOpenmpConcurrency(self):
        self.runner()

    def test_TestOpenmpTask(self):
        self.runner()

    def test_TestOpenmpTaskloop(self):
        self.runner()

    def test_TestOpenmpPi(self):
        self.runner()

    

x86_only = unittest.skipIf(platform.machine() not in ('i386', 'x86_64'), 'x86 only test')

def null_comparer(a, b):
    """
    Used with check_arq_equality to indicate that we do not care
    whether the value of the parameter at the end of the function
    has a particular value.
    """
    pass


@needs_subprocess
class TestOpenmpBase(TestCase):
    """
    Base class for testing OpenMP.
    Provides functions for compilation and three way comparison between
    python functions, njit'd functions and njit'd functions with
    OpenMP disabled.
    """

    _numba_parallel_test_ = False

    def __init__(self, *args):
        # flags for njit()
        self.cflags = Flags()
        self.cflags.nrt = True

        super(TestOpenmpBase, self).__init__(*args)

    def _compile_this(self, func, sig, flags):
        return compile_isolated(func, sig, flags=flags)

    def compile_njit_openmp_disabled(self, func, sig):
        with override_config('OPENMP_DISABLED', True):
            return self._compile_this(func, sig, flags=self.cflags)

    def compile_njit(self, func, sig):
        return self._compile_this(func, sig, flags=self.cflags)

    def compile_all(self, pyfunc, *args, **kwargs):
        sig = tuple([typeof(x) for x in args])

        # compile the OpenMP-disabled njit function
        cdfunc = self.compile_njit_openmp_disabled(pyfunc, sig)

        # compile a standard njit of the original function
        cfunc = self.compile_njit(pyfunc, sig)

        return cfunc, cdfunc

    def assert_outputs_equal(self, *outputs):
        assert(len(outputs) > 1)

        for op_num in range(len(outputs)-1):
            op1, op2 = outputs[op_num], outputs[op_num+1]
            if isinstance(op1, (bool, np.bool_)):
                assert isinstance(op2, (bool, np.bool_))
            elif (not isinstance(op1, numbers.Number) or
                not isinstance(op2, numbers.Number)):
                self.assertEqual(type(op1), type(op2))

            if isinstance(op1, np.ndarray):
                np.testing.assert_almost_equal(op1, op2)
            elif isinstance(op1, (tuple, list)):
                assert(len(op1) == len(op2))
                for i in range(len(op1)):
                    self.assert_outputs_equal(op1[i], op2[i])
            elif isinstance(op1, (bool, np.bool_, str, type(None))):
                assert(op1 == op2)
            elif isinstance(op1, numbers.Number):
                np.testing.assert_approx_equal(op1, op2)
            else:
                raise ValueError('Unsupported output type encountered')

    def check_openmp_vs_others(self, pyfunc, cfunc, cdfunc, *args, **kwargs):
        """
        Checks python, njit and njit without OpenMP impls produce the same result.

        Arguments:
            pyfunc - the python function to test
            cfunc - CompilerResult from njit of pyfunc
            cdfunc - CompilerResult from OpenMP-disabled njit of pyfunc
            args - arguments for the function being tested
        Keyword Arguments:
            scheduler_type - 'signed', 'unsigned' or None, default is None.
                           Supply in cases where the presence of a specific
                           scheduler is to be asserted.
            fastmath_pcres - a fastmath parallel compile result, if supplied
                             will be run to make sure the result is correct
            check_arg_equality - some functions need to check that a
                                 parameter is modified rather than a certain
                                 value returned.  If this keyword argument
                                 is supplied, it should be a list of
                                 comparison functions such that the i'th
                                 function in the list is used to compare the
                                 i'th parameter of the njit and OpenMP-disabled
                                 functions against the i'th parameter of the
                                 standard Python function, asserting if they
                                 differ.  The length of this list must be equal
                                 to the number of parameters to the function.
                                 The null comparator is available for use
                                 when you do not desire to test if some
                                 particular parameter is changed.
            Remaining kwargs are passed to np.testing.assert_almost_equal
        """
        check_args_for_equality = kwargs.pop('check_arg_equality', None)

        def copy_args(*args):
            if not args:
                return tuple()
            new_args = []
            for x in args:
                if isinstance(x, np.ndarray):
                    new_args.append(x.copy('k'))
                elif isinstance(x, np.number):
                    new_args.append(x.copy())
                elif isinstance(x, numbers.Number):
                    new_args.append(x)
                elif isinstance(x, tuple):
                    new_args.append(copy.deepcopy(x))
                elif isinstance(x, list):
                    new_args.append(x[:])
                elif isinstance(x, str):
                    new_args.append(x)
                else:
                    raise ValueError('Unsupported argument type encountered')
            return tuple(new_args)

        # python result
        py_args = copy_args(*args)
        py_expected = pyfunc(*py_args)

        # njit result
        njit_args = copy_args(*args)
        njit_output = cfunc.entry_point(*njit_args)

        # OpenMP-disabled result
        openmp_disabled_args = copy_args(*args)
        openmp_disabled_output = cdfunc.entry_point(*openmp_disabled_args)

        if check_args_for_equality is None:
            self.assert_outputs_equal(py_expected, njit_output, openmp_disabled_output)
        else:
            assert(len(py_args) == len(check_args_for_equality))
            for pyarg, njitarg, noomparg, argcomp in zip(
                py_args, njit_args, openmp_disabled_args,
                check_args_for_equality):
                argcomp(njitarg, pyarg, **kwargs)
                argcomp(noomparg, pyarg, **kwargs)

    def check(self, pyfunc, *args, **kwargs):
        """Checks that pyfunc compiles for *args under njit OpenMP-disabled and
        njit and asserts that all version execute and produce the same result
        """
        cfunc, cdfunc = self.compile_all(pyfunc, *args)
        self.check_openmp_vs_others(pyfunc, cfunc, cdfunc, *args, **kwargs)

    def check_variants(self, impl, arg_gen, **kwargs):
        """Run self.check(impl, ...) on array data generated from arg_gen.
        """
        for args in arg_gen():
            with self.subTest(list(map(typeof, args))):
                self.check(impl, *args, **kwargs)


class TestPipeline(object):
    def __init__(self, typingctx, targetctx, args, test_ir):
        self.state = compiler.StateDict()
        self.state.typingctx = typingctx
        self.state.targetctx = targetctx
        self.state.args = args
        self.state.func_ir = test_ir
        self.state.typemap = None
        self.state.return_type = None
        self.state.calltypes = None
        self.state.metadata = {}


@linux_only
class TestOpenmpBasic(TestOpenmpBase):
    """OpenMP smoke tests. These tests check the most basic
    functionality"""

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)


@linux_only
class TestOpenmpRuntimeFunctions(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    """
    def test_func_get_wtime(self):
        @njit
        def test_impl(t):
            start = omp_get_wtime()
            time.sleep(t)
            return omp_get_wtime() - start
        t = 0.5
        np.testing.assert_approx_equal(test_impl(t), t, signifcant=2)
    """

    def test_func_get_max_threads(self):
        @njit
        def test_impl():
            o_nt = omp_get_max_threads()
            count = 0
            with openmp("parallel"):
                i_nt = omp_get_max_threads()
                with openmp("critical"):
                    count += 1
            return count, i_nt, o_nt
        r = test_impl()
        assert(r[0] == r[1] == r[2])

    def test_func_get_num_threads(self):
        @njit
        def test_impl():
            o_nt = omp_get_num_threads()
            count = 0
            with openmp("parallel"):
                i_nt = omp_get_num_threads()
                with openmp("critical"):
                    count += 1
            return (count, i_nt), o_nt
        r = test_impl()
        assert(r[0][0] == r[0][1])
        assert(r[1] == 1)

    def test_func_set_num_threads(self):
        @njit
        def test_impl(N):
            omp_set_num_threads(N)
            count = 0
            with openmp("parallel"):
                with openmp("critical"):
                    count += 1
            return count
        nt = 32
        assert(test_impl(32) == 32)

    def test_func_in_parallel(self):
        @njit
        def test_impl():
            oa = omp_in_parallel()
            with openmp("parallel num_threads(1)"):
                ia = omp_in_parallel()
            return oa, ia
        self.assert_outputs_equal(test_impl(), (False, True))


@linux_only
class TestOpenmpParallelForResults(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_parallel_for_set_elements(self):
        def test_impl(v):
            with openmp("parallel for"):
                for i in range(len(v)):
                    v[i] = 1.0
            return v
        self.check(test_impl, np.zeros(100))

    def test_separate_parallel_for_set_elements(self):
        def test_impl(v):
            with openmp("parallel"):
                with openmp("for"):
                    for i in range(len(v)):
                        v[i] = 1.0
            return v
        self.check(test_impl, np.zeros(100))

    def test_parallel_for_const_var_omp_statement(self):
        def test_impl(v):
            ovar = "parallel for"
            with openmp(ovar):
                for i in range(len(v)):
                    v[i] = 1.0
            return v
        self.check(test_impl, np.zeros(100))

    def test_parallel_for_string_conditional(self):
        def test_impl(S):
            capitalLetters = 0
            with openmp("parallel for reduction(+:capitalLetters)"):
                for i in range(len(S)):
                    if S[i].isupper():
                        capitalLetters += 1
            return capitalLetters
        self.check(test_impl, "OpenMPstrTEST")

    def test_parallel_for_tuple(self):
        def test_impl(t):
            len_total = 0
            with openmp("parallel for reduction(+:len_total)"):
                for i in range(len(t)):
                    len_total += len(t[i])
            return len_total
        self.check(test_impl, ("32", "4", "test", "567", "re", ""))

    def test_parallel_for_range_step_2(self):
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            with openmp("parallel for"):
                for i in range(0, 10, 2):
                    a[i] = i + 1

            return a
        self.check(test_impl, 12)
    
    def test_parallel_for_range_backward_step(self):
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            with openmp("parallel for"):
                for i in range(N-1, -1, -1):
                    a[i] = i + 1

            return a
        self.check(test_impl, 12)

    """
    def test_parallel_for_dictionary(self):
        def test_impl(N, c):
            l = {}
            with openmp("parallel for"):
                for i in range(N):
                    l[i] = i % c
            return l
        self.check(test_impl, 32, 5)
    """


@linux_only
class TestOpenmpForSchedule(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    """
    def test_static_work_calculation(self):
        def test_impl(v):
            v_len = len(v)
            step = -2
            N = 4
            omp_set_num_threads(N)
            with openmp("parallel private(thread_num)"):
                running_omp = omp_get_num_threads() != 1
                thread_num = omp_get_thread_num()
                if not running_omp:
                    iters = v_len // abs(step)
                    itersPerThread = iters // N
                    finishToThread = {}
                    for t in range(N):
                        f = itersPerThread*(t+1)-1 + min(iters%itersPerThread, t+1)
                        finishToThread[f] = t
                with openmp("for schedule(static)"):
                    for index, i in enumerate(range(v_len-1, v_len%2 - 1, -2)):
                        if not running_omp:
                            for finish in finishToThread.keys():
                                if index <= finish:
                                    thread_num = finishToThread[finish]
                        if i % (thread_num+1) == 0:
                            v[i] = i/(thread_num+1)
            return v
        self.check(test_impl, np.zeros(100))
    """

    """ Giorgis pass doesn't support static with chunksize yet?
    def test_avg_sched_const(self):
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            with openmp("parallel for num_threads(nt) schedule(static, 4)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))

    def test_avg_sched_var(self):
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            ss = 4
            with openmp("parallel for num_threads(nt) schedule(static, ss)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))
    """


@linux_only
class TestOpenmpParallelClauses(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_num_threads_clause(self):
        @njit
        def test_impl(N, c):
            omp_set_num_threads(N + c)
            d_count = 0
            with openmp("parallel num_threads(N)"):
                with openmp("critical"):
                    d_count += 1
            a_count = 0
            with openmp("parallel"):
                with openmp("critical"):
                    a_count += 1
            return d_count, a_count
        a, b = 13, 3
        r = test_impl(a, b)
        assert(r[0] == a)
        assert(r[1] == a + b)

    def test_if_clause(self):
        def test_impl(s):
            def set_arr(v):
                for i in range(len(v)):
                    v[i] = 1

            run, dont_run = 1, 0
            ar = np.zeros(s, dtype=np.int32)
            adr = np.zeros(s, dtype=np.int32)
            with openmp("parallel for if(run)"):
                set_arr(ar)
            with openmp("parallel for if(dont_run)"):
                set_arr(adr)
            return ar, adr
        size = 20
        r = test_impl(size)
        np.testing.assert_array_equal(r[0], np.ones(size))
        np.testing.assert_array_equal(r[1], np.zeros(size))

    def test_avg_arr_prev_two_elements_base(self):
        def test_impl(n, a):
            b = np.zeros(n)
            omp_set_num_threads(5)

            with openmp("parallel for"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0
            return b
        self.check(test_impl, 10, np.ones(10))

    def test_avg_num_threads_clause(self):
        def test_impl(n, a):
            b = np.zeros(n)
            with openmp("parallel for num_threads(5)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))        

    def test_avg_num_threads_clause_var(self):
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            with openmp("parallel for num_threads(nt)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))

    """
    Uses apparently unsupported chunking.
    def test_avg_if_const(self):
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            with openmp("parallel for if(1) num_threads(nt) schedule(static, 4)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))

    def test_avg_if_var(self):
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            ss = 4
            do_if = 1
            with openmp("parallel for if(do_if) num_threads(nt) schedule(static, ss)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))
    """


@linux_only
class TestOpenmpDataClauses(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_default_none(self):
        @njit
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            x = 7
            with openmp("parallel for default(none)"):
                for i in range(N):
                    y = i + x
                    a[i] = y
                    z = i

            return a, z
        
        with self.assertRaises(UnspecifiedVarInDefaultNone) as raises:
            test_impl(100)
        self.assertIn("Variables with no data env clause", str(raises.exception))

    def test_data_sharing_default(self):
        @njit
        def test_impl(N, M):
            x = np.zeros(N)
            z = 3.14
            i = 7
            with openmp("parallel"):
                yn = M + 1
                zs = z
                y = np.zeros(N)
                with openmp("for"):
                    for i in range(N):
                        y[i] = yn + 2*(i+1)
                with openmp("for"):
                    for i in range(N):
                        x[i] = y[i] - i
                        with openmp("critical"):
                            z += 3
            return x, y, zs, z, i
        N, M = 10, 5
        r = test_impl(N, M)
        print("y =", r[1])
        np.testing.assert_array_equal(r[0], np.arange(M+3, M+N+3))
        np.testing.assert_array_equal(r[1], np.arange(M+3, M+2*N+2, 2))
        assert(r[2] == 3.14)
        assert(r[3] == 3.14 + 3*N)
        assert(r[4] == 7)

    def test_variables(self):
        @njit
        def test_impl():
            x    = 5
            y    = 3
            zfp  = 2
            zsh  = 7
            nerr = 0
            nsing = 0
            NTHREADS = 4
            numthrds = 0
            omp_set_num_threads(NTHREADS)
            vals    = np.zeros(NTHREADS)
            valsfp  = np.zeros(NTHREADS)

            with openmp ("""parallel private(x) shared(zsh)
                        firstprivate(zfp) private(ID)"""):
                ID = omp_get_thread_num()
                with openmp("single"):
                        nsing = nsing+1
                        numthrds = omp_get_num_threads()
                        if (y != 3):
                            nerr = nerr+1
                            print("Shared Default status failure y = ",
                                    y, " It should equal 3")

                # verify each thread sees the same variable vsh
                with openmp("critical"):
                        zsh = zsh+ID

                # test first private
                zfp = zfp+ID
                valsfp[ID] = zfp

                # setup test to see if each thread got its own x value
                x = ID
                vals[ID] = x

    # Shared clause test: assumes zsh starts at 7 and we add up IDs from 4 threads
            if (zsh != 13):
                print("Shared clause or critical failed",zsh)
                nerr = nerr+1

    # Single Test: How many threads updated nsing?
            if (nsing!=1):
                print(" Single test failed",nsing)
                nerr = nerr+1

    # Private clause test: did each thread get its own x variable?
            for i in range(numthrds):
                if(int(vals[i]) != i):
                    print("Private clause failed",numthrds,i,vals[i])
                    nerr = nerr+1

    # First private clause test: each thread should get 2 + ID for up to 4 threads
            for i in range(numthrds):
                if(int(valsfp[i]) != 2+i):
                    print("Firstprivate clause failed",numthrds,i,valsfp[i])
                    nerr = nerr+1

    # Test number of threads
            if (numthrds > NTHREADS):
                print("Number of threads error: too many threads",
                        numthrds, NTHREADS)
                nerr = nerr+1
        
            if nerr > 0:
                print(nerr, """ errors when testing parallel, private, shared,
                            firstprivate, critical  and single""")

            return nerr
        
        assert(test_impl() == 0)

    def test_privates(self):
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            x = 7
            with openmp("""parallel for firstprivate(x) private(y)
                         lastprivate(zzzz) private(i) shared(a)
                          firstprivate(N) default(none)"""):
                for i in range(N):
                    y = i + x
                    a[i] = y
                    zzzz = i

            return a, zzzz
        pass
        #self.check(test_impl, 100)

    def test_private_retain_value(self):
        @njit
        def test_impl():
            x = 5
            with openmp("parallel private(x)"):
                x = 13
            return x
        assert(test_impl() == 5)

    def test_private_retain_value_param(self):
        @njit
        def test_impl(x):
            with openmp("parallel private(x)"):
                x = 13
            return x
        assert(test_impl(5) == 5)

    def test_private_retain_value_for(self):
        @njit
        def test_impl():        
            x = 5
            with openmp("parallel private(x)"):
                with openmp("for"):
                    for i in range(10):
                        x = i
            return x
        assert(test_impl() == 5)

    def test_private_retain_value_for_param(self):
        @njit
        def test_impl(x):        
            with openmp("parallel private(x)"):
                with openmp("for"):
                    for i in range(10):
                        x = i
            return x
        assert(test_impl(5) == 5)

    def test_private_retain_value_combined_for(self):
        @njit
        def test_impl():
            x = 5
            with openmp("parallel for private(x)"):
                for i in range(10):
                    x = i
            return x
        assert(test_impl() == 5)

    def test_private_retain_value_combined_for_param(self):
        @njit
        def test_impl(x):
            with openmp("parallel for private(x)"):
                for i in range(10):
                    x = i
            return x
        assert(test_impl(5) == 5)

    def test_private_retain_two_values(self):
        @njit
        def test_impl():
            x = 5
            y = 7
            with openmp("parallel private(x,y)"):
                x = 13
                y = 40
            return x, y
        assert(test_impl() == (5, 7))
    
    def test_private_retain_array(self):
        @njit
        def test_impl(N, x):
            a = np.ones(N)
            with openmp("parallel private(a)"):
                with openmp("single"):
                    sa = a
                a = np.zeros(N)
                with openmp("for"):
                    for i in range(N):
                        a[i] = x
            return a, sa
        r = test_impl(10, 3)
        np.testing.assert_array_equal(r[0], np.ones(r[0].shape))
        with self.assertRaises(AssertionError) as raises:
            np.testing.assert_array_equal(r[1], np.ones(r[0].shape))

    def test_private_divide_work(self):
        def test_impl(v, npoints):
            omp_set_num_threads(3)

            with openmp("""parallel default(shared)
                        private(iam,nt,ipoints,istart)"""):
                iam = omp_get_thread_num()
                nt = omp_get_num_threads()
                ipoints = npoints // nt
                istart = iam * ipoints
                if (iam == nt-1):
                    ipoints = npoints - istart
                for i in range(ipoints):
                    v[istart+i] = 123.456
            return v
        self.check(test_impl, np.zeros(12), 12)

    def test_firstprivate(self):
        @njit
        def test_impl(x, y):
            with openmp("parallel firstprivate(x)"):
                xs = x
                x = y
            return xs, x
        x, y = 5, 3
        self.assert_outputs_equal(test_impl(x, y), (x, x))

    def test_lastprivate_for(self):
        @njit
        def test_impl(N):
            a = np.zeros(N)
            with openmp("parallel for lastprivate(si)"):
                for i in range(N):
                    si = i + 1
                    a[i] = si
            return si, a
        N = 10
        r = test_impl(N)
        assert(r[0] == N)
        np.testing.assert_array_equal(r[1], np.arange(1, N+1))

    def test_lastprivate_sections(self):
        @njit
        def test_impl(N, si):
            a = np.zeros(N)
            with openmp("parallel"):
                with openmp("sections lastprivate(si)"):
                    sis1 = si
                    for i in range(N):
                        with openmp("section"):
                            si = i
                sis2 = si
                with openmp("sections lastprivate(si)"):
                    for i in range(N):
                        with openmp("section"):
                            si = N - i
                            a[i] = si
            return si, sis1, sis2, a
        N, d = 10, 5
        r = test_impl(N, d)
        assert(r[0] == 1)
        assert(r[1] != d)
        assert(r[2] == N - 1)
        np.testing.assert_array_equal(r[3], np.arange(N, 0, -1))

    def test_lastprivate_conditional(self):
        @njit
        def test_impl():
            pass

    """
    def test_threadprivate(self):
        @njit
        def test_impl(N, c):
            omp_set_num_threads(N)
            a = np.zeros(N)
            ra = np.zeros(N)
            val = 0
            with openmp("threadprivate(val)"):
                pass
            with openmp("parallel private(tn, sn)"):
                tn = omp_get_thread_num()
                sn = c + tn
                val = sn
                a[tn] = sn
            with openmp("parallel private(tn)"):
                tn = omp_get_thread_num()
                ra[tn] = 1 if val == a[tn] else 0
            return ra
        nt = 8
        np.testing.assert_array_equal(test_impl(nt, 5), np.ones(nt))
    """

    # Linear is also compatible with 'for', but not implemented yet.
    def test_linear_clause(self):
        @njit
        def test_impl(N):
            a = np.arange(N) + 1
            b = np.zeros(N//2)

            linearj = 0
            with openmp("simd linear(linearj:1)"):
                for i in range(0, N, 2):
                    b[linearj] = a[i] * 2

            return b, linearj
        N = 50
        r = test_impl(N)
        print(r[0])
        np.testing.assert_array_equal(r[0], np.arange(2, N*2-1, 4))
        assert(r[1] == N//2-1)


@linux_only
class TestOpenmpConstraints(TestOpenmpBase):
    """Tests designed to confirm that errors occur when expected, or
    to see how OpenMP behaves in various circumstances"""

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_parallel_for_no_for_loop(self):
        @njit
        def test_impl():
            with openmp("parallel for"):
                pass

        with self.assertRaises(ParallelForWrongLoopCount) as raises:
            test_impl()
        self.assertIn("OpenMP parallel for regions must contain exactly one", str(raises.exception))

    def test_parallel_for_multiple_for_loops(self):
        @njit
        def test_impl():
            a = np.zeros(4)
            with openmp("parallel for"):
                for i in range(2):
                    a[i] = 1
                for i in range(2, 4):
                    a[i] = 1

        with self.assertRaises(ParallelForWrongLoopCount) as raises:
            test_impl()
        self.assertIn("OpenMP parallel for regions must contain exactly one", str(raises.exception))

    def test_statement_before_parallel_for(self):
        @njit
        def test_impl():
            a = np.zeros(4)
            with openmp("parallel for"):
                x = "Fail"
                for i in range(4):
                    a[i] = i
            return a

        with self.assertRaises(ParallelForExtraCode) as raises:
            test_impl()
        self.assertIn("Extra code near line", str(raises.exception))

    def test_statement_after_parallel_for(self):
        @njit
        def test_impl():
            a = np.zeros(4)
            with openmp("parallel for"):
                for i in range(4):
                    a[i] = i
                x = "Fail"
            return a

        with self.assertRaises(ParallelForExtraCode) as raises:
            test_impl()
        self.assertIn("Extra code near line", str(raises.exception))

    def test_parallel_for_incremented_step(self):
        @njit
        def test_impl(v, n):
            for i in range(n):
                with openmp("parallel for"):
                    for j in range(0, len(v), i):
                        v[j] = i
            return v

        with self.assertRaises(NotImplementedError) as raises:
            test_impl(np.zeros(100), 3)
        self.assertIn("Only constant step", str(raises.exception))

    def test_nonstring_var_omp_statement(self):
        @njit
        def test_impl(v):
            ovar = 7
            with openmp(ovar):
                for i in range(len(v)):
                    v[i] = 1.0
            return v
        with self.assertRaises(NonStringOpenmpSpecification) as raises:
            test_impl(np.zeros(100))
        self.assertIn("Non-string OpenMP specification at line", str(raises.exception))

    def test_parallel_for_nonconst_var_omp_statement(self):
        @njit
        def test_impl(v):
            ovar = "parallel "
            ovar += "for"
            with openmp(ovar):
                for i in range(len(v)):
                    v[i] = 1.0
            return v

        with self.assertRaises(NonconstantOpenmpSpecification) as raises:
            test_impl(np.zeros(100))
        self.assertIn("Non-constant OpenMP specification at line", str(raises.exception))

    """
    def test_parallel_for_blocking_if(self):
        @njit
        def test_impl():
            n = 0
            with openmp("parallel"):
                half_threads = omp_get_num_threads()//2
                if omp_get_thread_num() < half_threads:
                    with openmp("for reduction(+:n)"):
                        for _ in range(half_threads):
                            n += 1
            return n

        #with self.assertRaises(AssertionError) as raises:
         #   njit(test_impl)
        test_impl()
        #print(str(raises.exception))
    """

    def test_parallel_for_delaying_condition(self):
        @njit
        def test_impl():
            n = 0
            with openmp("parallel private(lc)"):
                lc = 0
                while lc < omp_get_thread_num():
                    lc += 1
                with openmp("for reduction(+:n)"):
                    for _ in range(omp_get_num_threads()):
                        n += 1
            return n
        test_impl()


@linux_only
class TestOpenmpConcurrency(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_single(self):
        @njit
        def test_impl(nt):
            omp_set_num_threads(nt)
            a = np.zeros(4, dtype=np.int64)
            with openmp("parallel"):
                with openmp("single"):
                    a[0] += 1
            return a
        np.testing.assert_array_equal(test_impl(4), np.array([1,0,0,0]))

    def test_master(self):
        @njit
        def test_impl(nt):
            omp_set_num_threads(nt)
            a = np.ones(4, dtype=np.int64)
            with openmp("parallel"):
                with openmp("master"):
                    a[0] += omp_get_thread_num()
            return a
        np.testing.assert_array_equal(test_impl(4), np.array([0,1,1,1]))

    def test_critical_threads1(self):
        def test_impl(N, iters):
            omp_set_num_threads(N)
            count = 0
            p = 0
            sum = 0
            with openmp("parallel"):
                with openmp("barrier"):
                    pass
                with openmp("for private(p, sum)"):
                    for _ in range(iters):
                        with openmp("critical"):
                            p = count
                            sum = 0
                            for i in range(10000):
                                if i % 2 == 0:
                                    sum += 1
                                else:
                                    sum -= 1
                            p += 1 + sum
                            count = p
            return count
        iters = 1000
        self.check(test_impl, 2, iters)
        
    def test_critical_threads2(self):
        @njit
        def test_impl(N):
            omp_set_num_threads(N)
            ca = np.zeros(N)
            sum = 0
            with openmp("parallel private(sum)"):
                c = N
                with openmp("barrier"):
                    pass
                with openmp("critical"):
                    ca[omp_get_thread_num()] = c - 1
                    # Sleep
                    sum = 0
                    for i in range(10000):
                        if i % 2 == 0:
                            sum += 1
                        else:
                            sum -= 1
                    c -= 1 + sum
            print("c =", c)
            return np.sort(ca)
        nt = 16
        np.testing.assert_array_equal(test_impl(nt), np.arange(nt))

    def test_critical_result(self):
        @njit
        def test_impl(N):
            omp_set_num_threads(N)
            count = 0
            with openmp("parallel"):
                if omp_get_thread_num() < N//2:
                    with openmp("critical"):
                        count += 1
                else:
                    with openmp("critical"):
                        count += 1
            return count
        nt = 16
        assert(test_impl(nt) == nt)

    """
    def test_named_critical(self):
        @njit
        def test_impl(N):
            omp_set_num_threads(N)
            a = np.zeros((2, N))
            sa = np.zeros(N)
            with openmp("parallel private(a0c, sum, tn)"):
                tn = omp_get_thread_num()
                with openmp("barrier"):
                    pass
                with openmp("critical (a)"):
                    # Sleep
                    sum = 0
                    for j in range(1000):
                        if j % 2 == 0:
                            sum += 1
                        else:
                            sum -= 1
                    a[0][tn] = 1 + sum
                with openmp("critical (b)"):
                    a0c = np.copy(a[0])
                    # Sleep
                    sum = 0
                    for j in range(10000):
                        if j % 2 == 0:
                            sum += 1
                        else:
                            sum -= 1
                    a[1][tn] = 1 + sum
                    sa[tn] = 1 if a[0] != a0c else 0
            return a, sa
        nt = 16
        r = test_impl(nt)
        np.testing.assert_array_equal(r[0], np.ones((2, nt)))
        assert(np.any(r[1]))
    """

    """
    # Revisit
    def test_atomic_threads(self):
        def test_impl(N, iters):
            omp_set_num_threads(N)
            count = 0
            p = 0
            sum = 0
            with openmp("parallel"):
                with openmp("barrier"):
                    pass
                with openmp("for private(p, sum)"):
                    for _ in range(iters):
                        with openmp("atomic"):
                            p = count
                            sum = 0
                            for i in range(10000):
                                if i % 2 == 0:
                                    sum += 1
                                else:
                                    sum -= 1
                            p += 1 + sum
                            count = p
            return count
        iters = 1000
        self.check(test_impl, 2, iters)
    """

    def test_atomic(self):
        @njit
        def test_impl(nt, N, c):
            omp_set_num_threads(nt)
            a = np.zeros(N)
            with openmp("parallel for private(b, index)"):
                for i in range(nt):
                    b = 0
                    index = i % N
                    with openmp("atomic write"):
                        a[index] = nt % c
                    with openmp("barrier"):
                        pass
                    with openmp("atomic read"):
                        b = a[index-1] + index
                    with openmp("barrier"):
                        pass
                    with openmp("atomic update"):
                        a[index] += b
            print("a =", a)
            return a
        nt, N, c = 27, 8, 6
        rc = np.zeros(N)
        """
        ba = np.zeros(nt)
        for i in range(nt):
            index = i % N
            rc[index] = nt % c
        print("rc1:", rc)

        for i in range(nt):
            index = i % N
            ba[i] = rc[index-1] + index

        for i in range(nt):
            index = i % N
            rc[index] += ba[i]
        print("rc2:", rc)
        """

        for i in range(nt):
            index = i % N
            ts = nt//N
            ts += 1 if index < nt % N else 0
            rc[index] = nt%c + (nt%c + index) * ts

        np.testing.assert_array_equal(test_impl(nt, N, c), rc)

    def test_atomic_capture(self):
        @njit
        def test_impl(nt, N, c):
            s = math.ceil(N//2)
            a = np.zeros(s)
            sva = np.zeros(N)
            tns = np.zeros(N)
            with openmp("parallel for num_threads(nt) private(sv, index)"):
                for i in range(N):
                    index = i % s
                    tns[i] = omp_get_thread_num()
                    with openmp("atomic write"):
                        a[index] = index * c + 1
                    with openmp("barrier"):
                        pass
                    with openmp("atomic capture"):
                        sv = a[index-1]
                        a[index-1] += sv + (tns[i]%c + 1)
                    #sva[index] = sv
            return a, sva, tns
        nt, N, c = 16, 30, 7
        r1, r2, tns = test_impl(nt, N, c)
        size = math.ceil(N//2)
        rc = np.arange(1, (size-1)*c+2, c)
        #np.testing.assert_array_equal(r2, np.roll(rc, 1))
        for i in range(N):
            index = i % size
            rc[index-1] += rc[index-1] + (tns[i]%c + 1)
        np.testing.assert_array_equal(r1, rc)

    def test_parallel_sections(self):
        @njit
        def test_impl(nt):
            ta = np.empty(nt)
            count = 0
            with openmp("parallel sections num_threads(nt) private(sum)"):
                for i in range(nt):
                    with openmp("section"):
                        ta[i] = omp_get_thread_num()
                        # Sleep
                        sum = 0
                        for i in range(10000):
                            if i % 2 == 0:
                                sum += 1
                            else:
                                sum -= 1
                        count += 1 + sum
            print(ta)
            return np.sort(ta), count
        nt = 5
        r = test_impl(nt)
        np.testing.assert_array_equal(r[0], np.arange(nt))
        assert(r[1] == nt)

    def test_nowait(self):
        def test_impl(n, m, a, b, y, z):
            omp_set_num_threads(5)

            with openmp("parallel"):
                with openmp("for nowait"):
                    for i in range(1, n):
                        b[i] = (a[i] + a[i-1]) / 2.0
                with openmp("for nowait"):
                    for i in range(m):
                        y[i] = math.sqrt(z[i])

            return b, y
        self.check(test_impl, 10, 20, np.ones(10), np.zeros(10),
                    np.zeros(20), np.full(20, 13))

    def test_nested_parallel_for(self):
        @njit
        def test_impl(N):
            omp_set_num_threads(N)
            omp_set_nested(1)
            omp_set_dynamic(0)
            a = np.zeros(N, dtype=np.int32)
            with openmp("parallel for"):
                for i in range(N):
                    with openmp("parallel for"):
                        for j in range(N):
                            a[omp_get_thread_num()] += 1
            return a

        nt = 4
        np.testing.assert_array_equal(test_impl(nt), np.full(nt, nt))

    def test_nested_parallel_regions(self):
        @njit
        def test_impl():
            omp_set_nested(1)
            omp_set_max_active_levels(8)
            omp_set_dynamic(0)
            omp_set_num_threads(2)
            a = np.zeros((10,3), dtype=np.int32)
            b = np.zeros((10,3), dtype=np.int32)
            with openmp("parallel private(tn)"):
                omp_set_num_threads(3)
                with openmp("parallel private(tn)"):
                    omp_set_num_threads(4)
                    #with openmp("single"):  # why single?
                    tn = omp_get_thread_num()
                    if tn < 10:
                        a[tn][0] = omp_get_max_active_levels()
                        a[tn][1] = omp_get_num_threads()
                        a[tn][2] = omp_get_max_threads()
                with openmp("barrier"):
                    pass
                #with openmp("single"): # why single?
                tn = omp_get_thread_num()
                # This test will fail if this useless if is left out.
                if tn < 10:
                    b[tn][0] = omp_get_max_active_levels()
                    b[tn][1] = omp_get_num_threads()
                    b[tn][2] = omp_get_max_threads()
            return a, b
        pass
        #print("nested test:", test_impl())


@linux_only
class TestOpenmpTask(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_task_basic(self):
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task"):
                            a[i] = 1
            return a
        self.check(test_impl, 15)

    def test_task_thread_assignment(self):
        @njit
        def test_impl(ntsks):
            a = np.empty(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task"):
                            a[i] = omp_get_thread_num()
            return a
        
        with self.assertRaises(AssertionError) as raises:
            v = test_impl(15)
            np.testing.assert_equal(v[0], v)

    def test_task_data_sharing_default(self):
        @njit
        def test_impl(n1, n2):
            x = n1
            with openmp("parallel private(y)"):
                y = n1
                with openmp("single"):
                    with openmp("task"):
                        print("x =", x)
                        print("y =", y)
                        xa = x == n1
                        ya = y == n1
                        x, y = n2, n2
                    with openmp("taskwait"):
                        ysave = y
            return (x, ysave), (xa, ya)
        n1, n2 = 1, 2
        r = test_impl(n1, n2)
        print(r[1])
        self.assert_outputs_equal(r[1], (True, True))
        print(r[0])
        self.assert_outputs_equal(r[0], (n2, n1))

    # Segmentation fault
    def test_task_single_implicit_barrier(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(sum)"):
                            # Sleep
                            sum = 0
                            for j in range(10000):
                                if j % 2 == 0:
                                    sum += 1
                                else:
                                    sum -= 1
                            a[i] = 1 + sum
                #with openmp("barrier"):
                #    pass
                sa = np.copy(a)
            return sa
        ntsks = 15
        r = test_impl(ntsks)
        print(r)
        np.testing.assert_array_equal(r, np.ones(ntsks))

    # Segmentation fault
    def test_task_single_nowait(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single nowait"):
                    for i in range(ntsks):
                        with openmp("task private(sum)"):
                            sum = 0
                            for j in range(10000):
                                if j % 2 == 0:
                                    sum += 1
                                else:
                                    sum -= 1
                            a[i] = 1 + sum
                sa = np.copy(a)
            return sa
        with self.assertRaises(AssertionError) as raises:
            ntsks = 15
            r = test_impl(ntsks)
            print(r)
            np.testing.assert_array_equal(r, np.ones(ntsks))

    # Error with commented out code, other version never finished running
    def test_task_barrier(self):
        @njit
        def test_impl(nt):
            omp_set_num_threads(nt)
            a = np.zeros((nt+1)*nt/2)
            #a = np.zeros(10)
            with openmp("parallel"):
                with openmp("single"):
                    for tn in range(nt):
                        with openmp("task"):
                            for i in range(tn+1):
                                with openmp("task"):
                                    a[i] = omp_get_thread_num() + 1
                    with openmp("barrier"):
                        print(a)
                        ret = np.all(a)
            return ret
        assert(test_impl(4))

    def test_taskwait(self):
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(sum)"):
                            sum = 0
                            for j in range(10000):
                                if j % 2 == 0:
                                    sum += 1
                                else:
                                    sum -= 1
                            a[i] = 1 + sum
                    with openmp("taskwait"):
                        ret = np.all(a)
            return ret
        self.check(test_impl, 15)

    # Segmentation fault
    def test_taskwait_descendants(self):
        @njit
        def test_impl(ntsks, dtsks):
            a = np.zeros(ntsks)
            da = np.zeros((ntsks, dtsks))
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task"):
                            a[i] = 1
                            for j in range(dtsks):
                                with openmp("task private(sum)"):
                                    sum = 0
                                    for k in range(10000):
                                        if k % 2 == 0:
                                            sum += 1
                                        else:
                                            sum -= 1
                                    da[i][j] = 1 + sum
                    with openmp("taskwait"):
                        ac = np.copy(a)
                        dac = np.copy(da)
                with openmp("barrier"):
                    pass
            return ac, dac

        r = test_impl(15, 10)
        np.testing.assert_array_equal(r[0], np.ones(r[0].shape))
        with self.assertRaises(AssertionError) as raises:
            print(r[1])
            np.testing.assert_array_equal(r[1], np.ones(r[1].shape))

    # Tree is not iterable
    def test_undeferred_task(self):
        @njit
        def test_impl():
            with openmp("parallel"):
                flag = 1
                with openmp("single"):
                    with openmp("task if(1) private(sum)"):
                        sum = 0
                        for i in range(10000):
                            if i % 2 == 0:
                                sum += 1
                            else:
                                sum -= 1
                        r = flag + sum
                        print("flag =", flag)
                        print("r =", r)
                    flag = 0
                    print("Reached flag 0")
            return r
        assert(test_impl())

    # Tree is not iterable
    def test_untied_task_thread_assignment(self):
        @njit
        def test_impl(ntsks):
            start_nums = np.zeros(ntsks)
            current_nums = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task untied private(sum)"):
                            start_nums[i] = omp_get_thread_num()
                            with openmp("task if(0) shared(sum)"):
                                # Sleep
                                sum = 0
                                for j in range(10000):
                                    if j % 2 == 0:
                                        sum += 1
                                    else:
                                        sum -= 1
                            current_nums[i] = omp_get_thread_num() + sum
                with openmp("barrier"):
                    pass
            return start_nums, current_nums

        with self.assertRaises(AssertionError) as raises:
            sids, cids = test_impl(15)
            print(sids, cids)
            np.testing.assert_array_equal(sids, cids)

    # Failed
    def test_taskyield_thread_assignment(self):
        @njit
        def test_impl(ntsks):
            start_nums = np.zeros(ntsks)
            finish_nums = np.zeros(ntsks)
            yielded_tasks = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(stn, start_i, finish_i, diff)"):
                            stn = omp_get_thread_num()
                            start_i = np.where(start_nums == stn)[0]
                            finish_i = np.where(finish_nums == stn)[0]
                            diff = np.zeros(len(start_i), dtype=np.int64)
                            for sindex in range(len(start_i)):
                                for findex in range(len(finish_i)):
                                    if start_i[sindex] == finish_i[findex]:
                                        break
                                else:
                                    diff[sindex] = start_i[sindex]
                            for dindex in diff[diff != 0]:
                                yielded_tasks[dindex] = 1
                            start_nums[i] = stn
                            with openmp("taskyield"):
                                pass
                            finish_nums[i] = omp_get_thread_num()
                with openmp("barrier"):
                    pass
            return yielded_tasks

        yt = test_impl(50)
        print(yt)
        assert(np.any(yt))

    # Parser error
    def test_final_task_thread_assignment(self):
        @njit
        def test_impl(ntsks, c):
            final_nums = np.zeros(ntsks)
            included_nums = np.zeros(ntsks)
            da = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task final(i>c) private(sum, d)"):
                            ftask_num = i
                            final_nums[ftask_num] = omp_get_thread_num()
                            # If it is a final task, generate an included task
                            if ftask_num > c:
                                d = 1
                                with openmp("task private(sum)"):
                                    itask_num = ftask_num
                                    # Sleep
                                    sum = 0
                                    for j in range(10000):
                                        if j % 2 == 0:
                                            sum += 1
                                        else:
                                            sum -= 1
                                    included_nums[itask_num] = omp_get_thread_num()
                                    da[itask_num] = d + sum
                                d = 0

            return final_nums, included_nums, da

        ntsks, c = 15, 5
        fns, ins, da = test_impl(ntsks, c)
        print(fns, ins)
        np.testing.assert_array_equal(fns[c:], ins[c:])
        np.testing.assert_array_equal(da, np.ones(ntsks))

    # Unimplemented
    def test_taskgroup(self):
        @njit
        def test_impl(ntsks, dtsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    with openmp("taskgroup"):
                        for i in range(ntsks):
                            with openmp("task"):
                                for _ in range(dtsks):
                                    with openmp("task"):
                                        # Sleep
                                        sum = 0
                                        for j in range(10000):
                                            if j % 2 == 0:
                                                sum += 1
                                            else:
                                                sum -= 1
                                        a[i] = 1 + sum
                    sa = np.copy(a)
            return a, sa
        ntsks = 15
        r = test_impl(ntsks, 10)
        np.testing.assert_array_equal(r[0], np.ones(ntsks))
        np.testing.assert_array_equal(r[1], np.ones(ntsks))

    # Unimplemented
    def test_task_priority(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            count = 0
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task priority(i)"):
                            count += i + 1
                            a[i] = count
            return a
        
        ntsks = 15
        r = test_impl(ntsks)
        print(r)
        rc = np.zeros(ntsks)
        for i in range(ntsks):
            rc[i] = sum(range(i+1, ntsks+1))
        np.testing.assert_array_equal(r, rc)

    def test_task_mergeable(self):
        @njit
        def test_impl(ntsks, c1, c2):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(x)"):
                            x = c1
                            with openmp("task mergeable if(0)"):
                                x = c2
                            a[i] = x
            return a
        ntsks, c1, c2 = 75, 2, 3
        assert(c2 in test_impl(ntsks, c1, c2))

    # Tree is not iterable
    def test_task_depend(self):
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            da = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(x, done)"):
                            x = 1
                            done = False
                            with openmp("task shared(x) depend(out: x)"):
                                x = 5
                            with openmp("""task shared(done, x)
                                        depend(out: done) depend(inout: x)"""):
                                x += i
                                done = True
                            with openmp("""task shared(done, x)
                                         depend(in: done) depend(inout: x)"""):
                                x *= i
                                da[i] = 1 if done else 0
                            with openmp("task shared(x) depend(in: x)"):
                                a[i] = x
            print(a, da)
            return a, da
        self.check(test_impl, 15)

    # Affinity clause should not affect result
    def test_task_affinity(self):
        def test_impl(ntsks, const):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task firstprivate(i)"):
                            with openmp("""task shared(b) depend(out: b)
                                         affinity(a)"""):
                                b = np.full(i, const)
                            with openmp("""task shared(b) depend(in: b)
                                         affinity(a)"""):
                                a[i] = np.sum(b)
            return a
        self.check(test_impl, 15, 4)


@linux_only
class TestOpenmpTaskloop(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_taskloop_basic(self):
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    with openmp("taskloop"):
                        for i in range(ntsks):
                            a[i] = 1
            return a
        self.check(test_impl, 15)

    def test_taskloop_num_tasks(self):
        @njit
        def test_impl(nt, iters, ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel num_threads(nt)"):
                with openmp("single"):
                    with openmp("taskloop num_tasks(ntsks)"):
                        for i in range(iters):
                            a[i] = omp_get_thread_num()     
            return a
        nt, iters, ntsks = 8, 10, 4
        assert(len(np.unique(test_impl(nt, iters, ntsks))) <= ntsks)

    def test_taskloop_grainsize(self):
        @njit
        def test_impl(nt, iters, ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel num_threads(nt)"):
                with openmp("single"):
                    iters_per_task = iters//ntsks
                    with openmp("taskloop grainsize(iters_per_task)"):
                        for i in range(iters):
                            a[i] = omp_get_thread_num()     
            return a
        nt, iters, ntsks = 8, 10, 4
        assert(len(np.unique(test_impl(nt, iters, ntsks))) <= ntsks)

    def test_taskloop_nogroup(self):
        @njit
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            sa = np.zeros(ntsks)
            with openmp("parallel"):
                with openmp("single"):
                    s = 0
                    with openmp("taskloop nogroup num_tasks(ntsks)"):
                        for i in range(ntsks):
                            a[i] = 1
                            sa[i] = s
                    with openmp("task priority(1)"):
                        s = 1
            return a, sa
        ntsks = 15
        r = test_impl(ntsks)
        np.testing.assert_array_equal(r[0], np.ones(ntsks))
        np.testing.assert_array_equal(r[1], np.ones(ntsks))
        
    def test_taskloop_collapse(self):
        @njit
        def test_impl(ntsks, nt):
            fl = np.zeros(ntsks)
            sl = np.zeros(ntsks)
            tl = np.zeros(ntsks)
            omp_set_num_threads(nt)
            with openmp("parallel"):
                with openmp("single"):
                    with openmp("taskloop collapse(2) num_tasks(ntsks)"):
                        for i in range(ntsks):
                            fl[i] = omp_get_thread_num()
                            for j in range(1):
                                sl[i] = omp_get_thread_num()
                                for k in range(1):
                                    tl[i] = omp_get_thread_num()

            return fl, sl, tl
        r = test_impl(25, 4)
        with self.assertRaises(AssertionError) as raises:
            np.testing.assert_array_equal(r[0], r[1])
        np.testing.assert_array_equal(r[1], r[2])


@linux_only
class TestOpenmpPi(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_pi_loop(self):
        def test_impl(num_steps):
            step = 1.0 / num_steps

            the_sum = 0.0
            omp_set_num_threads(4)

            with openmp("parallel"):
                with openmp("for reduction(+:the_sum) schedule(static)"):
                    for j in range(num_steps):
                        x = ((j-1) - 0.5) * step
                        the_sum += 4.0 / (1.0 + x * x)

            pi = step * the_sum
            print(pi)
            return pi

        self.check(test_impl, 100000)

    def test_pi_loop_combined(self):
        def test_impl(num_steps):        
            step = 1.0 / num_steps

            the_sum = 0.0
            omp_set_num_threads(4)

            with openmp("parallel for reduction(+:the_sum) schedule(static)"):
                for j in range(num_steps):
                    x = ((j-1) - 0.5) * step
                    the_sum += 4.0 / (1.0 + x * x)

            pi = step * the_sum
            return pi

        self.check(test_impl, 100000)

    def test_pi_spmd(self):
        def test_impl(num_steps):
            step = 1.0 / num_steps
            MAX_THREADS=8
            tsum = np.zeros(MAX_THREADS)

            j = 4
            omp_set_num_threads(j)
            full_sum = 0.0

            with openmp("parallel private(tid, numthreads, local_sum, x)"):
                tid = omp_get_thread_num()
                numthreads = omp_get_num_threads()
                local_sum = 0.0

                for i in range(tid, num_steps, numthreads):
                    x = (i + 0.5) * step
                    local_sum += 4.0 / (1.0 + x * x)

                tsum[tid] = local_sum

            for k in range(j):
                full_sum += tsum[k]

            pi = step * full_sum
            return pi

        self.check(test_impl, 10000000)

    def test_pi_task(self):
        def test_pi_comp(Nstart, Nfinish, step):
            MIN_BLK = 256
            pi_sum = 0.0
            if Nfinish - Nstart < MIN_BLK:
                for i in range(Nstart, Nfinish):
                    x = (i + 0.5) * step
                    pi_sum += 4.0 / (1.0 + x * x)
            else:
                iblk = Nfinish - Nstart
                pi_sum1 = 0.0
                pi_sum2 = 0.0
                cut = Nfinish-(iblk // 2)
                with openmp("task shared(pi_sum1)"):
                    pi_sum1 = test_pi_comp(Nstart, cut, step)
                with openmp("task shared(pi_sum2)"):
                    pi_sum2 = test_pi_comp(cut, Nfinish, step)
                with openmp("taskwait"):
                    pi_sum = pi_sum1 + pi_sum2
            return pi_sum

        @njit
        def test_pi_comp_njit(Nstart, Nfinish, step):
            MIN_BLK = 256
            pi_sum = 0.0
            if Nfinish - Nstart < MIN_BLK:
                for i in range(Nstart, Nfinish):
                    x = (i + 0.5) * step
                    pi_sum += 4.0 / (1.0 + x * x)
            else:
                iblk = Nfinish - Nstart
                pi_sum1 = 0.0
                pi_sum2 = 0.0
                cut = Nfinish-(iblk // 2)
                with openmp("task shared(pi_sum1)"):
                    pi_sum1 = test_pi_comp_njit(Nstart, cut, step)
                with openmp("task shared(pi_sum2)"):
                    pi_sum2 = test_pi_comp_njit(cut, Nfinish, step)
                with openmp("taskwait"):
                    pi_sum = pi_sum1 + pi_sum2
            return pi_sum

        def test_impl(lb, num_steps, pi_comp_func):
            step = 1.0 / num_steps

            j = 4
            omp_set_num_threads(j)
            full_sum = 0.0

            with openmp("parallel"):
                with openmp("single"):
                    full_sum = pi_comp_func(lb, num_steps, step)

            pi = step * full_sum
            return pi

        py_output = test_impl(0, 1024, test_pi_comp)
        njit_output = njit(test_impl)(0, 1024, test_pi_comp_njit)
        self.assert_outputs_equal(py_output, njit_output)


if __name__ == "__main__":
    unittest.main()
