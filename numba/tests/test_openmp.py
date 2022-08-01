import math
import re
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
                    omp_get_num_procs, UnspecifiedVarInDefaultNone)
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

    def test_TestOpenmpParallelFor(self):
        self.runner()

    def test_TestOpenmpConstraints(self):
        self.runner()

    def test_TestOpenmpDataClauses(self):
        self.runner()

    def test_TestOpenmpConcurrency(self):
        self.runner()

    def test_TestOpenmpThreadsScheduleClauses(self):
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
            if (not isinstance(op1, numbers.Number) or
                not isinstance(op2, numbers.Number)):
                self.assertEqual(type(op1), type(op2))

            if isinstance(op1, np.ndarray):
                np.testing.assert_almost_equal(op1, op2)
            elif isinstance(op1, (tuple, list)):
                assert(len(op1) == len(op2))
                for i in range(len(op1)):
                    self.assert_outputs_equal(op1[i], op2[i])
            elif isinstance(op1, (bool, str, type(None))):
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
    """Smoke tests for the OpenMP transforms. These tests check the most basic
    functionality"""

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)


@linux_only
class TestOpenmpParallelFor(TestOpenmpBase):

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

    def test_parallel_for_const_var_openmp_statement(self):
        def test_impl(v):
            ovar = "parallel for"
            with openmp(ovar):
                for i in range(len(v)):
                    v[i] = 1.0
            return v
        self.check(test_impl, np.zeros(100))

    def test_parallel_for_nonconst_var_openmp_statement(self):
        def test_impl(v):
            ovar = "parallel "
            ovar += "for"
            with openmp(ovar):
                for i in range(len(v)):
                    v[i] = 1.0
            return v
        self.check(test_impl, np.zeros(100))

    # Failed
    def test_parallel_for_string_conditional(self):
        def test_impl(S):
            capitalLetters = 0
            with openmp("parallel for reduction(+:capitalLetters)"):
                for i in range(len(S)):
                    if S[i].isupper():
                        capitalLetters += 1
            return capitalLetters
        self.check(test_impl, "OpenMPstrTEST")

    # Failed with heterogeneous tuples or otherwise with abort from
    # Giorgis pass.
    """
    def test_parallel_for_tuple(self):
        def test_impl(t):
            len_total = 0
            with openmp("parallel for reduction(+:len_total)"):
                for i in range(len(t)):
                    len_total += len(t[i])
            return len_total
        self.check(test_impl, ("32", "4", "test", "567", "re", ""))
    """

    def test_range_step_2(self):
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            with openmp("parallel for"):
                for i in range(0, 10, 2):
                    a[i] = i + 1

            return a
        self.check(test_impl, 12)
    
    def test_range_backward_step(self):
        def test_impl(N):
            a = np.zeros(N, dtype=np.int32)
            with openmp("parallel for"):
                for i in range(N-1, -1, -1):
                    a[i] = i + 1

            return a
        self.check(test_impl, 12)


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

        # Fails if no AssertionError
        with self.assertRaises(AssertionError) as raises:
            test_impl()

    # Expected error
    def test_statement_before_parallel_for(self):
        @njit
        def test_impl():
            with openmp("parallel for"):
                x = "Fail"
                for _ in range(4):
                    pass
            return x

        with self.assertRaises(Exception) as raises:
            test_impl()

    # Expected error
    def test_statement_after_parallel_for(self):
        @njit
        def test_impl():
            with openmp("parallel for"):
                for _ in range(4):
                    pass
                x = "Fail"
            return x

        with self.assertRaises(Exception) as raises:
            test_impl()

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

    # Not sure if this should create a failure in the way that it does
    def test_parallel_for_delaying_if(self):
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

        #with self.assertRaises(Exception) as raises:
        #   test_impl()
        test_impl()

    """
    # Does not work if threads aren't assigned indices based on thread number
    def test_work_calculation_comparison(self):
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


@linux_only
class TestOpenmpDataClauses(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_default(self):
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

    def test_variables1(self):
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
        self.check(test_impl, 100)

    def test_private_retain_value_1(self):
        @njit
        def test_impl():
            x = 5
            with openmp("parallel private(x)"):
                x = 13
            return x
        assert(test_impl() == 5)

    def test_private_retain_value_1_2(self):
        @njit
        def test_impl():
            x = 5
            y = 7
            with openmp("parallel private(x,y)"):
                x = 13
                y = 40
            return x, y
        assert(test_impl() == (5, 7))

    def test_private_retain_value_1_param(self):
        @njit
        def test_impl(x):
            with openmp("parallel private(x)"):
                x = 13
            return x
        assert(test_impl(5) == 5)

    def test_private_retain_value_2(self):
        @njit
        def test_impl():        
            x = 5
            with openmp("parallel private(x)"):
                with openmp("for"):
                    for i in range(10):
                        x = i
            return x
        assert(test_impl() == 5)

    def test_private_retain_value_2_param(self):
        @njit
        def test_impl(x):        
            with openmp("parallel private(x)"):
                with openmp("for"):
                    for i in range(10):
                        x = i
            return x
        assert(test_impl(5) == 5)

    def test_private_retain_value_3(self):
        @njit
        def test_impl():
            x = 5
            with openmp("parallel for private(x)"):
                for i in range(10):
                    x = i
            return x
        assert(test_impl() == 5)

    def test_private_retain_value_3_param(self):
        @njit
        def test_impl(x):
            with openmp("parallel for private(x)"):
                for i in range(10):
                    x = i
            return x
        assert(test_impl(5) == 5)

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

    """
    # Currently unimplemented?
    def test_linear_clause(self):
        def test_impl(N):
            a = np.arange(N, dtype=np.float64) + 1
            b = np.zeros(N//2)

            linearj = 0
            with openmp("parallel"):
                with openmp("for linear(linearj:1)"):
                    for i in range(0, N, 2):
                        b[linearj] = a[i] * 2.0
                        linearj = linearj + 1

            return b
        self.check(test_impl, 100)
    """


@linux_only
class TestOpenmpConcurrency(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_single1(self):
        @njit
        def test_impl():
            omp_set_num_threads(4)
            a = np.zeros(4, dtype=np.int64)
            with openmp("parallel"):
                with openmp("single"):
                    a[0] = 1
            return a
        np.testing.assert_array_equal(test_impl(), np.array([1,0,0,0]))

    def test_critical(self):
        @njit
        def test_impl(N):        
            omp_set_num_threads(N)
            count = 0
            with openmp("parallel"):
                with openmp("critical"):
                    count += 1
            return count
        nt = 16
        assert(test_impl(nt) == nt)

    def test_parallel_sections(self):
        def test_impl():
            count = 0
            with openmp("parallel sections num_threads(3)"):
                with openmp("section"):
                    count += 1
                with openmp("section"):
                    count += 1
                with openmp("section"):
                    count += 2
            return count
        self.check(test_impl)

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
        np.testing.assert_array_equal(test_impl(nt), np.array([nt,nt,nt,nt]))

    def test_nested_parallel_regions():
        @njit
        def test_impl():
            omp_set_nested(1)
            omp_set_max_active_levels(8)
            omp_set_dynamic(0)
            omp_set_num_threads(2)
            a = np.zeros((10,3), dtype=np.int32)
            b = np.zeros((10,3), dtype=np.int32)
            with openmp("parallel"):
                omp_set_num_threads(3)
                with openmp("parallel"):
                    omp_set_num_threads(4)
                    with openmp("single"):
                        tn = omp_get_thread_num()
                        a[tn][0] = omp_get_max_active_levels()
                        a[tn][1] = omp_get_num_threads()
                        a[tn][2] = omp_get_max_threads()
                with openmp("barrier"):
                    pass
                with openmp("single"):
                    tn = omp_get_thread_num()
                    b[tn][0] = omp_get_max_active_levels()
                    b[tn][1] = omp_get_num_threads()
                    b[tn][2] = omp_get_max_threads()
            return a, b


@linux_only
class TestOpenmpThreadsScheduleClauses(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

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
            with openmp("parallel for num_threads(2) schedule(static)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))        

    def test_avg_num_threads_clause_var(self):
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            with openmp("parallel for num_threads(nt) schedule(static)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))

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
class TestOpenmpPi(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_pi_loop(self):
        def test_impl():
            num_steps = 100000
            step = 1.0 / num_steps

            the_sum = 0.0
            omp_set_num_threads(4)

            with openmp("parallel"):
                with openmp("for reduction(+:the_sum) schedule(static)"):
                    for j in range(num_steps):
                        c = step
                        x = ((j-1) - 0.5) * step
                        the_sum += 4.0 / (1.0 + x * x)

            pi = step * the_sum
            return pi

        self.check(test_impl)

    def test_pi_loop_combined(self):
        def test_impl():        
            num_steps = 100000
            step = 1.0 / num_steps

            the_sum = 0.0
            omp_set_num_threads(4)

            with openmp("parallel for reduction(+:the_sum) schedule(static)"):
                for j in range(num_steps):
                    c = step
                    x = ((j-1) - 0.5) * step
                    the_sum += 4.0 / (1.0 + x * x)

            pi = step * the_sum
            return pi

        self.check(test_impl)

    def test_pi_spmd(self):
        def test_impl():
            num_steps = 10000000
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

        self.check(test_impl)

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
