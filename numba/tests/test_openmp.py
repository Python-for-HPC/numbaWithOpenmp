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
                    NonStringOpenmpSpecification, omp_get_thread_limit,
                    ParallelForExtraCode, ParallelForWrongLoopCount,
                    omp_in_parallel, omp_get_level, omp_get_active_level,
                    omp_get_team_size, omp_get_ancestor_thread_num,
                    omp_get_team_num, omp_get_num_teams, omp_in_final)
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

    """This is the test runner for all the OpenMP tests, it runs them in
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

    def test_TestOpenmpRoutinesEnvVariables(self):
        self.runner()

    def test_TestOpenmpParallelForResults(self):
        self.runner()

    def test_TestOpenmpWorksharingSchedule(self):
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

    def test_TestOpenmpTarget(self):
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

    To set a default value or state for all the tests in a class, set
    a variable *var* inside the class where *var* is:

    - MAX_THREADS - Thread team size for parallel regions.
    - MAX_ACTIVE_LEVELS - Number of nested parallel regions capable of
                          running in parallel.
    """

    _numba_parallel_test_ = False

    skip_disabled = int(os.environ.get("OVERRIDE_TEST_SKIP", 0)) != 0
    run_target = int(os.environ.get("RUN_TARGET", 0)) != 0
    test_device_1 = int(os.environ.get("TEST_DEVICE_1", 0)) != 0

    env_vars = {"OMP_NUM_THREADS": omp_get_num_procs(),
                "OMP_MAX_ACTIVE_LEVELS": 1,
                "OMP_DYNAMIC": True}

    def __init__(self, *args):
        # flags for njit()
        self.cflags = Flags()
        self.cflags.nrt = True

        super(TestOpenmpBase, self).__init__(*args)

    def setUp(self):
        omp_set_num_threads(getattr(self, "MAX_THREADS",
                        TestOpenmpBase.env_vars.get("OMP_NUM_THREADS")))
        omp_set_max_active_levels(getattr(self, "MAX_ACTIVE_LEVELS",
                        TestOpenmpBase.env_vars.get("OMP_MAX_ACTIVE_LEVELS")))
        self.beforeThreads = omp_get_max_threads()
        self.beforeLevels = omp_get_max_active_levels()

    def tearDown(self):
        omp_set_num_threads(self.beforeThreads)
        omp_set_max_active_levels(self.beforeLevels)

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


#@linux_only
#class TestOpenmpBasic(TestOpenmpBase):
#    """OpenMP smoke tests. These tests check the most basic
#    functionality"""
#
#    def __init__(self, *args):
#        TestOpenmpBase.__init__(self, *args)


@linux_only
class TestOpenmpRoutinesEnvVariables(TestOpenmpBase):
    MAX_THREADS = 5

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
            omp_set_dynamic(0)
            o_nt = omp_get_max_threads()
            count = 0
            with openmp("parallel"):
                i_nt = omp_get_max_threads()
                with openmp("critical"):
                    count += 1
            return count, i_nt, o_nt
        nt = self.MAX_THREADS
        with override_env_config("OMP_NUM_THREADS", str(nt)):
            r = test_impl()
        assert(r[0] == r[1] == r[2] == nt)

    def test_func_get_num_threads(self):
        @njit
        def test_impl():
            omp_set_dynamic(0)
            o_nt = omp_get_num_threads()
            count = 0
            with openmp("parallel"):
                i_nt = omp_get_num_threads()
                with openmp("critical"):
                    count += 1
            return (count, i_nt), o_nt
        nt = self.MAX_THREADS
        with override_env_config("OMP_NUM_THREADS", str(nt)):
            r = test_impl()
        assert(r[0][0] == r[0][1] == nt)
        assert(r[1] == 1)

    def test_func_set_num_threads(self):
        @njit
        def test_impl(n1, n2):
            omp_set_dynamic(0)
            omp_set_num_threads(n1)
            count1 = 0
            count2 = 0
            with openmp("parallel"):
                with openmp("critical"):
                    count1 += 1
                omp_set_num_threads(n2)
            with openmp("parallel"):
                with openmp("critical"):
                    count2 += 1
            return count1, count2
        nt = 32
        with override_env_config("OMP_NUM_THREADS", str(4)):
            r = test_impl(nt, 20)
        assert(r[0] == r[1] == nt)

    def test_func_set_max_active_levels(self):
        @njit
        def test_impl(n1, n2, n3):
            omp_set_dynamic(0)
            omp_set_max_active_levels(2)
            omp_set_num_threads(n2)
            count1, count2, count3 = 0, 0, 0
            with openmp("parallel num_threads(n1)"):
                with openmp("single"):
                    with openmp("parallel"):
                        with openmp("single"):
                            omp_set_num_threads(n3)
                            with openmp("parallel"):
                                with openmp("critical"):
                                    count3 += 1
                        with openmp("critical"):
                            count2 += 1
                with openmp("critical"):
                    count1 += 1
            return count1, count2, count3
        n1, n2 = 3, 4
        r = test_impl(n1, n2, 5)
        assert(r[0] == n1)
        assert(r[1] == n2)
        assert(r[2] == 1)

    def test_func_get_ancestor_thread_num(self):
        @njit
        def test_impl():
            oa = omp_get_ancestor_thread_num(0)
            with openmp("parallel"):
                with openmp("single"):
                    m1 = omp_get_ancestor_thread_num(0)
                    f1 = omp_get_ancestor_thread_num(1)
                    s1 = omp_get_ancestor_thread_num(2)
                    tn1 = omp_get_thread_num()
                    with openmp("parallel"):
                        m2 = omp_get_ancestor_thread_num(0)
                        f2 = omp_get_ancestor_thread_num(1)
                        s2 = omp_get_ancestor_thread_num(2)
                        tn2 = omp_get_thread_num()
            return oa, (m1, f1, s1, tn1), (m2, f2, s2, tn2)
        oa, r1, r2 = test_impl()
        assert(oa == r1[0] == r2[0] == 0)
        assert(r1[1] == r1[3] == r2[1])
        assert(r1[2] == -1)
        assert(r2[2] == r2[3])

    def test_func_get_team_size(self):
        @njit
        def test_impl(n1, n2):
            omp_set_max_active_levels(2)
            oa = omp_get_team_size(0)
            with openmp("parallel num_threads(n1)"):
                with openmp("single"):
                    m1 = omp_get_team_size(0)
                    f1 = omp_get_team_size(1)
                    s1 = omp_get_team_size(2)
                    nt1 = omp_get_num_threads()
                    with openmp("parallel num_threads(n2)"):
                        with openmp("single"):
                            m2 = omp_get_team_size(0)
                            f2 = omp_get_team_size(1)
                            s2 = omp_get_team_size(2)
                            nt2 = omp_get_num_threads()
            return oa, (m1, f1, s1, nt1), (m2, f2, s2, nt2)
        n1, n2 = 6, 8
        oa, r1, r2 = test_impl(n1, n2)
        assert(oa == r1[0] == r2[0] == 1)
        assert(r1[1] == r1[3] == r2[1] == n1)
        assert(r1[2] == -1)
        assert(r2[2] == r2[3] == n2)

    def test_func_get_level(self):
        @njit
        def test_impl():
            oa = omp_get_level()
            with openmp("parallel if(0)"):
                f = omp_get_level()
                with openmp("parallel num_threads(1)"):
                    s = omp_get_level()
                    with openmp("parallel"):
                        t = omp_get_level()
            return oa, f, s, t
        for i, l in enumerate(test_impl()):
            assert(i == l)

    def test_func_get_active_level(self):
        @njit
        def test_impl():
            oa = omp_get_active_level()
            with openmp("parallel if(0)"):
                f = omp_get_active_level()
                with openmp("parallel num_threads(1)"):
                    s = omp_get_active_level()
                    with openmp("parallel"):
                        t = omp_get_active_level()
            return oa, f, s, t
        r = test_impl()
        for i in range(3):
            assert(r[i] == 0)
        assert(r[3] == 1)

    def test_func_in_parallel(self):
        @njit
        def test_impl():
            omp_set_dynamic(0)
            omp_set_max_active_levels(1) # 1 because first region is inactive
            oa = omp_in_parallel()
            with openmp("parallel num_threads(1)"):
                ia = omp_in_parallel()
                with openmp("parallel"):
                    n1a = omp_in_parallel()
                    with openmp("single"):
                        with openmp("parallel"):
                            n2a = omp_in_parallel()
            with openmp("parallel if(0)"):
                ua = omp_in_parallel()
            return oa, ia, n1a, n2a, ua
        r = test_impl()
        assert(r[0] == False)
        assert(r[1] == False)
        assert(r[2] == True)
        assert(r[3] == True)
        assert(r[4] == False)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_func_in_final(self):
        @njit
        def test_impl(N, c):
            a = np.arange(N)[::-1]
            fa = np.zeros(N)
            fia = np.zeros(N)
            with openmp("parallel"):
                with openmp("single"):
                    for i in range(len(a)):
                        e = a[i]
                        with openmp("task final(e >= c)"):
                            fa[i] = omp_in_final()
                            with openmp("task"):
                                fia[i] = omp_in_final()
            return fa, fia
        N, c = 25, 10
        r = test_impl(N, c)
        np.testing.assert_array_equal(r[0], np.concatenate(np.ones(N-c), np.zeros(c)))
        np.testing.assert_array_equal(r[0], r[1])


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

    def test_parallel_for_num_threads(self):
        def test_impl(nt):
            a = np.zeros(nt)
            with openmp("parallel num_threads(nt)"):
                with openmp("for"):
                    for i in range(nt):
                        a[i] = i
            return a
        self.check(test_impl, 15)

    def test_parallel_for_only_inside_var(self):
        @njit
        def test_impl(nt):
            a = np.zeros(nt)
            with openmp("parallel num_threads(nt) private(x)"):
                with openmp("for private(x)"):
                    for i in range(nt):
                        x = 0
                        #print("out:", i, x, i + x, nt)
                        a[i] = i + x
            return a
        nt = 12
        np.testing.assert_array_equal(test_impl(nt), np.arange(nt))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_parallel_for_ordered(self):
        @njit
        def test_impl(N, c):
            a = np.zeros(N)
            b = np.zeros(N)
            with openmp("parallel for ordered"):
                for i in range(1, N):
                    b[i] = b[i-1] + c
                    with openmp("ordered"):
                        a[i] = a[i-1] + c
            return a
        N, c = 30, 4
        r = test_impl(N, c)
        rc = np.arange(0, N*c, c)
        np.testing.assert_array_equal(r[0], rc)
        assert(not np.array_equal(r[1], rc))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_parallel_for_collapse(self):
        @njit
        def test_impl(n1, n2, n3):
            ia = np.zeros(n1)
            ja = np.zeros((n1, n2))
            ka = np.zeros((n1, n2, n3))
            with openmp("parallel for collapse(2)"):
                for i in range(n1):
                    ia[i] = omp_get_thread_num()
                    for j in range(n2):
                        ja[i][j] = omp_get_thread_num()
                        for k in range(n3):
                            ka[i][j][k] = omp_get_thread_num()
            return ia, ja, ka
        ia, ja, ka = test_impl(5, 3, 2)
        print(ia)
        print(ja)
        for a1i in range(len(ja)):
            with self.assertRaises(AssertionError) as raises:
                np.testing.assert_equal(ia[a1i], ja[a1i]) # Scalar to array
        for a1i in range(len(ka)):
            for a2i in range(a1i):
                # Scalar to array
                np.testing.assert_equal(ja[a1i][a2i], ka[a1i][a2i])


@linux_only
class TestOpenmpWorksharingSchedule(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    """
    def test_static_work_calculation(self):
        def test_impl(N, nt):
            v = np.zeros(N)
            step = -2
            omp_set_num_threads(nt)
            with openmp("parallel private(thread_num)"):
                running_omp = omp_in_parallel()
                thread_num = omp_get_thread_num()
                if not running_omp:
                    iters = N // abs(step)
                    itersPerThread = iters // nt
                    finishToThread = {}
                    for t in range(N):
                        f = itersPerThread*(t+1)-1 + min(iters%itersPerThread, t+1)
                        finishToThread[f] = t
                with openmp("for schedule(static)"):
                    for index, i in enumerate(range(N-1, N%2 - 1, -2)):
                        if not running_omp:
                            for finish in finishToThread.keys():
                                if index <= finish:
                                    thread_num = finishToThread[finish]
                        if i % (thread_num+1) == 0:
                            v[i] = i/(thread_num+1)
            print(v)
            return v
        self.check(test_impl, 100, 8)
    """

    # Giorgis pass doesn't support static with chunksize yet?
    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - unimplemented")
    def test_avg_sched_const(self):
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            with openmp("parallel for num_threads(nt) schedule(static, 4)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - unimplemented")
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

    def test_static_distribution(self):
        @njit
        def test_impl(nt, c):
            a = np.empty(nt*c)
            with openmp("parallel for num_threads(nt) schedule(static)"):
                for i in range(nt*c):
                    a[i] = omp_get_thread_num()
            return a
        nt, c = 8, 3
        r = test_impl(nt, c)
        for tn in range(nt):
            indices = np.sort(np.where(r == tn)[0])
            si = indices[0]
            np.testing.assert_array_equal(indices, np.arange(si, si+c))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_static_chunk_distribution(self):
        @njit
        def test_impl(nt, c, cs):
            a = np.empty(nt*c)
            with openmp("parallel for num_threads(nt) schedule(static, cs)"):
                for i in range(nt*c):
                    a[i] = omp_get_thread_num()
            return a
        nt, c, cs = 8, 6, 3
        r = test_impl(nt, c, cs)
        for tn in range(nt):
            indices = np.sort(np.where(r == tn)[0])
            for i in range(c//cs):
                si = indices[i*cs]
                np.testing.assert_array_equal(indices,
                   np.arange(si, min(len(r), si+cs)))

    def test_static_consistency(self):
        @njit
        def test_impl(nt, c, cs):
            a = np.empty(nt*c)
            b = np.empty(nt*c)
            with openmp("parallel num_threads(8)"):
                with openmp("for schedule(static)"):
                    for i in range(nt*c):
                        a[i] = omp_get_thread_num()
                with openmp("for schedule(static)"):
                    for i in range(nt*c):
                        b[i] = omp_get_thread_num()
            return a, b
        r = test_impl(8, 7, 5)
        np.testing.assert_array_equal(r[0], r[1])

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_dynamic_distribution(self):
        @njit
        def test_impl(nt, c, cs):
            a = np.empty(nt*c)
            with openmp("parallel for num_threads(nt) schedule(dynamic)"):
                for i in range(nt*c):
                    a[i] = omp_get_thread_num()
            return a
        nt, c, cs = 10, 2, 1
        r = test_impl(nt, c, cs)
        a = np.zeros(nt)
        for tn in range(nt):
            indices = np.sort(np.where(r == tn)[0])
            if len(indices > 0):
                for i in range(c//cs):
                    si = indices[i*cs]
                    np.testing.assert_array_equal(indices,
                    np.arange(si, min(len(r), si+cs)))
            else:
                a[tn] = 1
        assert(np.any(a))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_guided_distribution(self):
        @njit
        def test_impl(nt, c, cs):
            a = np.empty(nt*c)
            with openmp("parallel for num_threads(nt) schedule(guided, cs)"):
                for i in range(nt*c):
                    a[i] = omp_get_thread_num()
            return a
        nt, c, cs = 8, 6, 3
        r = test_impl(nt, c, cs)
        chunksizes = []
        cur_tn = r[0]
        cur_chunk = 0
        for e in r:
            if e == cur_tn:
                cur_chunk += 1
            else:
                chunksizes.append(cur_chunk)
                cur_chunk = 1
        chunksizes.append(cur_chunk)
        ca = np.array(chunksizes)
        np.testing.assert_array_equal(ca, np.sort(ca)[::-1])
        assert(ca[-2] >= cs)


@linux_only
class TestOpenmpParallelClauses(TestOpenmpBase):

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def test_num_threads_clause(self):
        @njit
        def test_impl(N, c1, c2):
            omp_set_dynamic(0)
            omp_set_max_active_levels(2)
            omp_set_num_threads(N + c1)
            d_count = 0
            n_count = 0
            nc_count = 0
            a_count = 0
            with openmp("parallel num_threads(N) shared(c2)"):
                with openmp("critical"):
                    d_count += 1
                with openmp("parallel"):
                    with openmp("critical"):
                        n_count += 1
                with openmp("single"):
                    with openmp("parallel num_threads(6)"):
                        with openmp("critical"):
                            nc_count += 1
            with openmp("parallel"):
                with openmp("critical"):
                    a_count += 1
            return d_count, a_count, n_count, nc_count
        a, b, c = 13, 3, 6
        r = test_impl(a, b, c)
        assert(r[0] == a)
        assert(r[1] == a + b)
        assert(r[2] == a * (a+b))
        assert(r[3] == c)

    def test_if_clause(self):
        @njit
        def test_impl(s):
            rp = 2 # Should also work with anything non-zero
            drp = 0
            ar = np.zeros(s, dtype=np.int32)
            adr = np.zeros(s, dtype=np.int32)
            par = np.full(s, 2, dtype=np.int32)
            padr = np.full(s, 2, dtype=np.int32)

            omp_set_num_threads(s)
            omp_set_dynamic(0)
            with openmp("parallel for if(rp)"):
                for i in range(s):
                    ar[omp_get_thread_num()] = 1
                    par[i] = omp_in_parallel()
            with openmp("parallel for if(drp)"):
                for i in range(s):
                    adr[omp_get_thread_num()] = 1
                    padr[i] = omp_in_parallel()
            return ar, adr, par, padr
        size = 20
        r = test_impl(size)
        np.testing.assert_array_equal(r[0], np.ones(size))
        rc = np.zeros(size)
        rc[0] = 1
        np.testing.assert_array_equal(r[1], rc)
        np.testing.assert_array_equal(r[2], np.ones(size))
        np.testing.assert_array_equal(r[3], np.zeros(size))

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

    # Uses apparently unsupported chunking.
    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - unimplemented")
    def test_avg_if_const(self):
        def test_impl(n, a):
            b = np.zeros(n)
            nt = 5
            with openmp("parallel for if(1) num_threads(nt) schedule(static, 4)"):
                for i in range(1, n):
                    b[i] = (a[i] + a[i-1]) / 2.0

            return b
        self.check(test_impl, 10, np.ones(10))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - unimplemented")
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
            y = np.zeros(N)
            z = 3.14
            i = 7
            with openmp("parallel private(i)"):
                yn = M + 1
                zs = z
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
                         lastprivate(zzzz) private(private_index) shared(a)
                          firstprivate(N) default(none)"""):
                for private_index in range(N):
                    y = private_index + x
                    a[private_index] = y
                    zzzz = private_index

            return a, zzzz
        self.check(test_impl, 100)

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
            si = 0
            with openmp("parallel for lastprivate(si)"):
                for i in range(N):
                    si = i + 1
                    a[i] = si
            return si, a
        N = 10
        r = test_impl(N)
        assert(r[0] == N)
        np.testing.assert_array_equal(r[1], np.arange(1, N+1))

    def test_lastprivate_non_one_step(self):
        @njit
        def test_impl(n1, n2, s):
            a = np.zeros(math.ceil((n2-n1) / s))
            rl = np.arange(n1, n2, s)
            with openmp("parallel for lastprivate(si)"):
                for i in range(len(rl)):
                    si = rl[i] + 1
                    a[i] = si
            return si, a
        n1, n2, s = 4, 26, 3
        r = test_impl(n1, n2, s)
        ra = np.arange(n1, n2, s) + 1
        assert(r[0] == ra[-1])
        np.testing.assert_array_equal(r[1], ra)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_lastprivate_sections(self):
        @njit
        def test_impl(N2, si):
            a = np.zeros(N2)
            with openmp("parallel shared(sis1)"):
                with openmp("sections lastprivate(si)"):
                    sis1 = si
                    # N1 = number of sections
                    with openmp("section"):
                        si = 0
                    with openmp("section"):
                        si = 1
                    with openmp("section"):
                        si = 2
                sis2 = si
                with openmp("sections lastprivate(si)"):
                    # N2 = number of sections
                    with openmp("section"):
                        i = 0
                        si = N2 - i
                        a[i] = si
                    with openmp("section"):
                        i = 1
                        si = N2 - i
                        a[i] = si
                    with openmp("section"):
                        i = 2
                        si = N2 - i
                        a[i] = si
                    with openmp("section"):
                        i = 3
                        si = N2 - i
                        a[i] = si
            return si, sis1, sis2, a
        N1, N2, d = 3, 4, 5
        r = test_impl(N2, d)
        assert(r[0] == 1)
        assert(r[1] != d)
        assert(r[2] == N1 - 1)
        np.testing.assert_array_equal(r[3], np.arange(N2, 0, -1))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_lastprivate_conditional(self):
        @njit
        def test_impl(N, c1, c2):
            a = np.arange(0, N*2, c2)
            num = 0
            with openmp("parallel"):
                with openmp("for lastprivate(conditional: num)"):
                    for i in range(N):
                        if i < c1:
                            num = a[i] + c2
            return num
        c1, c2 = 11, 3
        assert(test_impl(15, c1, c2) == c1 * c2)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_copyin(self):
        @njit
        def test_impl(nt, n1, n2, n3):
            xsa1 = np.zeros(nt)
            xsa2 = np.zeros(nt)
            x = n1
            with openmp("threadprivate(x)"):
                pass
            x = n2
            with openmp("parallel num_threads(nt) copyin(x) private(tn)"):
                tn = omp_get_thread_num()
                xsa1[tn] = x
                if tn == 0:
                    x = n3
            with openmp("parallel copyin(x)"):
                xsa2[omp_get_thread_num()] = x
            return xsa1, xsa2
        nt, n2, n3 = 10, 12.5, 7.1
        r = test_impl(nt, 4.3, n2, n3)
        np.testing.assert_array_equal(r[0], np.full(nt, n2))
        np.testing.assert_array_equal(r[1], np.full(nt, n3))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_copyin_nested(self):
        def test_impl(nt1, nt2, mt, n1, n2, n3):
            omp_set_nested(1)
            omp_set_dynamic(0)
            xsa1 = np.zeros(nt1)
            xsa2 = np.zeros(nt2)
            x = n1
            with openmp("threadprivate(x)"):
                pass
            x = n2
            with openmp("parallel num_threads(nt1) copyin(x) private(tn)"):
                tn = omp_get_thread_num()
                xsa1[tn] = x
                if tn == mt:
                    x = n3
                    with openmp("parallel num_threads(nt2) copyin(x)"):
                        xsa2[omp_get_thread_num()] = x
            return xsa1, xsa2
        nt1, nt2, n2, n3 = 10, 4, 12.5, 7.1
        r = test_impl(nt1, nt2, 2, 4.3, n2, n3)
        np.testing.assert_array_equal(r[0], np.full(nt1, n2))
        np.testing.assert_array_equal(r[1], np.full(nt2, n3))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_copyprivate(self):
        @njit
        def test_impl(nt, n1, n2, n3):
            x = n1
            a = np.zeros(nt)
            xsa = np.zeros(nt)
            ar = np.zeros(nt)
            omp_set_num_threads(nt)
            with openmp("parallel firstprivate(x, a) private(tn)"):
                with openmp("single copyprivate(x, a)"):
                    x = n2
                    a = np.full(nt, n3)
                tn = omp_get_thread_num()
                xsa[tn] = x
                ar[tn] = a[tn]
            return xsa, a, ar
        nt, n2, n3 = 16, 12, 3
        r = test_impl(nt, 5, n2, n3)
        np.testing.assert_array_equal(r[0], np.full(nt, n2))
        self.assert_outputs_equal(r[1], r[2], np.full(nt, n3))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_linear_clause(self):
        @njit
        def test_impl(N):
            a = np.arange(N) + 1
            b = np.zeros(N//2)

            linearj = 0
            with openmp("parallel for linear(linearj:1)"):
                for i in range(0, N, 2):
                    b[linearj] = a[i] * 2

            return b, linearj
        N = 50
        r = test_impl(N)
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
                print("Fail")
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
                print("Fail")
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

    #def test_parallel_for_blocking_if(self):
    #    @njit
    #    def test_impl():
    #        n = 0
    #        with openmp("parallel"):
    #            half_threads = omp_get_num_threads()//2
    #            if omp_get_thread_num() < half_threads:
    #                with openmp("for reduction(+:n)"):
    #                    for _ in range(half_threads):
    #                        n += 1
    #        return n

    #    #with self.assertRaises(AssertionError) as raises:
    #     #   njit(test_impl)
    #    test_impl()
    #    #print(str(raises.exception))

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

    def test_parallel_for_nowait(self):
        @njit
        def test_impl(nt):
            a = np.zeros(nt)
            with openmp("parallel for num_threads(nt) nowait"):
                for i in range(nt):
                    a[omp_get_thread_num] = i
            return a

        with self.assertRaises(Exception) as raises:
            test_impl(12)
        self.assertIn("No terminal matches", str(raises.exception))

    def test_parallel_double_num_threads(self):
        @njit
        def test_impl(nt1, nt2):
            count = 0
            with openmp("parallel num_threads(nt1) num_threads(nt2)"):
                with openmp("critical"):
                    count += 1
            print(count)
            return count

        with self.assertRaises(Exception) as raises:
            test_impl(5, 7)

    def test_conditional_barrier(self):
        @njit
        def test_impl(nt):
            hp = nt//2
            a = np.zeros(hp)
            b = np.zeros(nt - hp)
            with openmp("parallel num_threads(nt) private(tn)"):
                tn = omp_get_thread_num()
                if tn < hp:
                    with openmp("barrier"):
                        pass
                    a[tn] = 1
                else:
                    with openmp("barrier"):
                        pass
                    b[tn - hp] = 1
            return a, b

        # The spec seems to say this should be an error but in practice maybe not?
        #with self.assertRaises(Exception) as raises:
        test_impl(12)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Hangs")
    def test_closely_nested_for_loops(self):
        @njit
        def test_impl(N):
            a = np.zeros((N, N))
            with openmp("parallel"):
                with openmp("for"):
                    for i in range(N):
                        with openmp("for"):
                            for j in range(N):
                                a[i][j] = 1
            return a
        with self.assertRaises(Exception) as raises:
            test_impl(4)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Hangs")
    def test_nested_critical(self):
        @njit
        def test_impl():
            num = 0
            with openmp("parallel"):
                with openmp("critical"):
                    num += 1
                    with openmp("critical"):
                        num -= 1
            return num

        with self.assertRaises(Exception) as raises:
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

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
            with openmp("parallel private(sum) shared(c)"):
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

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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

    # Revisit - how to prove atomic works without a race condition?
    #def test_atomic_threads(self):
    #    def test_impl(N, iters):
    #        omp_set_num_threads(N)
    #        count = 0
    #        p = 0
    #        sum = 0
    #        with openmp("parallel"):
    #            with openmp("barrier"):
    #                pass
    #            with openmp("for private(p, sum)"):
    #                for _ in range(iters):
    #                    with openmp("atomic"):
    #                        p = count
    #                        sum = 0
    #                        for i in range(10000):
    #                            if i % 2 == 0:
    #                                sum += 1
    #                            else:
    #                                sum -= 1
    #                        p += 1 + sum
    #                        count = p
    #        return count
    #    iters = 1000
    #    self.check(test_impl, 2, iters)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
            return a
        nt, N, c = 27, 8, 6
        rc = np.zeros(N)
        #ba = np.zeros(nt)
        #for i in range(nt):
        #    index = i % N
        #    rc[index] = nt % c
        #print("rc1:", rc)

        #for i in range(nt):
        #    index = i % N
        #    ba[i] = rc[index-1] + index

        #for i in range(nt):
        #    index = i % N
        #    rc[index] += ba[i]
        #print("rc2:", rc)

        for i in range(nt):
            index = i % N
            ts = nt//N
            ts += 1 if index < nt % N else 0
            rc[index] = nt%c + (nt%c + index) * ts

        np.testing.assert_array_equal(test_impl(nt, N, c), rc)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def test_parallel_sections(self):
        @njit
        def test_impl(nt):
            ta0 = np.zeros(nt)
            ta1 = np.zeros(nt)
            secpa = np.zeros(nt)

            with openmp("parallel sections num_threads(nt)"):
                with openmp("section"):
                    ta0[omp_get_thread_num()] += 1
                    secpa[0] = omp_in_parallel()
                with openmp("section"):
                    ta1[omp_get_thread_num()] += 1
                    secpa[1] = omp_in_parallel()
            print(ta0, ta1)
            return ta0, ta0, secpa
        NT = 2  # Must equal the number of section directives in the test
        r = test_impl(NT)
        assert(np.sum(r[0]) == 1)
        assert(np.sum(r[1]) == 1)
        assert(np.sum(r[2]) == NT)
        np.testing.assert_array_equal(r[0] + r[1], np.ones(NT))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Abort - needs fix")
    def test_barrier(self):
        @njit
        def test_impl(nt, iters, c):
            a = np.zeros(nt)
            ac = np.zeros((nt, nt))
            x = iters//c
            iters = x * c
            sum = 0
            with openmp("parallel num_threads(nt) private(tn, sum)"):
                tn = omp_get_thread_num()
                with openmp("critical"):
                    sum = 0
                    for i in range(iters):
                        if i % x == 0:
                            sum += 1
                    a[tn] = sum
                with openmp("barrier"):
                    pass
                for j in range(nt):
                    ac[tn][j] = a[j]
            return ac
        nt, c = 15, 12
        r = test_impl(nt, 10000, c)
        a = np.full(nt, c)
        for i in range(nt):
            np.testing.assert_array_equal(r[i], a)

#    def test_for_nowait(self):
#        @njit
#        def test_impl(nt, n, c1, c2):
#            a = np.zeros(n)
#            b = np.zeros(n)
#            ac = np.zeros((nt, n))
#            sum = 0
#            with openmp("parallel num_threads(nt) private(tn)"):
#                tn = omp_get_thread_num()
#                with openmp("for nowait schedule(static) private(sum)"):
#                    for i in range(n):
#                        # Sleep
#                        sum = 0
#                        for j in range(i * 1000):
#                            if j % 2 == 0:
#                                sum += 1
#                            else:
#                                sum -= 1
#                        a[i] = i * c1 + sum
#                for j in range(nt):
#                    ac[tn][j] = a[j]
#                with openmp("for schedule(static)"):
#                    for i in range(n):
#                        b[i] = a[i] + c2
#            return b, ac
#        nt, n, c1, c2 = 8, 30, 5, -7
#        r = test_impl(nt, n, c1, c2)
#        a = np.arange(n) * c1
#        np.testing.assert_array_equal(r[0], a + c2)
#        arc = [np.array_equal(r[1][i], a) for i in range(nt)]
#        assert(not np.all(arc))
#
#    def test_nowait_result(self):
#        def test_impl(n, m, a, b, y, z):
#            omp_set_num_threads(5)
#
#            with openmp("parallel"):
#                with openmp("for nowait"):
#                    for i in range(1, n):
#                        b[i] = (a[i] + a[i-1]) / 2.0
#                with openmp("for nowait"):
#                    for i in range(m):
#                        y[i] = math.sqrt(z[i])
#
#            return b, y
#        n, m = 10, 20
#        self.check(test_impl, n, m, np.ones(n), np.zeros(n),
#                    np.zeros(m), np.full(m, 13))

    def test_nested_parallel_for(self):
        @njit
        def test_impl(nt):
            omp_set_num_threads(nt)
            omp_set_nested(1)
            omp_set_dynamic(0)
            a = np.zeros((nt, nt), dtype=np.int32)
            with openmp("parallel for"):
                for i in range(nt):
                    with openmp("parallel for"):
                        for j in range(nt):
                            a[i][j] = omp_get_thread_num()
            return a
        nt = 8
        r = test_impl(nt)
        for i in range(len(r)):
            np.testing.assert_array_equal(np.sort(r[i]), np.arange(nt))

    def test_nested_parallel_regions_1(self):
        @njit
        def test_impl(nt1, nt2):
            omp_set_dynamic(0)
            omp_set_max_active_levels(2)
            ca = np.zeros(nt1)
            omp_set_num_threads(nt1)
            with openmp("parallel private(tn)"):
                tn = omp_get_thread_num()
                with openmp("parallel num_threads(3)"):
                    with openmp("critical"):
                        ca[tn] += 1
                    with openmp("single"):
                        ats = omp_get_ancestor_thread_num(1) == tn
                        ts = omp_get_team_size(1)
            return ca, ats, ts
        nt1, nt2 = 6, 3
        r = test_impl(nt1, nt2)
        np.testing.assert_array_equal(r[0], np.full(nt1, nt2))
        assert(r[1] == True)
        assert(r[2] == nt1)

    def test_nested_parallel_regions_2(self):
        @njit
        def set_array(a):
            tn = omp_get_thread_num()
            a[tn][0] = omp_get_max_active_levels()
            a[tn][1] = omp_get_num_threads()
            a[tn][2] = omp_get_max_threads()
            a[tn][3] = omp_get_level()
            a[tn][4] = omp_get_team_size(1)
            a[tn][5] = omp_in_parallel()

        @njit
        def test_impl(mal, n1, n2, n3):
            omp_set_max_active_levels(mal)
            omp_set_dynamic(0)
            omp_set_num_threads(n1)
            a = np.zeros((n2, 6), dtype=np.int32)
            b = np.zeros((n1, 6), dtype=np.int32)
            with openmp("parallel"):
                omp_set_num_threads(n2)
                with openmp("single"):
                    with openmp("parallel"):
                        omp_set_num_threads(n3)
                        set_array(a)
                set_array(b)

            return a, b
        mal, n1, n2, n3 = 8, 2, 4, 5
        a, b = test_impl(mal, n1, n2, n3)
        for i in range(n2):
            np.testing.assert_array_equal(a[i], np.array([8, n2, n3, 2, n1, 1]))
        for i in range(n1):
            np.testing.assert_array_equal(b[i], np.array([8, n1, n2, 1, n1, 1]))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled,
                        "Abort / Segmentation Fault")
    def test_parallel_two_dimensional_array(self):
        @njit
        def test_impl(N):
            omp_set_dynamic(0)
            omp_set_num_threads(N)
            a = np.zeros((N, 2), dtype=np.int32)
            with openmp("parallel private(tn)"):
                tn = omp_get_thread_num()
                a[tn][0] = 1
                a[tn][1] = 2
            return a
        N = 5
        r = test_impl(N)
        for i in range(N):
            np.testing.assert_array_equal(r[i], np.array([1, 2]))

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

    @unittest.skipUnless(TestOpenmpBase.skip_disabled,
                    "Sometimes segmentation fault")
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
                        xa = x == n1
                        ya = y == n1
                        x, y = n2, n2
                    with openmp("taskwait"):
                        ysave = y
            return (x, ysave), (xa, ya)
        n1, n2 = 1, 2
        r = test_impl(n1, n2)
        self.assert_outputs_equal(r[1], (True, True))
        self.assert_outputs_equal(r[0], (n2, n1))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Segmentation fault")
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
        np.testing.assert_array_equal(r, np.ones(ntsks))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Segmentation fault")
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
            np.testing.assert_array_equal(r, np.ones(ntsks))

    # Error with commented out code, other version never finished running
    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Error")
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
                        ret = np.all(a)
            return ret
        assert(test_impl(4))

    def test_taskwait(self):
        def test_impl(ntsks):
            a = np.zeros(ntsks)
            with openmp("parallel private(i)"):
                with openmp("single"):
                    for i in range(ntsks):
                        with openmp("task private(sum) private(j)"):
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

    @unittest.skipUnless(TestOpenmpBase.skip_disabled,
                        "Sometimes segmentation fault")
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
            np.testing.assert_array_equal(r[1], np.ones(r[1].shape))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
                    flag = 0
            return r
        assert(test_impl())

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
            np.testing.assert_array_equal(sids, cids)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
        assert(np.any(yt))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
        np.testing.assert_array_equal(fns[c:], ins[c:])
        np.testing.assert_array_equal(da, np.ones(ntsks))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
        rc = np.zeros(ntsks)
        for i in range(ntsks):
            rc[i] = sum(range(i+1, ntsks+1))
        np.testing.assert_array_equal(r, rc)

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
            return a, da
        self.check(test_impl, 15)

    # Affinity clause should not affect result
    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
@unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
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
@unittest.skipUnless(TestOpenmpBase.skip_disabled or
                    TestOpenmpBase.run_target, "Unimplemented")
class TestOpenmpTarget(TestOpenmpBase):
    devices = [0]
    if TestOpenmpBase.test_device_1:
        devices.append(1)

    def __init__(self, *args):
        TestOpenmpBase.__init__(self, *args)

    def target_map_tofrom_scalar_parallel(self, device):
        for explicit in [True, False]:
            with self.subTest(explicit_shared=explicit):
                target_pragma = f"target device({device}) map(to: n) map(tofrom: a)"
                parallel_pragma = "parallel" + (" shared(a)" if explicit else "")
                @njit
                def test_impl(n):
                    a = 1
                    with openmp(target_pragma):
                        with openmp(parallel_pragma):
                            a = n
                    return a
                n = 123
                assert(test_impl(n) == n)

    def target_map_tofrom_scalar_teams_nested(self, device):
        target_pragma = f"target device({device}) map(to: n) map(tofrom: a)"
        @njit
        def test_impl(n):
            a = 1
            with openmp(target_pragma):
                with openmp("teams"):
                    a = n
            return a
        n = 123
        assert(test_impl(n) == n)

    def target_map_to_var(self, device):
        target_pragma = f"target device({device}) map(to: x, n2) map(from: xs)"
        @njit
        def test_impl(n1, n2):
            x = n1
            with openmp(target_pragma):
                xs = x
                x = n2
            return x, xs
        n1, n2 = 3, 5
        r = test_impl(n1, n2)
        assert(r[0] == n1)
        assert(r[1] == n1)

    def target_map_from_var(self, device):
        target_pragma = f"target device({device}) map(to: n2) map(from: x)"
        @njit
        def test_impl(n1, n2):
            x = n1
            with openmp(target_pragma):
                # x would be undefined at this point
                # xs = x
                x = n2
            return (x, )#xs
        n1, n2 = 3, 5
        r = test_impl(n1, n2)
        assert(r[0] == n2)
        # assert(r[1] != n1)

    def target_map_tofrom_scalar_var(self, device):
        target_pragma = f"target device({device}) map(to: n2) map(from: xs) map(tofrom: x)"
        @njit
        def test_impl(n1, n2):
            x = n1
            with openmp(target_pragma):
                xs = x
                x = n2
            return x, xs
        n1, n2 = 3, 5
        r = test_impl(n1, n2)
        assert(r[0] == n2)
        assert(r[1] == n1)

    def target_multiple_tofrom_scalar(self, device):
        target_pragma = f"target device({device}) map(tofrom: x)"
        @njit
        def test_impl(n1, n2, n3):
            x = n1
            with openmp(target_pragma):
                x = n2
            xs1 = x
            with openmp(target_pragma):
                x = n3
            return xs1, x
        n2, n3 = 2, 3
        r = test_impl(1, n2, n3)
        assert(r[0] == n2)
        assert(r[1] == n3)

    def target_map_from_array(self, device):
        target_pragma = f"target device({device}) map(from: a)"
        @njit
        def test_impl(n1, n2):
            a = np.zeros(n1)
            with openmp(target_pragma):
                for i in range(len(a)):
                    a[i] = n2
            return a
        n1, n2 = 75, 3
        np.testing.assert_array_equal(test_impl(n1, n2), np.full(n1, n2))

    def target_map_tofrom_array(self, device):
        target_pragma = f"target device({device}) map(tofrom: a)"
        @njit
        def test_impl(s, n1, n2):
            a = np.full(s, n1)
            with openmp(target_pragma):
                for i in range(len(a)):
                    a[i] += n2
            return a
        s, n1, n2 = 75, 3, 13
        np.testing.assert_array_equal(test_impl(s, n1, n2), np.full(s, n1 + n2))

    def target_parallel_for_index(self, device):
        target_pragma = f"target device({device}) map(to: iters, nt) map(tofrom: a)"
        @njit
        def test_impl(nt, c):
            iters = nt * c
            a = np.zeros(iters)
            with openmp(target_pragma):
                with openmp("parallel for num_threads(nt)"):
                    for i in range(iters):
                        a[i] = i
            return a
        nt, c = 8, 3
        np.testing.assert_array_equal(test_impl(nt, c), np.arange(nt*c))

    def target_parallel_for_addition(self, device):
        target_pragma = f"target device({device}) map(to: c) map(tofrom: a, b)"
        @njit
        def test_impl(s, v, c):
            a = np.full(s, v)
            b = np.empty(a.shape[0])
            with openmp (target_pragma):
                with openmp ("parallel for"):
                    for i in range(len(a)):
                        b[i] = a[i] + c
            return b
        s, v, c = 50000, 2, 5
        np.testing.assert_array_equal(test_impl(s, v, c), np.full(50000, v + c))

    def target_scalar_firstprivate_default(self, device):
        for explicit in [True, False]:
            with self.subTest(explicit_firstprivate=explicit):
                target_pragma = f"target device({device})" + " firstprivate(a)" if explicit else ""
                def test_impl(n1, n2):
                    a = n1
                    with openmp(target_pragma):
                        a = n2
                    return a
                # Assuming default for scalars is firstprivate
                n = 5
                print(test_impl(n, n + 1))
                assert(test_impl(n, n + 1) == n)

    def target_firstprivate_parallel(self, device):
        target_pragma = f"target device({device}) firstprivate(count) map(to: nt) map(from: cs)"
        @njit
        def test_impl(nt, n1, n2):
            count = n1
            with openmp(target_pragma):
                count += n2
                with openmp("parallel num_threads(nt)"):
                    with openmp("critical"):
                        count += 1
                count += n2
                cs = count
            return count, cs
        nt, n1, n2 = 8, 2, 3
        r = test_impl(nt, n1, n2)
        assert(r[0] == n1)
        assert(r[1] == n1 + nt + 2*n2)

    def target_data_nested(self, device):
        target_pragma = f"""target data device({device}) map(to: a)
                        map(tofrom: b) map(from: as1, as2, bs1, bs2)"""
        @njit
        def test_impl(s, n1, n2):
            a = np.full(s, n1)
            as1 = np.empty(s, dtype=a.dtype)
            as2 = np.empty(s, dtype=a.dtype)
            b = n1
            with openmp(target_pragma):
                with openmp("target"):
                    as1[:] = a
                    bs1 = b
                with openmp("target"):
                    for i in range(s):
                        a[i] = n2
                    b = n2
                with openmp("target"):
                    as2[:] = a
                    bs2 = b
            return a, as1, as2, b, bs1, bs2

        s, n1, n2 = 50, 1, 2
        ao, a1, a2, bo, b1, b2 = test_impl(s, n1, n2)
        np.testing.assert_array_equal(ao, np.full(s, n1))
        np.testing.assert_array_equal(a1, np.full(s, n1))
        np.testing.assert_array_equal(a2, np.full(s, n2))
        assert(bo == n2)
        assert(b1 == n1)
        assert(b2 == n2)

    def target_enter_data(self, device):
        target_enter_pragma = f"""target enter data device({device})
                            map(to: a) map(to: b)"""
        target_exit_pragma = f"""target exit data device({device})
                            map(from: b, as1, as2, bs1, bs2)"""
        @njit
        def test_impl(s, n1, n2):
            a = np.full(s, n1)
            as1 = np.empty(s)
            as2 = np.empty(s)
            b = n1

            #def setToValueRegion(val):
            #    nonlocal a, b
            #    with openmp("target"):
            #        for i in range(s):
            #            a[i] = val
            #        b = val

            with openmp(target_enter_pragma):
                pass

            with openmp("target"):
                as1[:] = a
                bs1 = b
            #setToValueRegion(n2)
            with openmp("target"):
                for i in range(s):
                    a[i] = n2
                b = n2
            with openmp("target"):
                as2[:] = a
                bs2 = b

            with openmp(target_exit_pragma):
                pass

            return a, as1, as2, b, bs1, bs2

        s, n1, n2 = 50, 1, 2
        ao, a1, a2, bo, b1, b2 = test_impl(s, n1, n2)
        np.testing.assert_array_equal(ao, np.full(s, n1))
        np.testing.assert_array_equal(a1, np.full(s, n1))
        np.testing.assert_array_equal(a2, np.full(s, n2))
        assert(bo == n2)
        assert(b1 == n1)
        assert(b2 == n2)

    def target_enter_data_alloc(self, device):
        target_enter_pragma = f"""target enter data device({device})
                                map(alloc: v1, v2)"""
        target_exit_pragma = f"target exit data device({device}) map(from: v1, v2)"
        @njit
        def test_impl(s, n1, n2):
            v1 = np.empty(s)
            v2 = np.empty(s)
            p = np.empty(s)

            with openmp(target_enter_pragma):
                pass

            with openmp("target nowait depend(out: v1, v2)"):
                for i in range(s):
                    v1[i] = n1
                    v2[i] = n2

            with openmp("task"): # execute asynchronously on host device
                sum = 0
                for _ in range(10000):
                    sum += 1

            with openmp("target map(from: p) nowait depend(in: v1, v2)"):
                with openmp("distribute parallel for"):
                    for i in range(s):
                        p[i] = v1[i] * v2[i]

            with openmp("taskwait"):
                pass

            with openmp(target_exit_pragma):
                pass

            return p
        
        s, n1, n2 = 150, 7, 9
        np.testing.assert_array_equal(test_impl(s, n1, n2), np.full(s, n1 * n2))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def target_teams_distribute_parallel_for(self, device):
        for simd in [True, False]:
            with self.subTest(simd=simd):
                nt = 125
                target_pragma = f"""target teams distribute parallel for{" simd" if simd else ""}
                                device({device}) num_teams(5) map(tofrom: a)"""
                @njit
                def test_impl(nt):
                    a = np.zeros(nt)
                    with openmp(target_pragma):
                        for i in range(nt):
                            a[i] = i
                    return a
                np.testing.assert_array_equal(test_impl(nt), np.arange(nt))

    @unittest.skipUnless(TestOpenmpBase.skip_disabled, "Unimplemented")
    def target_teams_distribute_parallel_for_proof(self, device):
        nt = 5
        target_pragma = f"""target teams distribute parallel for
                        device({device}) num_teams({nt}) map(tofrom: team_num_arr, thread_num_arr)""" 
        @njit
        def test_impl(nt):
            team_num_arr = np.ones(nt)
            thread_num_arr = np.ones(nt)
            with openmp(target_pragma):
                for i in range(nt):
                    team_num_arr[i] = omp_get_team_num()
                    thread_num_arr[i] = omp_get_thread_num()
            return team_num_arr, thread_num_arr
        r = test_impl(nt)
        np.testing.assert_array_equal(r[0].sort(), np.arange(nt))
        np.testing.assert_array_equal(r[1], np.zeros(nt))

    """

    def test_target_threads(self):
        @njit
        def test_impl(nt):
            tts, tip = 0, -1
            pts, pip = -1, -1
            ttna = np.zeros(nt)
            ptna = np.zeros(nt)

            with openmp("target map(from: tts, ttna, tip, pts, ptna, pip)"):
                tts = omp_get_team_size(1)
                ttna[omp_get_thread_num()] = 1
                tip = omp_in_parallel()
                with openmp("parallel num_threads(nt)"):
                    pts = omp_get_team_size(1)
                    ptna[omp_get_thread_num()] = 1
                    pip = omp_in_parallel()
            return (tts, ttna, tip), (pts, ptna, pip)
        nt = 8
        r = test_impl(nt)
        assert(r[0][0] == -1)
        ttnc = np.zeros(nt)
        ttnc[0] = 1
        assert(r[0][1] == ttnc)
        assert(r[0][2] == 0)
        assert(r[1][0] == nt)
        np.testing.assert_array_equal(r[1][1], np.ones(nt))
        assert(r[1][2] == 1)

    def test_target_defaultmap(self):
        @njit
        def test_impl(n1, n2, n3):
            x = n1
            xs = -1
            a = np.full(n3, n4)
            with openmp("target defaultmap(from: scalar) defaultmap(tofrom: aggregate) map(from: xs) map(to: n2, n3)"):
                xs = x
                x = n2
                for i in range(n3):
                    a[i] += 1
            return x, xs, a
        n1, n2, n3, n4 = 2, 3, 10, 5
        r = test_impl(n1, n2, n3)
        assert(r[0] == n2)
        assert(r[1] != n1)
        np.testing.assert_array_equal(r[2], np.full(n3, n4+1))

    def test_target_teams_threads(self):
        @njit
        def test_impl(nt, tl):
            nta = np.zeros(nt)
            tna = np.zeros(nt)
            tia = np.zeros(nt)
            tsa = np.zeros(nt)
            with openmp("target teams num_teams(nt) thread_limit(tl) private(tn)"):
                tn = omp_get_team_num()
                nta[tn] = omp_get_num_teams()
                tna[tn] = tn + 1
                tia[tn] = omp_get_thread_num() + 1
                with openmp("parallel"):
                    tsa[tn] = omp_get_team_size(1)
            return nta, tna, tia, tsa
        nt, tl = 5, 3
        r = test_impl(nt, tl)
        assert(1 <= r[0][0] <= nt)
        np.testing.assert_equal(r[0][0], r[0][r[0] != 0])
        np.testing.assert_array_equal(r[1][r[1] != 0], np.arange(1, nt+1))
        np.testing.assert_equal(1, r[2][r[2] != 0])
        assert(1 <= r[3][0] <= tl)
        np.testing.assert_equal(r[3][0], r[3][r[3] != 0])
    """

for memberName in dir(TestOpenmpTarget):
    if memberName.startswith("target"):
        test_func = getattr(TestOpenmpTarget, memberName)
        def make_func_with_subtest(func):
            def func_with_subtest(self):
                for device in TestOpenmpTarget.devices:
                    with self.subTest(device=device):
                        func(self, device)
            return func_with_subtest
        setattr(TestOpenmpTarget, "test_" + test_func.__name__, make_func_with_subtest(test_func))

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
