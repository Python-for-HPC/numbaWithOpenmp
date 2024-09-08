#!/bin/bash
set -e

hostname="$(hostname)"
machine=${hostname//[0-9]/}

source /usr/workspace/ggeorgak/${machine}/miniconda3-env.sh
conda create -y -p ${CI_BUILDS_DIR}/test-numba-pyomp -c python-for-hpc -c conda-forge --override-channels \
    llvmdev=14.0.6 numpy=1.24 lark-parser cffi python=3.10 \
    llvm-openmp-dev=14.0.6 \
    llvmlite=pyomp_0.40
conda activate ${CI_BUILDS_DIR}/test-numba-pyomp
pushd ${CI_PROJECT_DIR}
MACOSX_DEPLOYMENT_TARGET=10.10 python setup.py \
    build_static build_ext build install --single-version-externally-managed --record=record.txt || { echo "Build failed"; exit 1; }
popd

pushd ${CI_BUILDS_DIR}
echo "=> Test Numba PyOMP Host"
TEST_DEVICES=0 RUN_TARGET=0 python -m numba.runtests -- numba.tests.test_openmp \
    || { echo "Test Numba PyOMP Host failed"; exit 1; }

if [[ "${machine}" == "lassen" ]]; then
    echo "=> Test Numba PyOMP Target GPU device(0)"
    TEST_DEVICES=0 RUN_TARGET=1 python -m numba.runtests -- numba.tests.test_openmp.TestOpenmpTarget \
        || { echo "Test Numba PyOMP Target GPU failed"; exit 1; }
fi

# TODO: Enable once we fix codegen in the backend.
#if [[ "${machine}" == "ruby" ]]; then
#    echo "=> Test Numba PyOMP Target Host device(1)"
#    TEST_DEVICES=1 RUN_TARGET=1 python -m numba.runtests -- numba.tests.test_openmp.TestOpenmpTarget \
#        || { echo "Test Numba PyOMP Target Host failed"; exit 1; }
#fi

popd
