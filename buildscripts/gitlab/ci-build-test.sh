#!/bin/bash
set -e

hostname="$(hostname)"
machine=${hostname//[0-9]/}

source /usr/workspace/ggeorgak/${machine}/miniconda3-env.sh
conda create -y -p ${CI_BUILDS_DIR}/test-numba-pyomp -c python-for-hpc -c conda-forge --override-channels \
    llvmdev llvmlite numpy=1.24 lark-parser cffi python=3.10
conda activate ${CI_BUILDS_DIR}/test-numba-pyomp
pushd ${CI_PROJECT_DIR}
mkdir -p ${CI_PROJECT_DIR}/numba/bin
mkdir -p ${CI_PROJECT_DIR}/numba/lib
cp ${CONDA_PREFIX}/bin/clang      ${CI_PROJECT_DIR}/numba/bin
cp ${CONDA_PREFIX}/bin/opt        ${CI_PROJECT_DIR}/numba/bin
cp ${CONDA_PREFIX}/bin/llc        ${CI_PROJECT_DIR}/numba/bin
cp ${CONDA_PREFIX}/bin/llvm-link  ${CI_PROJECT_DIR}/numba/bin
cp ${CONDA_PREFIX}/lib/lib*omp*   ${CI_PROJECT_DIR}/numba/lib
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

if [[ "${machine}" == "ruby" ]]; then
    echo "=> Test Numba PyOMP Target Host device(1)"
    TEST_DEVICES=1 RUN_TARGET=1 python -m numba.runtests -- numba.tests.test_openmp.TestOpenmpTarget \
        || { echo "Test Numba PyOMP Target Host failed"; exit 1; }
fi

popd
