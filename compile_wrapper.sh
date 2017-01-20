#!/bin/bash

SOURCE_DIR=$1
shift
BUILD_DIR=$1
shift
TARGET_NAME=$1
shift

echo "Creating skeletopyze build directory in $BUILD_DIR"
mkdir $BUILD_DIR
cd $BUILD_DIR
rm CMakeCache.txt
echo "Calling CMake for sources in $SOURCE_DIR"
echo "Setting boost search dir to $PREFIX"
cmake $SOURCE_DIR \
  -DBOOST_ROOT=${PREFIX} \
  -DBoost_NO_SYSTEM_PATHS=ON
  #-DBoost_PYTHON_LIBRARY=${PREFIX}/lib/libboost_python.${DYLIB_EXT} \
  #-DBoost_PYTHON_LIBRARY_RELEASE=${PREFIX}/lib/libboost_python.${DYLIB_EXT} \
  #-DBoost_PYTHON_LIBRARY_DEBUG=${PREFIX}/lib/libboost_python.${DYLIB_EXT} \
echo "Creating target $TARGET_NAME"
make -j3 "$@" $TARGET_NAME
