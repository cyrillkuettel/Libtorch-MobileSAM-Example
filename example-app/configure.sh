#!/usr/bin/env bash


# Get the directory of the script

# Get the directory of the script
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)

# use a little python inline just for convenience, we need to get the absolute path
LIBTORCH_PATH=$(python3 -c "
import os
libtorch_path = os.path.abspath(os.path.join('$SCRIPT_DIR', '..', 'libtorch'))
if os.path.isdir(libtorch_path):
    print(libtorch_path)
else:
    exit(1)
")

if [ $? -ne 0 ]; then
  echo 'Error: libtorch directory not found or Python script failed!' >&2
  exit 1
else
  echo $LIBTORCH_PATH
fi

BUILD_DIR="build"
mkdir -p $BUILD_DIR

# Check if the build directory is not empty
if [ "$(ls -A $BUILD_DIR)" ]; then
    echo "The build directory is not empty. Do you want to delete its contents? (y/n)"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        echo "Deleting contents of the build directory..."
        rm -rf $BUILD_DIR/*
    else
        echo "Exiting the script as per user request."
        exit 1
    fi
fi

cd $BUILD_DIR

echo "Running cmake -DCMAKE_BUILD_TYPE=Debug  "
# Use the discovered LIBTORCH_PATH (the absolute path to libtorch) in the cmake command
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" ..
