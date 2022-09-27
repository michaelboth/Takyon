#!/bin/bash

function cleanFolder {
    echo ""
    echo "Cleaning: $1"
    cd $1
    if [ $? -ne 0 ]; then
        echo "Failed to go to $1 folder"
        exit 1
    fi
    make clean
    if [ $? -ne 0 ]; then
        echo "Failed to clean $1"
        exit 1
    fi
}

function build {
    echo ""
    echo "Building: `pwd`"
    $1
    if [ $? -ne 0 ]; then
        echo "Failed to compile"
        exit 1
    fi
}

debug="no"
mmap="no"
rdma="no"
cuda="no"
clean="no"

for arg in "$@"
do
    if [ "$arg" == "debug" ]; then
        debug="yes"
    fi
    if [ "$arg" == "mmap" ]; then
        mmap="yes"
    fi
    if [ "$arg" == "rdma" ]; then
        rdma="yes"
    fi
    if [ "$arg" == "cuda" ]; then
        cuda="yes"
    fi
    if [ "$arg" == "clean" ]; then
        clean="yes"
    fi
done

if [ "$clean" == "yes" ]; then
    echo "Cleaning Takyon lib and examples..."
    cleanFolder ../lib
    cleanFolder ../examples/hello-one_sided
    cleanFolder ../hello-two_sided
    cleanFolder ../throughput
    exit 0
fi

echo "debug = $debug"
echo "mmap  = $mmap"
echo "rdma  = $rdma"
echo "cuda  = $cuda"

# Set the make options
options=""
if [ "$debug" == "yes" ]; then
    options+=" DEBUG=Yes"
fi
if [ "$cuda" == "yes" ]; then
    options+=" CUDA=Yes"
fi
echo "extra compile flags = $options"

# Takyon library
cleanFolder ../lib
command="make -j4 $options InterThread=Yes SocketTcp=Yes SocketUdp=Yes"
if [ "$mmap" == "yes" ]; then
    command+=" InterProcess=Yes"
fi
if [ "$rdma" == "yes" ]; then
    command+=" RdmaUDMulticast=Yes RdmaUC=Yes"
fi
build "$command"

# hello-one_sided
cleanFolder ../examples/hello-one_sided
command="make -j4 $options"
if [ "$mmap" == "yes" ]; then
    command+=" MMAP=Yes"
fi
if [ "$rdma" == "yes" ]; then
    command+=" RDMA=Yes"
fi
build "$command"

# hello-two_sided
cleanFolder ../hello-two_sided
command="make -j4 $options"
if [ "$mmap" == "yes" ]; then
    command+=" MMAP=Yes"
fi
if [ "$rdma" == "yes" ]; then
    command+=" RDMA=Yes"
fi
build "$command"

# throughput
cleanFolder ../throughput
command="make -j4 $options"
if [ "$mmap" == "yes" ]; then
    command+=" MMAP=Yes"
fi
if [ "$rdma" == "yes" ]; then
    command+=" RDMA=Yes"
fi
build "$command"
