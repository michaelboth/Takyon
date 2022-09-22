#!/bin/bash

function toFolder {
    echo ""
    echo "cd $1"
    cd $1
    if [ $? -ne 0 ]; then
        echo "Failed to go to $1 folder"
        exit 1
    fi
}

toFolder ../examples/hello-two_sided
echo ""
./hello_mt "InterThread -pathID=1" 0
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThread -pathID=1" 0
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThread -pathID=1" 1
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThread -pathID=1" 10
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

toFolder ../hello-one_sided
echo ""
./hello_mt "InterThread -pathID=1" 0
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThread -pathID=1" 1
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThread -pathID=1" 10
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

toFolder ../throughput
echo ""
./throughput_mt "InterThread -pathID=1" -n=100000 -b=32768 -v
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThread -pathID=1" -b=32768
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThread -pathID=1" -n=100000 -b=32768 -v -o
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThread -pathID=1" -b=32768 -o
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

exit 0
