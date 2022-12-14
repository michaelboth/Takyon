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
./hello_mt "InterThreadRC -pathID=1" 0
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThreadRC -pathID=1" 1
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThreadRC -pathID=1" 10
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThreadUC -pathID=1" 0
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThreadUC -pathID=1" 1
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThreadUC -pathID=1" 10
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

toFolder ../hello-one_sided
echo ""
./hello_mt "InterThreadRC -pathID=1" 0
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThreadRC -pathID=1" 1
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./hello_mt "InterThreadRC -pathID=1" 10
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

toFolder ../throughput
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -i=100000 -V
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -i=100000 -V -write
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -write
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -i=100000 -V -read
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -read
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -i=100000 -V -e
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -e
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -i=100000 -V -write -e
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -write -e
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -i=100000 -V -read -e
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -read -e
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

echo ""
./throughput_mt "InterThreadUC -pathID=1" -bytes=32768 -i=100000 -V
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadUC -pathID=1" -bytes=32768
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

echo ""
./throughput_mt "InterThreadUC -pathID=1" -bytes=32768 -i=100000 -V -e
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
echo ""
./throughput_mt "InterThreadUC -pathID=1" -bytes=32768 -e
if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

exit 0
