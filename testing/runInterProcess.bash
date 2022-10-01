#!/bin/bash

function toFolder {
    cd $1
    if [ $? -ne 0 ]; then
        echo "Failed to go to $1 folder"
        exit 1
    fi
}

# Determine if endpoint A or B
if [ "$#" -lt "1" ]; then
    echo "USAGE: runInterProcess.bash <A|B> [mmap] [socket] [ephemeral]"
    exit 0
fi

if [[ "$1" != "A" && "$1" != "B" ]]; then
    echo "USAGE: runInterProcess.bash <A|B> [mmap] [socket] [ephemeral]"
    exit 0
fi

endpoint="$1"

# Get optional args
mmap="no"
socket="no"
ephemeral="no"
for arg in "$@"
do
    if [ "$arg" == "mmap" ]; then
        mmap="yes"
    fi
    if [ "$arg" == "socket" ]; then
        socket="yes"
    fi
    if [ "$arg" == "ephemeral" ]; then
        ephemeral="yes"
    fi
done
echo "mmap  = $mmap"
echo "socket  = $socket"
echo "ephemeral  = $ephemeral"

# hello-two_sided
toFolder ../examples/hello-two_sided
if [ "$mmap" == "yes" ]; then
    echo ""
    ./hello_mp $endpoint "InterProcess -pathID=1" 0
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./hello_mp $endpoint "InterProcess -pathID=2" 1
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./hello_mp $endpoint "InterProcess -pathID=3" 10
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./hello_mp $endpoint "InterProcessU -pathID=1" 0
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./hello_mp $endpoint "InterProcessU -pathID=2" 1
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./hello_mp $endpoint "InterProcessU -pathID=3" 10
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
fi
if [ "$socket" == "yes" ]; then
    echo ""
    ./hello_mp $endpoint "SocketTcp -local -pathID=1" 0
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./hello_mp $endpoint "SocketTcp -local -pathID=2" 10
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -port=23456" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -port=23456" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
    if [ "$ephemeral" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            echo ""
            ./hello_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" 0
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./hello_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" 10
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./hello_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" 0
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./hello_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" 10
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi
    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./hello_mp $endpoint "SocketUdpSend -unicast -remoteIP=127.0.0.1 -port=23456" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "SocketUdpRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./hello_mp $endpoint "SocketUdpSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "SocketUdpRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456 -reuse" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
fi

# hello-one_sided
toFolder ../hello-one_sided
if [ "$mmap" == "yes" ]; then
    echo ""
    ./hello_mp $endpoint "InterProcess -pathID=1" 0
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./hello_mp $endpoint "InterProcess -pathID=2" 1
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./hello_mp $endpoint "InterProcess -pathID=3" 10
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
fi

# throughput
toFolder ../throughput
if [ "$mmap" == "yes" ]; then
    echo ""
    ./throughput_mp $endpoint "InterProcess -pathID=1" -n=100000 -b=32768 -v
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "InterProcess -pathID=2" -n=100000 -b=32768
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "InterProcess -pathID=2" -n=100000 -b=32768 -e
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "InterProcess -pathID=1" -n=100000 -b=32768 -v -o
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "InterProcess -pathID=2" -n=100000 -b=32768 -o
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "InterProcess -pathID=2" -n=100000 -b=32768 -e -o
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "InterProcessU -pathID=1" -n=100000 -b=32768 -v
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "InterProcessU -pathID=2" -n=100000 -b=32768
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "InterProcessU -pathID=2" -n=100000 -b=32768 -e
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
fi

if [ "$socket" == "yes" ]; then
    echo ""
    ./throughput_mp $endpoint "SocketTcp -local -pathID=1" -n=100000 -b=1024 -v
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "SocketTcp -local -pathID=2" -n=100000 -b=1024
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    echo ""
    ./throughput_mp $endpoint "SocketTcp -local -pathID=2" -n=100000 -b=1024 -e
    if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -port=23456" -n=100000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -port=23457" -n=100000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -port=23457" -n=100000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse" -n=100000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -port=23457 -reuse" -n=100000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -port=23457 -reuse" -n=100000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$ephemeral" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" -n=100000 -b=1024 -v
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" -n=100000 -b=1024
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" -n=100000 -b=1024 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" -n=100000 -b=1024 -v
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" -n=100000 -b=1024
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" -n=100000 -b=1024 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi

    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=127.0.0.1 -port=23456" -n=100000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=127.0.0.1 -port=23456" -n=100000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=127.0.0.1 -port=23456" -n=100000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse" -n=100000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse" -n=100000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse" -n=100000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23457" -n=100000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.57 -port=23457" -n=100000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.58 -port=23457" -n=100000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23457 -reuse" -n=100000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.57 -port=23457 -reuse" -n=100000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.58 -port=23457 -reuse" -n=100000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
fi

exit 0
