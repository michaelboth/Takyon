#!/bin/bash

function toFolder {
    cd $1
    if [ $? -ne 0 ]; then
        echo "Failed to go to $1 folder"
        exit 1
    fi
}

function runExample {
    echo ""
    echo "$1 $2 \"$3\"" $4
    $1 $2 "$3" $4
    if [ $? -ne 0 ]; then
        echo "Failed to run example"
        exit 1
    fi
    #sleep 1
}

# Determine if endpoint A or B
if [ "$#" -lt "1" ]; then
    echo "USAGE: runInterProcess.bash <A|B> [mmap] [socket]"
    exit 0
fi

if [[ "$1" != "A" && "$1" != "B" ]]; then
    echo "USAGE: runInterProcess.bash <A|B> [mmap] [socket]"
    exit 0
fi

endpoint="$1"

# Get optional args
mmap="no"
socket="no"
for arg in "$@"
do
    if [ "$arg" == "mmap" ]; then
        mmap="yes"
    fi
    if [ "$arg" == "socket" ]; then
        socket="yes"
    fi
done
echo "mmap  = $mmap"
echo "socket  = $socket"

toFolder ../examples/hello-two_sided
if [ "$mmap" == "yes" ]; then
    runExample ./hello_mp $endpoint "InterProcess -pathID=1" 0
    runExample ./hello_mp $endpoint "InterProcess -pathID=2" 1
    runExample ./hello_mp $endpoint "InterProcess -pathID=3" 10
fi

if [ "$socket" == "yes" ]; then
    runExample ./hello_mp $endpoint "SocketTcp -local -pathID=1" 0
    runExample ./hello_mp $endpoint "SocketTcp -local -pathID=2" 10

    if [ "$endpoint" == "A" ]; then
        runExample ./hello_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -port=23456" 0
        runExample ./hello_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -port=23456" 10
    else
        runExample ./hello_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse" 0
        runExample ./hello_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse" 10
    fi

    if [ "$endpoint" == "A" ]; then
        runExample ./hello_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" 0
        runExample ./hello_mp $endpoint "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" 10
    else
        runExample ./hello_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" 0
        runExample ./hello_mp $endpoint "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" 10
    fi

    if [ "$endpoint" == "A" ]; then
        sleep 1
        runExample ./hello_mp $endpoint "SocketUdpSend -unicast -remoteIP=127.0.0.1 -port=23456" 10
    else
        runExample ./hello_mp $endpoint "SocketUdpRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse" 10
    fi

    if [ "$endpoint" == "A" ]; then
        sleep 1
        runExample ./hello_mp $endpoint "SocketUdpSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456" 10
    else
        runExample ./hello_mp $endpoint "SocketUdpRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456 -reuse" 10
    fi
fi

toFolder ../hello-one_sided
if [ "$mmap" == "yes" ]; then
    runExample ./hello_mp $endpoint "InterProcess -pathID=1" 0
    runExample ./hello_mp $endpoint "InterProcess -pathID=2" 1
    runExample ./hello_mp $endpoint "InterProcess -pathID=3" 10
fi

exit 0
