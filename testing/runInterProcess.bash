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

#+if [[ "$#" -ne "0" && "$#" -ne "1" ]]; then
if [ "$#" -ne "1" ]; then
    echo "USAGE: runInterProcess.bash <A|B>"
    exit 0
fi

if [[ "$1" != "A" && "$1" != "B" ]]; then
    echo "USAGE: runInterProcess.bash <A|B>"
    exit 0
fi

endpoint="$1"

toFolder ../examples/hello-two_sided
runExample ./hello_mp $endpoint "InterProcess -pathID=1" 0
runExample ./hello_mp $endpoint "InterProcess -pathID=1" 1
runExample ./hello_mp $endpoint "InterProcess -pathID=1" 10

runExample ./hello_mp $endpoint "TcpSocket -local -pathID=1" 0
runExample ./hello_mp $endpoint "TcpSocket -local -pathID=1" 10

if [ "$endpoint" == "A" ]; then
    runExample ./hello_mp $endpoint "TcpSocket -client -remoteIP=127.0.0.1 -port=23456" 0
    runExample ./hello_mp $endpoint "TcpSocket -client -remoteIP=127.0.0.1 -port=23456" 10
else
    runExample ./hello_mp $endpoint "TcpSocket -server -localIP=127.0.0.1 -port=23456 -reuse" 0
    runExample ./hello_mp $endpoint "TcpSocket -server -localIP=127.0.0.1 -port=23456 -reuse" 10
fi

if [ "$endpoint" == "A" ]; then
    runExample ./hello_mp $endpoint "TcpSocket -client -remoteIP=127.0.0.1 -ephemeralID=1" 0
    runExample ./hello_mp $endpoint "TcpSocket -client -remoteIP=127.0.0.1 -ephemeralID=1" 10
else
    runExample ./hello_mp $endpoint "TcpSocket -server -localIP=127.0.0.1 -ephemeralID=1" 0
    runExample ./hello_mp $endpoint "TcpSocket -server -localIP=127.0.0.1 -ephemeralID=1" 10
fi

if [ "$endpoint" == "A" ]; then
    sleep 1
    runExample ./hello_mp $endpoint "UdpSocketSend -unicast -remoteIP=127.0.0.1 -port=23456" 10
else
    runExample ./hello_mp $endpoint "UdpSocketRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse" 10
fi

if [ "$endpoint" == "A" ]; then
    sleep 1
    runExample ./hello_mp $endpoint "UdpSocketSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456" 10
else
    runExample ./hello_mp $endpoint "UdpSocketRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456 -reuse" 10
fi

toFolder ../hello-one_sided
runExample ./hello_mp $endpoint "InterProcess -pathID=1" 0
runExample ./hello_mp $endpoint "InterProcess -pathID=1" 1
runExample ./hello_mp $endpoint "InterProcess -pathID=1" 10

exit 0
