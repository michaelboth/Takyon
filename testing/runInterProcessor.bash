#!/bin/bash

function toFolder {
    cd $1
    if [ $? -ne 0 ]; then
        echo "Failed to go to $1 folder"
        exit 1
    fi
}

# Determine if endpoint A or B
if [ "$#" -lt "3" ]; then
    echo "USAGE: runInterProcess.bash <A|B> <local_ip> <remote_ip> [socket] [ephemeral] [rdma]"
    exit 0
fi

if [[ "$1" != "A" && "$1" != "B" ]]; then
    echo "USAGE: runInterProcess.bash <A|B> <local_ip> <remote_ip> [socket] [ephemeral] [rdma]"
    exit 0
fi

endpoint="$1"
local_ip="$2"
remote_ip="$3"

# Get optional args
rdma="no"
socket="no"
ephemeral="no"
for arg in "$@"
do
    if [ "$arg" == "rdma" ]; then
        rdma="yes"
    fi
    if [ "$arg" == "socket" ]; then
        socket="yes"
    fi
    if [ "$arg" == "ephemeral" ]; then
        ephemeral="yes"
    fi
done
echo "rdma  = $rdma"
echo "socket  = $socket"
echo "ephemeral  = $ephemeral"
echo "local_ip  = $local_ip"
echo "remote_ip  = $remote_ip"

# hello-two_sided
toFolder ../examples/hello-two_sided
if [ "$socket" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23456" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23456" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23456 -reuse" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23456 -reuse" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
    if [ "$ephemeral" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            echo ""
            ./hello_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=1" 0
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./hello_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=1" 10
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./hello_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=1" 0
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./hello_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=1" 10
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi
    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./hello_mp $endpoint "SocketUdpSend -unicast -remoteIP=$remote_ip -port=23456" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "SocketUdpRecv -unicast -localIP=$local_ip -port=23456 -reuse" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./hello_mp $endpoint "SocketUdpSend -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23456" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "SocketUdpRecv -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23456 -reuse" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
fi

if [ "$rdma" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 50
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 25
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 50
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 25
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./hello_mp $endpoint "RdmaUDMulticastSend -localIP=$local_ip -groupIP=233.23.33.56" 50
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "RdmaUDMulticastRecv -localIP=$local_ip -groupIP=233.23.33.56" 25
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
fi

# hello-one_sided
toFolder ../hello-one_sided
if [ "$rdma" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
fi

# throughput
toFolder ../throughput
if [ "$socket" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23456" -n=10000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23457" -n=10000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23457" -n=10000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23456 -reuse" -n=10000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23457 -reuse" -n=10000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23457 -reuse" -n=10000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$ephemeral" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=1" -n=10000 -b=1024 -v
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=1" -n=10000 -b=1024
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=1" -n=10000 -b=1024 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=1" -n=10000 -b=1024 -v
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=1" -n=10000 -b=1024
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=1" -n=10000 -b=1024 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi

    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=$remote_ip -port=23456" -n=10000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=$remote_ip -port=23456" -n=10000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=$remote_ip -port=23456" -n=10000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=$local_ip -port=23456 -reuse" -n=10000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=$local_ip -port=23456 -reuse" -n=10000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=$local_ip -port=23456 -reuse" -n=10000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23457" -n=10000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23457" -n=10000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23457" -n=10000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23457 -reuse" -n=10000 -b=1024 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23457 -reuse" -n=10000 -b=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23457 -reuse" -n=10000 -b=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
fi

if [ "$rdma" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -s=10 -r=100 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -s=10 -r=100
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -s=10 -r=100 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -r=100 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -r=100
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -r=100 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -s=10 -r=100 -v -o
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -s=10 -r=100 -o
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -s=10 -r=100 -e -o
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -r=100 -v -o
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -r=100 -o
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -r=100 -e -o
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -s=10 -r=100 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -s=10 -r=100
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -s=10 -r=100 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -r=100 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -r=100
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=100000 -b=40960 -r=100 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=4096 -s=100 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=4096 -s=100
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=4096 -s=100 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=4136 -r=1000 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=4136 -r=1000
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=4136 -r=1000 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./throughput_mp $endpoint "RdmaUDMulticastSend -localIP=$local_ip -groupIP=233.23.33.56" -n=1000000 -b=4096 -s=100 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "RdmaUDMulticastSend -localIP=$local_ip -groupIP=233.23.33.56" -n=1000000 -b=4096 -s=100
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "RdmaUDMulticastSend -localIP=$local_ip -groupIP=233.23.33.56" -n=1000000 -b=4096 -s=100 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "RdmaUDMulticastRecv -localIP=$local_ip -groupIP=233.23.33.56" -n=1000000 -b=4136 -r=1000 -v
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDMulticastRecv -localIP=$local_ip -groupIP=233.23.33.56" -n=1000000 -b=4136 -r=1000
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDMulticastRecv -localIP=$local_ip -groupIP=233.23.33.56" -n=1000000 -b=4136 -r=1000 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
fi

exit 0
