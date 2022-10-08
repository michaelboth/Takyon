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
    echo "USAGE: runInterProcess.bash <A|B> <local_ip> <remote_ip> [socket] [ephemeral] [multicast] [rdma]"
    exit 0
fi

if [[ "$1" != "A" && "$1" != "B" ]]; then
    echo "USAGE: runInterProcess.bash <A|B> <local_ip> <remote_ip> [socket] [ephemeral] [multicast] [rdma]"
    exit 0
fi

endpoint="$1"
local_ip="$2"
remote_ip="$3"

# Get optional args
rdma="no"
socket="no"
ephemeral="no"
multicast="no"
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
    if [ "$arg" == "multicast" ]; then
        multicast="yes"
    fi
done
echo "rdma      = $rdma"
echo "socket    = $socket"
echo "ephemeral = $ephemeral"
echo "multicast = $multicast"
echo "local_ip  = $local_ip"
echo "remote_ip = $remote_ip"

# hello-two_sided
toFolder ../examples/hello-two_sided
if [ "$socket" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23456" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23457" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23456 -reuse" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23457 -reuse" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
    if [ "$ephemeral" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            echo ""
            ./hello_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=1" 0
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./hello_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=2" 10
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./hello_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=1" 0
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./hello_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=2" 10
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi
    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./hello_mp $endpoint "SocketUdpSend -unicast -remoteIP=$remote_ip -port=23458" 50
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "SocketUdpRecv -unicast -localIP=$local_ip -port=23458 -reuse" 50
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
    if [ "$multicast" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            sleep 1
            echo ""
            ./hello_mp $endpoint "SocketUdpSend -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23459" 50
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./hello_mp $endpoint "SocketUdpRecv -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23459 -reuse" 50
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi
fi

if [ "$rdma" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23457 -rdmaDevice=mlx5_0 -rdmaPort=1" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23457 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23459 -rdmaDevice=mlx5_0 -rdmaPort=1" 50
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23459 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" 50
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23457 -rdmaDevice=mlx5_0 -rdmaPort=1" 50
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23457 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" 50
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$multicast" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            sleep 1
            echo ""
            ./hello_mp $endpoint "RdmaUDMulticastSend -localIP=$local_ip -groupIP=233.23.33.57" 50
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./hello_mp $endpoint "RdmaUDMulticastRecv -localIP=$local_ip -groupIP=233.23.33.57" 50
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi
fi

# hello-one_sided
toFolder ../hello-one_sided
if [ "$rdma" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./hello_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23459 -rdmaDevice=mlx5_0 -rdmaPort=1" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./hello_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" 0
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./hello_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23459 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" 10
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi
fi

# throughput
toFolder ../throughput
if [ "$socket" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23456" -i=10000 -bytes=1024 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23457" -i=10000 -bytes=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23458" -i=10000 -bytes=1024 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -port=23458" -i=10000 -bytes=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23456 -reuse" -i=10000 -bytes=1024 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23457 -reuse" -i=10000 -bytes=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23458 -reuse" -i=10000 -bytes=1024 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -port=23458 -reuse" -i=10000 -bytes=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$ephemeral" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=1" -i=10000 -bytes=1024 -V
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=2" -i=10000 -bytes=1024
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=3" -i=10000 -bytes=1024 -V -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -client -remoteIP=$remote_ip -ephemeralID=3" -i=10000 -bytes=1024 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=1" -i=10000 -bytes=1024 -V
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=2" -i=10000 -bytes=1024
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=3" -i=10000 -bytes=1024 -V -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketTcp -server -localIP=$local_ip -ephemeralID=3" -i=10000 -bytes=1024 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi

    if [ "$endpoint" == "A" ]; then
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=$remote_ip -port=23456" -i=10000 -bytes=1024 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=$remote_ip -port=23457" -i=10000 -bytes=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        sleep 1
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=$remote_ip -port=23458" -i=10000 -bytes=1024 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpSend -unicast -remoteIP=$remote_ip -port=23458" -i=10000 -bytes=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=$local_ip -port=23456 -reuse" -i=10000 -bytes=1024 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=$local_ip -port=23457 -reuse" -i=10000 -bytes=1024
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=$local_ip -port=23458 -reuse" -i=10000 -bytes=1024 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "SocketUdpRecv -unicast -localIP=$local_ip -port=23458 -reuse" -i=10000 -bytes=1024 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$multicast" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            sleep 1
            echo ""
            ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23456" -i=10000 -bytes=1024 -V
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            sleep 1
            echo ""
            ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=$local_ip -groupIP=233.23.33.57 -port=23457" -i=10000 -bytes=1024
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            sleep 1
            echo ""
            ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=$local_ip -groupIP=233.23.33.58 -port=23458" -i=10000 -bytes=1024 -V -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            sleep 1
            echo ""
            ./throughput_mp $endpoint "SocketUdpSend -multicast -localIP=$local_ip -groupIP=233.23.33.58 -port=23458" -i=10000 -bytes=1024 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=$local_ip -groupIP=233.23.33.56 -port=23456 -reuse" -i=10000 -bytes=1024 -V
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=$local_ip -groupIP=233.23.33.57 -port=23457 -reuse" -i=10000 -bytes=1024
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=$local_ip -groupIP=233.23.33.58 -port=23458 -reuse" -i=10000 -bytes=1024 -V -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "SocketUdpRecv -multicast -localIP=$local_ip -groupIP=233.23.33.58 -port=23458 -reuse" -i=10000 -bytes=1024 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi
fi

if [ "$rdma" == "yes" ]; then
    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23457 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23457 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -V -write
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23457 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -write
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -V -e -write
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -e -write
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -V -write
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23457 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -write
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -V -e -write
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaRC -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -e -write
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        #+ The following two test are failing because the receiver can't keep up will the sender, and the receiver times out cause it's not getting any messages.
        #+ echo ""
        #+ ./throughput_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23457 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100
        #+ if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        #+ echo ""
        #+ ./throughput_mp $endpoint "RdmaUC -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -sbufs=10 -dbufs=100 -e
        #+ if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        #+ The following two test are failing because the receiver can't keep up will the sender, and the receiver times out cause it's not getting any messages.
        #+ echo ""
        #+ ./throughput_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23457 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100
        #+ if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        #+ echo ""
        #+ ./throughput_mp $endpoint "RdmaUC -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=100000 -bytes=40960 -dbufs=100 -e
        #+ if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$endpoint" == "A" ]; then
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=1024 -sbufs=100 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23457 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=1024 -sbufs=100
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=1024 -sbufs=100 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastSend -client -remoteIP=$remote_ip -port=23458 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=1024 -sbufs=100 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    else
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=1064 -dbufs=1000 -V
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23457 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=1064 -dbufs=1000
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=1064 -dbufs=1000 -V -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        echo ""
        ./throughput_mp $endpoint "RdmaUDUnicastRecv -server -localIP=$local_ip -port=23458 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=1064 -dbufs=1000 -e
        if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
    fi

    if [ "$multicast" == "yes" ]; then
        if [ "$endpoint" == "A" ]; then
            sleep 1
            echo ""
            ./throughput_mp $endpoint "RdmaUDMulticastSend -localIP=$local_ip -groupIP=233.23.33.56" -i=1000000 -bytes=1024 -sbufs=100 -V
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            sleep 1
            echo ""
            ./throughput_mp $endpoint "RdmaUDMulticastSend -localIP=$local_ip -groupIP=233.23.33.57" -i=1000000 -bytes=1024 -sbufs=100
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            sleep 1
            echo ""
            ./throughput_mp $endpoint "RdmaUDMulticastSend -localIP=$local_ip -groupIP=233.23.33.58" -i=1000000 -bytes=1024 -sbufs=100 -V -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "RdmaUDMulticastSend -localIP=$local_ip -groupIP=233.23.33.58" -i=1000000 -bytes=1024 -sbufs=100 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        else
            echo ""
            ./throughput_mp $endpoint "RdmaUDMulticastRecv -localIP=$local_ip -groupIP=233.23.33.56" -i=1000000 -bytes=1064 -dbufs=1000 -V
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "RdmaUDMulticastRecv -localIP=$local_ip -groupIP=233.23.33.57" -i=1000000 -bytes=1064 -dbufs=1000
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "RdmaUDMulticastRecv -localIP=$local_ip -groupIP=233.23.33.58" -i=1000000 -bytes=1064 -dbufs=1000 -V -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
            echo ""
            ./throughput_mp $endpoint "RdmaUDMulticastRecv -localIP=$local_ip -groupIP=233.23.33.58" -i=1000000 -bytes=1064 -dbufs=1000 -e
            if [ $? -ne 0 ]; then echo "Failed to run example"; exit 1; fi
        fi
    fi
fi

echo ""
echo "Testing complete"

exit 0
