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
    echo "$1 \"$2\" $3"
    $1 "$2" $3
    if [ $? -ne 0 ]; then
        echo "Failed to run example"
        exit 1
    fi
}

toFolder ../examples/hello-two_sided
runExample ./hello_mt "InterThread -pathID=1" 0
runExample ./hello_mt "InterThread -pathID=1" 1
runExample ./hello_mt "InterThread -pathID=1" 10

toFolder ../hello-one_sided
runExample ./hello_mt "InterThread -pathID=1" 0
runExample ./hello_mt "InterThread -pathID=1" 1
runExample ./hello_mt "InterThread -pathID=1" 10

exit 0
