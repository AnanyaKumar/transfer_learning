#!/bin/bash

echo $1

if [ "$1" = "domainnet" ]; then
    echo "Extracting $1"
    cp -r /mnt/external/domainnet .
    echo "Copied file"
    ls
    cd domainnet
    ls
    unzip -q '*.zip'
    ls
    cd ..
elif [ "$1" = "imagenet" ]; then
    echo "Extracting $1"

else
    echo "Dataset '$1' not supported"
fi
