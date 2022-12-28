#!/bin/bash

cp -r /mnt/default/imagenet .
echo "Copied file"
ls
cd imagenet
ls
for file in *.tar.gz
do
    tar -xzf $file
done
ls
cd ..
