#!/bin/bash

cp -r /mnt/default/domainnet .
echo "Copied file"
ls
cd domainnet
ls
unzip -q '*.zip'
ls
cd ..
