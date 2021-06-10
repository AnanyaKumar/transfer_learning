#!/bin/bash

set -ex

# Usernames to give read/write permissions to
USERNAMES="ananya eix"
USERNAMES+=" kshen6"
USERNAMES+=" rmjones"

show_help() {
    usage_string="Usage: copy_dataset.sh"
    usage_string+=" dataset_name"
    usage_string+=" [-s|--src DATASET_SOURCE]"
    usage_string+=" [-d|--dst DESTINATION_FOLDER]"

    usage_string+="\n\nOptions:\n"
    usage_string+="\t-s|--src Path to dataset to copy\n"
    usage_string+="\t-d|--dst Path to destination folder\n"
    printf "$usage_string"
}

if [[ "$1" = "-h" || "$1" = "--help" ]]; then
    show_help
    exit
fi

if [[ $# -lt 1 ]]; then
    show_help
    exit 1
fi

dataset_name=$1


if [[ "$dataset_name" != imagenet && "$dataset_name" != domainnet ]]; then
    echo "Unsupported dataset: $dataset_name"
    exit 1
fi
shift

if [ "$dataset_name" = imagenet ]; then
    dataset_src=/u/scr/nlp/eix/imagenet
elif [ "$dataset_name" = domainnet ]; then
    dataset_src=/u/scr/nlp/domainnet/domainnet.zip
else
    echo "Unsupported dataset $dataset_name"
    exit 1
fi
dst_folder=/scr/scr-with-most-space/$dataset_name

while true; do
    case $1 in
	-h|--help)
	    show_help
	    exit
	    ;;
	-s|--src)
	    if [[ "$2" ]]; then
		dataset_src=$2
		shift
	    else
		echo "-s|--src must be non-empty!"; exit 1
	    fi
	    ;;
	-d|--dst)
	    if [[ "$2" ]]; then
		dst_folder=$2
		shift
	    else
		echo "-d|--dst must be non-empty!"; exit 1
	    fi
	    ;;
	*)
	    if [ -z "$1" ]; then
		break
	    else
		echo "Unsupported argument: $1"; exit 1
	    fi
	    ;;
    esac
    shift
done

if [ "$dataset_name" = imagenet ]; then
    for folder in train val; do
	if [ ! -f "$dataset_src/$folder.tar.gz" ]; then
	    echo "Missing ImageNet file $folder.tar.gz!"
	    exit 1
	fi
    done
elif [ "$dataset_name" = domainnet ]; then
    if [ ! "${dataset_src##*.}" = zip ]; then
	echo "Expecting .zip file for DomainNet dataset"
	exit 1
    fi
fi

ACL_STRING=""
for username in $(echo "$USERNAMES"); do
    if [[ $username = $(whoami) ]]; then
	continue
    fi
    ACL_STRING+="u:$username:rw,"
done
ACL_STRING=${ACL_STRING: : -1} # Remove trailing comma

if [ ! -d "$dst_folder" ]; then
    mkdir -p $dst_folder
    setfacl -d -m $ACL_STRING $dst_folder # Set default permissions for folder
fi

if [ "$dataset_name" = imagenet ]; then
    for folder in train val; do
	if [ ! -f "$dst_folder/$folder.tar.gz" ]; then
	    echo "Copying $dataset_src/$folder.tar.gz to $dst_folder..."
	    cp $dataset_src/$folder.tar.gz $dst_folder
	fi
	if [ ! -d "$dst_folder/$folder" ]; then
	    echo "Extracting $dst_folder/$folder.tar.gz..."
	    tar xzf $dst_folder/$folder.tar.gz -C $dst_folder
	fi
    done
elif [ "$dataset_name" = domainnet ]; then
  echo "Copying DomainNet files to $dst_folder"
  cp $dataset_src $dst_folder
  unzip -q $dst_folder/domainnet.zip -d $dst_folder
fi
