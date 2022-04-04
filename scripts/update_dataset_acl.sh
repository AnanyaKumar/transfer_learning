#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --partition=jag-standard
#SBATCH --requeue

if [[ -d "/scr/biggest/$1" ]]; then
    setfacl -Rm u:eix:rwx,u:ananya:rwx,u:kshen6:rwx "/scr/biggest/$1"
fi

