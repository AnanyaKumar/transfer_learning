#!/usr/bin/env bash
#SBATCH --partition=john
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

print_usage () {
    USAGE_STRING="Usage: send_email.sh [-r|--recipient RECIPIENT]"
    USAGE_STRING="$USAGE_STRING subject {text,command} text_or_command"
    echo $USAGE_STRING
}

recipient=$(whoami)@cs.stanford.edu
if [ "$1" = "-r" -o "$1" = "--recipient" ]; then
    recipient=$2
    shift 2
fi

if [ $# -ne 3 ]; then
    print_usage
    exit 1
fi

subject=$1
email_generation=$2
if [ "$email_generation" = "text" ]; then
    email_body=$3
elif [ "$email_generation" = "command" ]; then
    email_body=$(eval "$3")
fi

{
    echo "To: $recipient"
    echo "Subject: $subject"
    echo
    echo "$email_body"
} | sendmail -t
