#!/usr/bin/env sh

print_usage () {
    USAGE_STRING="Usage: send_email_sbatch.sh [-h|--help]"
    USAGE_STRING="$USAGE_STRING [--slurm_dependency JOBID]"
    USAGE_STRING="$USAGE_STRING [-r|--recipient EMAIL]"
    USAGE_STRING="$USAGE_STRING subject {text,command} text_or_command"
    echo $USAGE_STRING
}

set -e
recipient=$(whoami)@cs.stanford.edu
while true; do
    case "$1" in
	-h|--help) # Print help
	    print_usage
	    exit
	    ;;
	--slurm_dependency) # Slurm job to wait for
	    if [ "$2" ]; then
		slurm_dependency=$2
		shift
	    else
		echo '--slurm_dependency must be non-empty!'
		exit 1
	    fi
	    ;;
	-r|--recipient) # Email of recipient
	    if [ "$2" ]; then
		recipient=$2
		shift
	    else
		echo '-r|--recipient must be non-empty!'
		exit 1
	    fi
	    ;;
	-*) # Other option that's not supported
	    echo "Unsupported option $1"
	    exit 1
	    ;;
	*)
	    break
	    ;;
    esac
    shift
done

if [ $# -ne 3 ]; then
    print_usage
    exit 1
fi

subject=$1
email_generation=$2
email_text_or_command=$3

cmd="sbatch"
if [ "$slurm_dependency" ]; then
    cmd="$cmd --dependency=afterok:$slurm_dependency --kill-on-invalid-dep=yes"
fi
cmd="$cmd send_email.sh -r \"$recipient\" \"$subject\""
cmd="$cmd \"$email_generation\" \"$email_text_or_command\""
eval "$cmd"
