#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

# pwd prints the current directory
echo $PYTHONPATH

echo 'The current directory is:'
pwd

# ls -l gives a list of everything of files and subdirectories
echo 'and in this directory you can find:'
ls -l