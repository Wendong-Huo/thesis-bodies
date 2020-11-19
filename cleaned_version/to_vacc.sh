#!/bin/sh
set -x
rsync -r --exclude-from=../.gitignore . sliu1@vacc-user1.uvm.edu:~/gpfs2/thesis-bodies/cleaned_version/