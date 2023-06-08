#!/bin/bash

cd "$(dirname "$0")"
qsub -N test -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m eas `#end abort suspend` <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
echo $HOSTNAME
echo $PATH
echo $LD_LIBRARY_PATH
echo $LIBRARY_PATH

python <<EOF
import pickle
import os
print(os.environ)
task_id = 1
print('hello world from cmd')
EOF

EOT
