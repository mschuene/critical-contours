#!/bin/bash

cd "$(dirname "$0")"
qsub -N testmultnotify -q `cat /0/queue_maximus.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m eas `#end abort suspend` -t 1-2 <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
python <<EOF
import pickle
import os
import sh
sh.cd("/home/maik/master/src")
print(os.environ)
task_id = int(os.environ.get('SGE_TASK_ID'))
print("hello "+str(task_id))
EOF

EOT
