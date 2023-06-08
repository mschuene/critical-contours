#!/bin/bash

cd "$(dirname "$0")"
jobid=$(qsub -terse -N distancesl5L1e3N1000 -q `cat /0/queue_long.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-100 <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
python <<EOF
import pickle
import os
import sh
sh.cd("/home/maik/master/src")
print(os.environ)
task_id = int(os.environ.get('SGE_TASK_ID'))

from ehe import *
from plfit_cd import *
calc_dists(task_id,sep=10)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessdistancesl5L1e3N1000
 -q `cat /0/queue_long.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
