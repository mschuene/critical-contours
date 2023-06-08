#!/bin/bash
cd "$(dirname "$0")"
mkdir test_conjecture
cd test_conjecture
jobid=$(qsub -terse -N test_conjecture -q `cat /0/queue_maximus.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 7-8 <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
export OMP_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen
python <<EOF
# common setup code
import pickle 
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/test_conjecture"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ,flush=True)
task_id = int(os.environ.get('SGE_TASK_ID'))
from ehe import * 
from ehe_ana import * 
from utils import * 
from plfit_cd import * 
from ehe_subnetworks import *
test_conjecture(task_id)
EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocesstest_conjecture -q `cat /0/queue_maximus.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
