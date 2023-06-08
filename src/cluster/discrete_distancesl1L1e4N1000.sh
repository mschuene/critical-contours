#!/bin/bash
cd "$(dirname "$0")"
mkdir discrete_distancesl1L1e4N1000
cd discrete_distancesl1L1e4N1000
jobid=$(qsub -terse -N discrete_distancesl1L1e4N1000 -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-100 <<EOT
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
outdir = "../avalanches/discrete_distancesl1L1e4N1000"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
task_id = int(os.environ.get('SGE_TASK_ID'))
from ehe import * 
from utils import * 
from plfit_cd import * 

from ehe import *
from plfit_cd import *
calc_dists(1,10000,3/2,task_id,outdir,sep=10,null_ccdf=discrete_S_pl,sample_func=draw_sample_discrete)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessdiscrete_distancesl1L1e4N1000 -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
# common setup code
import pickle
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/discrete_distancesl1L1e4N1000"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
from ehe import * 
from utils import * 
from plfit_cd import * 
from utils import *;post_command_concat(outdir+'/dists',range(1,101))
EOF
EOT

qstat
