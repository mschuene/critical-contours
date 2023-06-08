#!/bin/bash
cd "$(dirname "$0")"
mkdir distancesl5L1e3N1000_pp
cd distancesl5L1e3N1000_pp
jobid=$(qsub -terse -N distancesl5L1e3N1000_pp -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-100 <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
export OMP_NUM_THREADS=1
python <<EOF
# common setup code
import pickle 
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/distancesl5L1e3N1000_pp"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
task_id = int(os.environ.get('SGE_TASK_ID'))

from ehe import *
from plfit_cd import *
calc_dists(1,10000,3/2,task_id,outdir,sep=10,null_ccdf=discrete_S_pl,sample_func=draw_sample_discrete)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessdistancesl5L1e3N1000_pp -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
# common setup code
import pickle
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/distancesl5L1e3N1000_pp"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
from utils import *;post_command_concat(outdir+'/dists',range(1,201))
EOF
EOT

qstat
