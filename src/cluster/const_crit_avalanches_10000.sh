#!/bin/bash
cd "$(dirname "$0")"
mkdir const_crit_avalanches_10000
cd const_crit_avalanches_10000
jobid=$(qsub -terse -N const_crit_avalanches_10000 -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-200 <<EOT
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
outdir = "../avalanches/const_crit_avalanches_10000"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
task_id = int(os.environ.get('SGE_TASK_ID'))
from ehe import * 
from utils import * 
from plfit_cd import * 

from ehe import *;
ehe_arma = load_module('ehe_arma');
N = 10000
avs,avd = ehe_arma.simulate_model_const(np.random.random(N),int(1e7),0.99/N,deltaU)
np.save(outdir+"/avs"+str(task_id),avs)
np.save(outdir+"/avd"+str(task_id),avd)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessconst_crit_avalanches_10000 -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
# common setup code
import pickle
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/const_crit_avalanches_10000"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
from ehe import * 
from utils import * 
from plfit_cd import * 
post_command_concat(outdir+'/avs',range(1,201))
EOF
EOT

qstat
