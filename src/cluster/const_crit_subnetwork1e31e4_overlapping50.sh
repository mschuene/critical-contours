#!/bin/bash
cd "$(dirname "$0")"
mkdir const_crit_subnetwork1e31e4_overlapping50
cd const_crit_subnetwork1e31e4_overlapping50
jobid=$(qsub -terse -N const_crit_subnetwork1e31e4_overlapping50 -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-10 <<EOT
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
outdir = "../avalanches/const_crit_subnetwork1e31e4_overlapping50"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
task_id = int(os.environ.get('SGE_TASK_ID'))
from ehe import * 
from utils import * 
from plfit_cd import * 
from ehe_subnetworks import *

ehe = load_module('ehe_detailed')
N_sub = 1000;
N_whole = 10*N_sub;
inds_sub = list(range(N_sub))
inds_sub2 = list(range(int(0.5*N_sub),int(1.5*N_sub)))
W = const_critical_subnetwork_weights(N_whole,inds_sub)
W[W==0] += const_critical_subnetwork_weights(N_whole,inds_sub2)[W==0]
print(W)
avs,avs_inds = ehe.simulate_model_mat(np.random.random(N_whole),10*N_sub,W,deltaU)
np.save(outdir+"/avs"+str(task_id),avs)
np.save(outdir+"/avs_inds"+str(task_id),avs_inds)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessconst_crit_subnetwork1e31e4_overlapping50 -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
# common setup code
import pickle
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/const_crit_subnetwork1e31e4_overlapping50"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
from ehe import * 
from utils import * 
from plfit_cd import * 
from ehe_subnetworks import *
post_command_concat_detailed(outdir+'/avs',outdir+'/avs_inds',range(1,11))
EOF
EOT

qstat
