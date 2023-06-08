#!/bin/bash
cd "$(dirname "$0")"
mkdir ehe_mat_crit
cd ehe_mat_crit
jobid=$(qsub -terse -N ehe_mat_crit -q maximus@fatbastard -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
python <<EOF
# common setup code
import pickle 
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/ehe_mat_crit"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
task_id = 1

from ehe import *;
import numpy as np;
ehe_arma = load_module('ehe_arma')
N = 10000;
avs,avd = ehe_arma.simulate_model_mat(np.random.random(N),1000*N,ehe_critical_weights(N),deltaU)
np.save(outdir+"/avs"+str(task_id),avs)
np.save(outdir+"/avd"+str(task_id),avd)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessehe_mat_crit -q maximus@fatbastard -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
