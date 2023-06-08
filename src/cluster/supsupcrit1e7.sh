#!/bin/bash
cd "$(dirname "$0")"
mkdir supsupcrit1e7
cd supsupcrit1e7
jobid=$(qsub -terse -N supsupcrit1e7 -q `cat /0/queue_maximus.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
python <<EOF
# common setup code
import pickle 
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/supsupcrit1e7"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
task_id = 1

from ehe import *;
ehe_arma = load_module('ehe_arma');
N = 10000
res = ehe_arma.simulate_model_const(np.random.random(N),10000000,0.9999/N,deltaU)
pickle.dump(res,open('../avalanches/subsubcrit1e7.pickle','wb'))
EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocesssupsupcrit1e7 -q `cat /0/queue_maximus.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
