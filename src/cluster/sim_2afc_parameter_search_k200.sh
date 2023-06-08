#!/bin/bash
cd "$(dirname "$0")"
mkdir sim_2afc_parameter_search_k200
cd sim_2afc_parameter_search_k200
jobid=$(qsub -terse -N sim_2afc_parameter_search_k200 -q `cat /0/maik/queue_maximus.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-160 <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
export OMP_NUM_THREADS=1
export QT_QPA_PLATFORM=offscreen
python <<EOF
# common setup code
import matplotlib
matplotlib.use('Agg')
import pickle 
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/sim_2afc_parameter_search_k200"
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

import itertools
import pickle
parameters =  [(Ns,Ne,200) for Ns in range(80,120,4) for Ne in range(40,71,2)]
Ns,Ne,K = parameters[task_id - 1]
sim_res = sim_2afc(N_s=Ns,N_e=Ne,K=K,beta=2,N_u=1000,num_ens=10,num_act=10)
pickle.dump(sim_res,open(outdir+"/sim_res"+str(Ns)+","+str(Ne)+","+str(K)+".pickle",'wb'))

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocesssim_2afc_parameter_search_k200 -q `cat /0/queue_maximus.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
