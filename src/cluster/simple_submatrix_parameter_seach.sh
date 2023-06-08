#!/bin/bash
cd "$(dirname "$0")"
mkdir simple_submatrix_parameter_seach
cd simple_submatrix_parameter_seach
jobid=$(qsub -terse -N simple_submatrix_parameter_seach -q `cat /0/maik/queue_maximus+long.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-42 <<EOT
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
outdir = "../avalanches/simple_submatrix_parameter_seach"
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
parameters = ([(100,int(Ne),0) for Ne in np.linspace(50,70,21)] +
              [(int(Ns),50,0)  for Ns in np.linspace(100,120,21)])
Ns,Ne,K = parameters[task_id - 1]
sim_res = run_simulation(Ns,Ne,K,M_ens=20,M_act=20)
sim_res['parameters'] = (Ns,Ne,K)
pickle.dump(sim_res,open(outdir+"/sim_res"+str(Ns)+","+str(Ne)+","+str(K)+".pickle",'wb'))
sim_res['f'].savefig(outdir+"/plot"+str(Ns)+","+str(Ne)+","+str(K)+".png")

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocesssimple_submatrix_parameter_seach -q maximus+long -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
