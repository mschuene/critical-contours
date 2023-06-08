#!/bin/bash
cd "$(dirname "$0")"
mkdir simple_submatrix_1e4
cd simple_submatrix_1e4
jobid=$(qsub -terse -N simple_submatrix_1e4 -q `cat /0/maik/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -pe omp 4 -t 1-100 <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
export OMP_NUM_THREADS=4
export QT_QPA_PLATFORM=offscreen
python <<EOF
# common setup code
import pickle 
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/simple_submatrix_1e4"
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

(ens_id,act_id) = [(ens_id,act_id) for ens_id in range(10) for act_id in range(10)][task_id-1]
ensembles = pickle.load(open('ensembles_1e4.pickle','rb'))
ens = ensembles[ens_id]
activations = pickle.load(open('activated_ensembles_1e4.pickle','rb'))
act = activations[act_id]
N_tot = 10000
W = simple_submatrix(N_tot,ens)
e = load_module('ehe_detailed').EHE()
e.simulate_model_mat(np.random.random(N_tot),10*N_tot,W,deltaU,np.array(ens[act]))
np.save(outdir+"/avs"+str(task_id),e.avs)
np.save(outdir+"/avs_inds"+str(task_id),e.avs_inds)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocesssimple_submatrix_1e4 -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
