#!/bin/bash
cd "$(dirname "$0")"
mkdir finite_size_scaling
cd finite_size_scaling
jobid=$(qsub -terse -N finite_size_scaling -q `cat /0/queue_long.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 6-7 <<EOT
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
outdir = "../avalanches/finite_size_scaling"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
task_id = int(os.environ.get('SGE_TASK_ID'))
from ehe import * 
from utils import * 
from plfit_cd import * 

ehe_arma = load_module('ehe_arma')
Ns = [100,300,500,700,1000,3000,5000,7000,10000]
N = Ns[task_id-1]
avs,avd = ehe_arma.simulate_model_const(np.random.random(N),int(1e8),(1-1/np.sqrt(N))/N,deltaU)
np.save(outdir+"/avs"+str(N),avs)
np.save(outdir+"/avd"+str(N),avd)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessfinite_size_scaling -q `cat /0/queue_long.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
