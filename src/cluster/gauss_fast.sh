#!/bin/bash
cd "$(dirname "$0")"
mkdir gauss_fast
cd gauss_fast
jobid=$(qsub -terse -N gauss_fast -q maximus+long64_gb -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1 <<EOT
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
outdir = "../avalanches/gauss_fast"
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

import time
N = 10000;
W = (0.8 +np.random.normal(0,0.1,(N,N)))/N
num_avs = int(1e7)
ehe = load_module('ehe_detailed')
e = ehe.EHE();
start = time.time()
e.simulate_model_mat(np.random.random(N),10000,W,deltaU);
end = time.time()
print("elapsed time ",end-start)
np.save(outdir+"/avs"+str(task_id),e.avs)
np.save(outdir+"/avs_inds"+str(task_id),e.avs_inds)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessgauss_fast -q maximus+long64_gb -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
# common setup code
import pickle
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/gauss_fast"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ,flush=True)
from ehe import * 
from utils import * 
from plfit_cd import * 
from ehe_subnetworks import *
concat_detailed(outdir+'/avs',outdir+'/avs_inds',range(1,2))
EOF
EOT

qstat
