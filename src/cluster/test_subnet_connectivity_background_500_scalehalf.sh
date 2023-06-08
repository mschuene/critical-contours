#!/bin/bash
cd "$(dirname "$0")"
mkdir test_subnet_connectivity_background_500_scalehalf
cd test_subnet_connectivity_background_500_scalehalf
jobid=$(qsub -terse -N test_subnet_connectivity_background_500_scalehalf -q `cat /0/maik/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-4 <<EOT
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
outdir = "../avalanches/test_subnet_connectivity_background_500_scalehalf"
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

import pickle
f,res = test_subnet_connectivity(10000,100000,np.array([[1000,250,300],[250,2000,500],[300,500,3000]]),
                                [[0],[0,1],[0,2],[0,1,2]][task_id-1],500,c=1/2)
f.savefig(outdir+'/avalanches'+str(task_id)+'.png')
res['figure'] = f
res['avs'] = res['ehe'].avs
res['avs_inds'] = res['ehe'].avs_inds
del res['ehe']
pickle.dump(res,open(outdir+'/results'+str(task_id)+'.pickle','wb'))

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocesstest_subnet_connectivity_background_500_scalehalf -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
