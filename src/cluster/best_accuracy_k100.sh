#!/bin/bash
cd "$(dirname "$0")"
mkdir best_accuracy_k100
cd best_accuracy_k100
jobid=$(qsub -terse -N best_accuracy_k100 -q `cat /0/maik/queue_maximus.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1 <<EOT
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
outdir = "../avalanches/best_accuracy_k100"
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

ext_dir = outdir+"/.."

def esr_params2name_func(params):
    return 'evaluated_sim_res'+','.join([str(c) for c in params])+'.pickle'

sim_res_parameter_array = np.array([[(Ns,Ne,100) for Ns in range(80,120,4)] for Ne in range(40,71,2)],dtype='i8,i8,i8')
def best_accuracy(esr):
    return np.max(esr['accuracies_thresholds'][-1])
psm = phase_space_matrix(best_accuracy,ext_dir+'/evaluate_sim_2afc_parameter_search_background_contrast_k100',sim_res_parameter_array,esr_params2name_func,error_value=0.4)

plt.imshow(psm,origin='lower')
plt.colorbar()

plt.title('criticality_regions indicated by total number of recorded avalanches')
plt.xlabel('N_s in range(80,120,4)')
plt.ylabel('N_e in range(40,71,2)')
plt.savefig('best_accuracy_k100.png')
pickle.dump(plt.gcf(),open(outdir+"/best_accuracy_k100.pickle","wb"))

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessbest_accuracy_k100 -q `cat /0/queue_maximus.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
