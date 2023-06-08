#!/bin/bash
cd "$(dirname "$0")"
mkdir figures_evaluate_sim_2afc_parameter_search
cd figures_evaluate_sim_2afc_parameter_search
jobid=$(qsub -terse -N figures_evaluate_sim_2afc_parameter_search -q `cat /0/maik/queue_maximus+long.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-160 <<EOT
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
outdir = "../avalanches/figures_evaluate_sim_2afc_parameter_search"
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

parameters = [(Ns,Ne,50) for Ns in range(80,120,4) for Ne in range(40,71,2)]
Ns,Ne,K = parameters[task_id - 1]
res = pickle.load(open(outdir+'/../evaluate_sim_2afc_parameter_search/evaluated_sim_res'+str(Ns)+","+str(Ne)+","+str(K)+".pickle","rb"))
params = np.concatenate(([1,2,4,6,8,10,12,14,16,18],np.logspace(np.log10(20),3,20))).astype(int)
for p,fig in zip(params,res['fig_rates']):
    fig.savefig(outdir+'/rate_fig_'+str(Ns)+","+str(Ne)+","+str(K)+'rate '+str(p)+'.png')
for p,fig in zip(params,res['fig_num_obs']):
    fig.savefig(outdir+'/num_obs_fig_'+str(Ns)+","+str(Ne)+","+str(K)+' num_obs '+str(p)+'.png')
[I,J] = res['fig_thresholds'].shape
for i in range(I):
    for j in range(J):
        res['fig_thresholds'][i,j].savefig(outdir+'/thresholds_fig'+str(Ns)+","+str(Ne)+","+str(K)+' t_hor='+str(params[i])+', s_0='+str(params[j])+'.png')
res['acc_figure'].savefig(outdir+'/acc_figure'+str(Ns)+","+str(Ne)+","+str(K)+'.png')

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessfigures_evaluate_sim_2afc_parameter_search -q maximus+long -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
