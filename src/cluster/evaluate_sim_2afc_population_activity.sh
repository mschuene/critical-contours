#!/bin/bash
cd "$(dirname "$0")"
mkdir evaluate_sim_2afc_population_activity
cd evaluate_sim_2afc_population_activity
jobid=$(qsub -terse -N evaluate_sim_2afc_population_activity -q `cat /0/maik/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-160 <<EOT
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
outdir = "../avalanches/evaluate_sim_2afc_population_activity"
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
sim_res = pickle.load(open(outdir+'/../sim_2afc_parameter_search/sim_res'+str(Ns)+","+str(Ne)+","+str(K)+".pickle","rb"))
res = evaluate_population_activity(sim_res)
pickle.dump(res,open(outdir+'/evaluated_sim_res_population_activity'+str(Ns)+","+str(Ne)+","+str(K)+".pickle",'wb'))
for i in range(10):
    for j in range(10):
        res['ens+act_figures'][i,j].savefig(outdir+'/pop_act_plots'+str(Ns)+","+str(Ne)+","+str(K)+' ens,act='+str(i)+','+str(j)+'.png')
res['all_pop_act_figure'].savefig(outdir+'/all_pop_act_plots'+str(Ns)+","+str(Ne)+","+str(K)+'.png')

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessevaluate_sim_2afc_population_activity -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
