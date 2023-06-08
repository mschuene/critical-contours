#!/bin/bash
cd "$(dirname "$0")"
mkdir p_value_num_samples
cd p_value_num_samples
jobid=$(qsub -terse -N p_value_num_samples -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 100000 <<EOT
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
outdir = "../avalanches/p_value_num_samples"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ)
task_id = int(os.environ.get('SGE_TASK_ID'))
from ehe import * 
from utils import * 
from plfit_cd import * 

def p_values(task_id,outdir,exponent=2.0,recreate_ehe_plot_res=pickle.load(open('../avalanches/recreate_ehe_plot_res.pickle','rb'))):
    resdict = dict()
    for k in recreate_ehe_plot_res.keys():
        avalanches = np.array(recreate_ehe_plot_res[k][0])[100000:100000+task_id]
        alpha,p_value = plfit_cd_discrete(avalanches,5,10000,fix_exponent=exponent)
        resdict[k] = p_value
    pickle.dump(resdict,open(outdir+"/p_value"+str(exponent)+"num_samples"+str(task_id)+".pickle",'wb'))
p_values(task_id,outdir)

EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub -N postprocessp_value_num_samples -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m ea `#end abort suspend` -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
python <<EOF
EOF
EOT

qstat
