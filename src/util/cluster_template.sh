#!/bin/bash
cd "$(dirname "$0")"
mkdir ${name}
cd ${name}
jobid=$(qsub -terse\
 -N ${name}\
% if queue in ['short','short_64gb','long','long_64gb','maximus','maximus+long_64gb','maximus+long']:
 -q `cat /0/maik/queue_${queue}.txt`\
% else:
 -q ${queue}\
% endif
 -cwd `#execute in current directory`\
 -p ${priority}\
% if mail is not None:
 -M ${mail}\
 -m a `#end abort suspend`\
% endif
% if max_threads > 1:
 -pe omp ${max_threads}\
% endif
 -t ${task_ids}\
 <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
% if max_threads != 0:
export OMP_NUM_THREADS=${max_threads}
% endif
export QT_QPA_PLATFORM=offscreen
${interpreter} <<EOF
# common setup code
import matplotlib
matplotlib.use('Agg')
import pickle 
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/${name}"
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
${content}
EOF
EOT
)

echo "submitted job with id $jobid"
jid=`echo $jobid | cut -d. -f1`
qsub\
 -N ${"postprocess"+name}\
% if queue in ['short','short_64gb','long','long_64gb','maximus']:
 -q `cat /0/queue_${queue}.txt`\
% else:
 -q ${queue}\
% endif
 -cwd `#execute in current directory`\
 -p ${priority}\
% if mail is not None:
 -M ${mail}\
 -m ea `#end abort suspend`\
% endif
 -l h_rt=00:00:15 -hold_jid $jid <<EOT
source $HOME/.bashrc
${interpreter} <<EOF
% if postprocessing_content != '':
# common setup code
import pickle
import os
import sh
sh.cd("/home/maik/master/src")
outdir = "../avalanches/${name}"
try:
    os.mkdir(outdir)
except:
    pass
print(os.environ,flush=True)
from ehe import * 
from utils import * 
from plfit_cd import * 
from ehe_subnetworks import *
${postprocessing_content}
% endif
EOF
EOT

qstat
