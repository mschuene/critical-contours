#!/bin/bash

cd "$(dirname "$0")"
jobid=$(qsub -terse -N testjobarraynotif -q `cat /0/queue_short.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m a `#end abort suspend` -t 1-2 <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
python <<EOF
import pickle
import os
import sh
sh.cd("/home/maik/master/src")
print(os.environ)
task_id = int(os.environ.get('SGE_TASK_ID'))
import time
time.sleep(60)
print('hi')
EOF
EOT)
echo "submitted job with id $jobid"

qstat

jid=`echo "$jobid"| cut -d. -f1`

qsub -o /dev/null -e /dev/null -M maikschuenemann@gmail.com -m ea -b y -l h_rt=00:00:15 -hold_jid $jid -N 'Job_array_finished' true
