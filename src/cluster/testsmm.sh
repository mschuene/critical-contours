#!/bin/bash

cd "$(dirname "$0")"
qsub -N testsmm -q `cat /0/queue_long_64gb.txt` -cwd `#execute in current directory` -p 0 -M maikschuenemann@gmail.com -m eas `#end abort suspend` <<EOT
# source .bashrc since it isn't sourced by default
source $HOME/.bashrc
python <<EOF
import pickle
import os
import sh
print(os.environ)
task_id = 1

sh.cd("/home/maik/master/src")
from ehe import *;
print(load_module('ehe_arma').simulate_model_mat(np.random.random(100),10,ehe_critical_weights(100),deltaU))
EOF

EOT
