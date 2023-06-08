#!/usr/bin/env python

import sh
import sys
from os.path import basename,dirname
from glob import glob
servername = sys.argv[1]
serverfile = sys.argv[2]
localpath  = sys.argv[3]
sdir = dirname(serverfile);sname = basename(serverfile)
print(servername,serverfile,localpath)
print(sh.ssh("-t",servername,'split','-b','100K',serverfile,serverfile+'splitted'))
print("splitted file on server side")
print("start scp ")
print(sh.scp(servername+":"+serverfile+"splitted*",localpath,_fg=True))
print("delete splitted files on server")
print(sh.ssh("-t",servername,'rm',serverfile+'splitted*'))
print("cat splitted files local and delete them")
print(sh.cd(localpath))
print(sh.cat(glob(sname+"splitted*"),_out=sname))
print(sh.rm(glob(sname+"splitted*")))



