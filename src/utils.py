1+1

import sys
from wurlitzer import sys_pipes
sys.path.insert(0,'./cpp')

from cppimport import imp
from time import gmtime, strftime
import sys
from cppimport.find import find_module_cpppath
from cppimport.importer import setup_module_data
from cppimport.importer import should_rebuild
import os.path
import os
def load_module(fullname):
    name = find_module_cpppath(fullname)
    hostname = os.uname()[1].replace("-","_")
    suffix = hostname+strftime("%Y%m%d%H%M%S",gmtime())
    (d,fname) = os.path.split(name)
    rendered_file_prefix = '.rendered.'+fname.split(".cpp")[0]
    with open(name) as f:
        content = f.read()
        new_content = content.replace("__module_suffix__",suffix)
    last_suffixes = [f.split(rendered_file_prefix)[-1].split('.cpp')[0]
                     for f in os.listdir(d) if f.startswith(rendered_file_prefix)]
    last_suffixes = [s for s in last_suffixes if s.startswith(hostname)]
    if len(last_suffixes) != 0:
        last_suffix = max(last_suffixes)
        with open(d+"/"+fullname+last_suffix+'.cpp','w') as flr:
            flr.write(content.replace("__module_suffix__",last_suffix))
        module_data = setup_module_data(fullname+last_suffix, name.split('.cpp')[0]+last_suffix+'.cpp')
        if not(should_rebuild(module_data)):
            return imp(fullname+last_suffix)
        for key in ['ext_path','filepath']:
            try:
              os.remove(module_data[key])
            except:
              pass
        try:
            pass
            #os.remove(d+"/"+'.rendered.'+fullname+last_suffix+'.cpp')    
            #os.remove(d+"/"+'.'+fullname+last_suffix+'.cpp.cppimporthash')
        except:
          pass
    with open(d+"/"+fullname+suffix+".cpp",'w') as f2:
        f2.write(new_content)
    print('reloading module')
    with sys_pipes():
        res = imp(fullname+suffix)
    return res

from mako.template import Template
from mako.runtime import Context
from io import StringIO
from os.path import basename
from os.path import splitext

def render_cluster_template(**opts):
    default_options={'queue':'long_64gb',
                     'mail':'maikschuenemann@gmail.com',
                     'output':'o',
                     'error':'e',
                     'priority':0,
                     'task_ids':'1',
                     'content':"print('hi')",
                     'max_threads':1,
                     'postprocessing_content':"",
                     'interpreter':'python',
                     'template_file':"./util/cluster_template.sh",
                     'output_file':None}
    options = default_options.copy()
    options.update(opts)
    if 'name' not in options and options['output_file'] is not None:
        options['name'] = splitext(basename(options['output_file']))[0]
    template = Template(strict_undefined=True,filename=options['template_file'])
    output = StringIO()
    ctx = Context(output,**options)
    template.render_context(ctx)
    if options['output_file'] is not None:
        with open(options['output_file'],"w") as f:
            f.write(output.getvalue())
    return output.getvalue()

import sh
def ssh_cluster(servername='server_inline'):
    return sh.ssh.bake("-t",servername,
    "export GE_CELL=neuro;export SGE_ROOT=/sge-root; export SGE_CLUSTER_NAME=OurCluster;cd /home/maik/master/src;")

server= ssh_cluster('server')
server_inline = ssh_cluster()

def rsync_server(server='server_inline'):
    print('rsyncing server')
    print(sh.rsync('-rtvu','./',server+":/home/maik/master/src/"))
    #print(sh.rsync('-rtvu','../avalanches/',server+":/home/maik/master/avalanches/"))
    print('done')

def qsub(command="print('hi')",post_command='',name=None,outfile=None,servername='server_inline',execute=True,**options):
    ssh = ssh_cluster(servername)
    options['content'] = command
    options['postprocessing_content']=post_command
    if name is not None:
        options['name'] = name
        if outfile is None:
            outfile = "./cluster/"+name+".sh"
            options['output_file'] = outfile
    elif outfile is not None:
        options['output_file'] = outfile
    else:
        raise Exception('must be called with either name or outfile')
    rendered = render_cluster_template(**options)
    rsync_server(servername)
    if execute:
        print(ssh("sh "+options['output_file']))

def scp_result(filename,servername='server_inline'):
   outfile = '../avalanches/'+filename 
   try:
      os.makedirs(os.path.dirname(outfile))
   except:
        pass
   print(sh.scp(servername+':/home/maik/master/avalanches/'+filename,outfile))

import os
from os.path import basename
from os.path import splitext
import numpy as np
def post_command_concat(prefix,tid_range,axis=0):
    dname,bname = os.path.split(prefix)
    fname = splitext(bname)[0]
    arrays = [];
    failed = "";
    for tid in tid_range:
        try:
            arrays.append(np.load(os.path.join(dname,fname+str(tid)+".npy")))
        except:
            failed += str(tid)+"\n"
    post_array = np.concatenate(arrays,axis=axis)
    np.save(os.path.join(dname,fname+"concatenated.npy"),post_array)
    with open(os.path.join(dname,'failed_idx.txt'),'w') as f:
        f.write(failed)

import os
from os.path import basename
from os.path import splitext
from os.path import join
import numpy as np

def concat_detailed(avs_prefix,avs_inds_prefix,tid_range):
    dname,bname = os.path.split(avs_prefix)
    fname = splitext(bname)[0]
    fname_inds = splitext(basename(avs_inds_prefix))[0]
    concatenated_avs = []
    concatenated_avs_inds = []
    failed = ""
    for tid in tid_range:
        try:
            avs_tid = np.load(join(dname,fname+str(tid)+".npy"))
            avs_inds_tid = np.load(join(dname,fname_inds+str(tid)+".npy")) + len(concatenated_avs)
            concatenated_avs.extend(avs_tid)
            concatenated_avs_inds.extend(avs_inds_tid)
        except:
            import traceback
            traceback.print_exc()
            failed += str(tid)+"\n"
    concatenated_avs = np.array(concatenated_avs)
    concatenated_avs_inds = np.array(concatenated_avs_inds)
    np.save(join(dname,fname+"concatenated.npy"),concatenated_avs)
    np.save(join(dname,fname_inds+"concatenated.npy"),concatenated_avs_inds)
    with open(os.path.join(dname,'failed_idx.txt'),'w') as f:
        f.write(failed)
    return (concatenated_avs,concatenated_avs_inds)

def sym_kl(pdf,pl_exp,upper_limit=None,lower_limit=None):
    pdf_emp,unq_points = pdf
    lower_limit = unq_points[0] if lower_limit is None else lower_limit
    upper_limit = unq_points[-1] if upper_limit is None else upper_limit
    pdf = np.zeros(upper_limit - lower_limit + 1)
    for pe,up in zip(pdf_emp,unq_points): 
        if 0 <= up - lower_limit < len(pdf):
            pdf[up - lower_limit] = pe
    pdf[pdf != 0] /= np.sum(pdf)        
    pdf_pl,unq_true = discrete_power_law_dist(pl_exp,
                                              lower_limit=lower_limit,
                                              upper_limit=upper_limit)
    tmp = (pdf - pdf_pl)*(np.log(pdf)-np.log(pdf_pl))
    tmp[~np.isfinite(tmp)] = 0
    return np.nansum(tmp)

def sym_kl_old(pdf,pl_exp,N):
    pdf,unq_points = pdf
    pdf_pl,unq_true = discrete_power_law_dist(pl_exp,lower_limit=1,upper_limit=unq_points[-1])
    tmp = (pdf - pdf_pl)*(np.log(pdf)-np.log(pdf_pl))
    tmp[~np.isfinite(tmp)] = 0
    return np.nansum(tmp)

def discrete_power_law_dist(pl_exp,lower_limit=1,upper_limit=10000):
    """limits are inclusive"""
    unique = list(range(lower_limit,upper_limit+1))
    pdf_pl = np.array([np.power(l,pl_exp) for l in unique])
    return (pdf_pl/np.sum(pdf_pl),np.array(unique))

def kl(pdf,pl_exp,N):
    pdf,unq_points = pdf
    pdf_pl,unq_true = discrete_power_law_dist(pl_exp,lower_limit=1,upper_limit=unq_points[-1])
    tmp = pdf_pl*(np.log(pdf)-np.log(pdf_pl))
    tmp[~np.isfinite(tmp)] = 0
    return -np.nansum(tmp)
