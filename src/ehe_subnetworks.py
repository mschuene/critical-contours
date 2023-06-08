def embed_ensembles(N_tot=1000,N_subunits=100,N_ensembles=50):
    return [np.random.choice(np.arange(N_tot),size=N_subunits,replace=False) for i in range(N_ensembles)]

def simple_submatrix(N_tot,ensembles,beta=2,alpha_factor=1):
    W = np.zeros((N_tot,N_tot))
    for ens in ensembles:
        W[np.ix_(ens,ens)] = ehe_critical_weights(len(ens))*alpha_factor
    N = len(ens)
    W[W == 0] = -beta*(1 - 1/np.sqrt(N))/N
    return W

def sim_2afc(N_s=100,N_e=50,K=0,beta=2,N_u=1000,num_ens=10,num_act=10,alpha_factor=1,background_contrast=1,figure_factor=1):
    ensembles = [embed_ensembles(N_tot=N_u,N_subunits=N_s,N_ensembles=N_e) for i in range(num_ens)]
    e = load_module('ehe_detailed').EHE()
    sim_res = {'ensembles':ensembles,'sim_res':[None]*num_ens,'N_s':100,'N_e':50,'K':0,'beta':2,'N_u':1000,'num_ens':10,'num_act':10}
    for i,ens in enumerate(ensembles):
        W = simple_submatrix(N_u,ens,beta=beta,alpha_factor=alpha_factor)
        figure_units = [ens[index] for index in np.random.choice(N_e,num_act,replace=False)] #ensembles does not have to be numpy array
        noise_units = [np.random.choice(list(set(range(N_u)) - set(fig_u)),K,replace=False) for fig_u in figure_units]
        background_units = [np.random.choice(list(set(range(N_u)) - set(noise_u)),N_s,replace=False) for noise_u in noise_units]
        # Simulate A: figure activation
        sim_res_a = []
        for fig_u,noise_u in zip(figure_units,noise_units):
            external_weights = np.zeros(W.shape[0],dtype=int)
            external_weights[noise_u] = 1
            external_weights[np.random.choice(fig_u,int(figure_factor*len(fig_u)),replace=False)] = 1
            #external_weights[np.hstack((np.array(fig_u),noise_u))] = 1
            e.simulate_model_mat(np.random.random(N_u),10*N_u,W,deltaU,external_weights)
            sim_res_a.append((e.avs,e.avs_inds,e.act_inds))
        # Simulate B: background activation
        sim_res_b = []
        for background_u,noise_u in zip(background_units,noise_units):
            external_weights = np.zeros(W.shape[0],dtype=int)
            external_weights[noise_u] = 1
            external_weights[background_u] = background_contrast
            e.simulate_model_mat(np.random.random(N_u),10*N_u,W,deltaU,external_weights)
            sim_res_b.append((e.avs,e.avs_inds,e.act_inds))
        sim_res_i = {'figure_units':figure_units,'noise_units':noise_units,'background_units':background_units,
                     'sim_res_a':sim_res_a,'sim_res_b':sim_res_b,'W':W,'ens':ens,'background_contrast':background_contrast}        
        sim_res['sim_res'][i] = sim_res_i
    return sim_res

def extract_2afc_avalanche_sizes_and_times(sim_res):
    avalanche_informations = []
    for ensemble_res in sim_res['sim_res']:
        avalanche_informations.extend(zip(ensemble_res['sim_res_a'],ensemble_res['sim_res_b']))
    return avalanche_informations

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline


def extract_time_slices(avs,avs_inds,act_inds,t_hor,min_time,max_time=None,max_num_obs=int(1e4)):
    if max_time is None: 
        max_time = act_inds[-1]
    e = load_module('ehe_detailed').EHE()
    act_inds = np.array(act_inds[1:]) # act_inds_always starts with 0
    avs_inds = np.array(avs_inds)
    avs_inds_range = np.searchsorted(act_inds,[min_time,max_time])
    avs_range = avs_inds[avs_inds_range-[0,1]]
    avs_sizes,avs_durations = e.get_avs_size_and_duration(avs[slice(*avs_range)],
                                avs_inds[slice(*avs_inds_range)] - avs_inds[avs_inds_range[0]])
    start_time,end_time = min_time,act_inds[avs_inds_range[1]-1] # end exclusive
    t_hor_multiples = np.arange(int(start_time/t_hor)+1,int(end_time/t_hor)+1+1)#multiples of t_hor lying in the time range
    splitpoints = np.searchsorted(act_inds[slice(*avs_inds_range)],t_hor_multiples*t_hor)
    # split the avs_sizes at each point where time becomes another multiple of t_hor
    # with searchsorted this also handles 0 observations in a time frame
    observations = np.split(avs_sizes,splitpoints)
    obs = observations if act_inds[-1] >= max_time else observations[-1] # potentially exclude last incompete observation
    return obs[:max_num_obs]

class TimeHorizonAvalancheExtractor(BaseEstimator):

    def __init__(self,t_hor=100,max_time=110000,min_time=10000):
        self.t_hor = t_hor
        self.max_time = max_time

    def fit(self,trials,y=None):
        pass

    def sample(self,avalanches_2afc,labels=None):
        X,y = ([],[])
        e = load_module('ehe_detailed').EHE()
        for ((avs_1,avs_inds_1,act_inds_1),(avs_2,avs_inds_2,act_inds_2)) in avalanches_2afc:
            X.append((extract_time_slices(avs_1,avs_inds_1,act_inds_1,self.t_hor,self.min_time,self.max_time),
                      extract_time_slices(avs_2,avs_inds_2,act_inds_2,self.t_hor,self.min_time,self.max_time)))

def thresholds(observations,s_0):
    return [np.sum(obs >= s_0) for obs in observations]

import numpy as np
def evaluate_sim_res(sim_res,t_hors=np.concatenate(([1,2,4,6,8,10,12,14,16,18],np.logspace(np.log10(20),3,20))).astype(int),
                     s_0s = np.concatenate(([1,2,4,6,8,10,12,14,16,18],np.logspace(np.log10(20),3,20))).astype(int),
                     min_time=10000):
    plt.ioff()
    accuracies_thresholds = np.empty((len(t_hors),len(s_0s)))
    accuracies_rates = np.empty((len(t_hors),len(s_0s)))
    hists_thresholds = np.empty((len(t_hors),len(s_0s)),dtype=object)
    fig_thresholds = np.empty((len(t_hors),len(s_0s)),dtype=object)
    avs_2afc = extract_2afc_avalanche_sizes_and_times(sim_res)
    fig_rates,fig_num_obs,hists_rates,hists_num_obs = [],[],[],[]
    for i,t_hor in enumerate(t_hors):
        print('i',i,flush=True)
        all_observations1,all_observations2 = [],[]
        for k,((avs1,avs_inds1,act_inds1),(avs2,avs_inds2,act_inds2)) in enumerate(avs_2afc):
            print('k',k,flush=True)
            max_time = min(np.max(act_inds1),np.max(act_inds2))
            if max_time > min_time:
                obs1 = extract_time_slices(avs1,avs_inds1,act_inds1,t_hor,min_time,max_time)
                obs2 = extract_time_slices(avs2,avs_inds2,act_inds2,t_hor,min_time,max_time)
                ncobs = min(len(obs1),len(obs2))
                all_observations1.extend(obs1[:ncobs])
                all_observations2.extend(obs2[:ncobs])
        r1,r2 = ([np.sum(obs) for obs in all_observations1],[np.sum(obs) for obs in all_observations2])
        bc = max(np.mean(r1)/np.mean(r2),1) # background contrast
        # now do it again scaled by the calculated background contrast
        all_observations1 = []
        all_observations2 = []
        for k,((avs1,avs_inds1,act_inds1),(avs2,avs_inds2,act_inds2)) in enumerate(avs_2afc):
            print('k',k,flush=True)
            max_time = min(np.max(act_inds1),np.max(act_inds2))
            if max_time > min_time:
                obs1 = extract_time_slices(avs1,avs_inds1,act_inds1,t_hor,min_time,int(max_time/bc))
                obs2 = extract_time_slices(avs2,avs_inds2,act_inds2,int(t_hor*bc),min_time,max_time)
                ncobs = min(len(obs1),len(obs2))
                all_observations1.extend(obs1[:ncobs])
                all_observations2.extend(obs2[:ncobs])
        print('extracted all observations',flush=True)
        plt.figure(figsize=(12,8))
        num_obs1,num_obs2 = ([len(obs) for obs in all_observations1],[len(obs) for obs in all_observations2])
        mi,ma = int(min(np.min(num_obs1),np.min(num_obs2))),int(max(np.max(num_obs1),np.max(num_obs2)))
        hist_num_obs = plt.hist([num_obs1,num_obs2],histtype='step',bins=np.arange(mi,ma+1),label=['figure','background'])
        acc_num_obs = acc_2afc(num_obs2,num_obs1)
        plt.title('number avalanches in observation time trame for t_hor='+str(t_hor)+" accuracy is "+str(acc_num_obs)+', bc '+str(bc))
        plt.legend()
        fig_num_obs.append(plt.gcf())
        print('num_obs figure and hist',flush=True)
        plt.figure(figsize=(12,8))
        rates1,rates2 = ([np.sum(obs) for obs in all_observations1],[np.sum(obs) for obs in all_observations2])
        acc_rates = acc_2afc(rates2,rates1)
        accuracies_rates[i,:] = acc_rates
        hist_rates = plt.hist([[r/t_hor for r in rates1],[r/t_hor for r in rates2]],histtype='step',label=['figure','background'])
        plt.title('firing rates in time horizon for t_hor='+str(t_hor)+" accuracy is "+str(acc_rates)+', bc ' + str(bc))
        plt.legend()
        fig_rates.append(plt.gcf())
        print('firing rates figure and hists',flush=True)
        for j,s_0 in enumerate(s_0s):
            print('j',j,flush=True)
            plt.figure(figsize=(12,8))
            thds1,thds2 = (thresholds(all_observations1,s_0),thresholds(all_observations2,s_0))
            mi,ma = int(min(np.min(thds1),np.min(thds2))),int(max(np.max(thds1),np.max(thds2)))
            acc = acc_2afc(thds2,thds1)
            accuracies_thresholds[i,j] = acc
            hists_thresholds[i,j] = plt.hist([thds1,thds2],histtype='step',bins=np.arange(mi,ma+1),label=['figure','background'])
            plt.title('thredholds distribution for t_hor='+str(t_hor)+", s_0="+str(s_0)+" accuracy is "+str(acc)+', bc ' + str(bc))
            plt.legend()
            fig_thresholds[i,j] = plt.gcf()
    acc_figure,(ax1,ax2) = plt.subplots(2,1,figsize=(16,8))
    im1 = ax1.imshow(accuracies_thresholds,origin='lower')
    im2 = ax2.imshow(accuracies_rates,origin='lower')
    acc_figure.colorbar(im1,ax=ax1)
    acc_figure.colorbar(im2,ax=ax2)
    plt.ion()
    return {'accuracies_thresholds':accuracies_thresholds,
            'accuracies_rates':accuracies_rates,
            'hists_thresholds':hists_thresholds,
            'fig_thresholds':fig_thresholds,
            'avs_2afc':avs_2afc,
            'fig_rates':fig_rates,
            'fig_num_obs':fig_num_obs,
            'hists_rates':hists_rates,
            'hists_num_obs':hists_num_obs,
            'acc_figure':acc_figure}

import pickle

ext_dir = '/media/kima/50AC0F1379302ADD'

def sr_params2name_func(params):
    return 'sim_res'+','.join([str(c) for c in params])+'.pickle'

def phase_space_matrix(sim_res_func,sim_res_dir,parameter_array,params2name_func,dtype=float,error_value=-1):
    def inner(params):
        print('params',params)
        sim_res = None
        try:
            sim_res = pickle.load(open(sim_res_dir+'/'+params2name_func(params),'rb'))
        except:
            print('file not found or not loadable '+sim_res_dir+'/'+params2name_func(params),flush=True)
            print('for params '+str(params),flush=True)
        return error_value if sim_res is None else sim_res_func(sim_res)
    ret = np.empty(parameter_array.shape,dtype=dtype)
    it = np.nditer(ret,flags=['multi_index'])
    while not it.finished:
        ret[it.multi_index] = inner(parameter_array[it.multi_index])
        it.iternext()
    return ret

def num_avs_matrix(sim_res):
    """returns a matrix M_{i,j} = (a_{i,j},b_{i,j}) of the number of recorded avalanches
    for ensemble i and activation j of simulation a and b"""
    num_ens,num_act = (sim_res['num_ens'],sim_res['num_act'])
    M = np.empty((num_ens,num_act),dtype=object)
    for i,sr in enumerate(sim_res['sim_res']):
        for j,((_,avs_inds_a,_),(_,avs_inds_b,_)) in enumerate(zip(sr['sim_res_a'],sr['sim_res_b'])):
            M[i,j] = (len(avs_inds_a),len(avs_inds_b))
    return M


def inds2subnet_dict(ens):
    d = defaultdict(lambda :[])
    for i,e in enumerate(ens):
        for unit in e:
            d[unit].append(i)
    return d

def pairwise_overlap(ens):
    Ovl = np.empty([len(ens)]*2,dtype=object)
    for i,e in enumerate(ens):
        for j,e2 in enumerate(ens):
            Ovl[i,j] = len(set(e)-set(e2))
    return Ovl

import networkx
def tograph(W,only_pos=True):
    """converts W to a undirected (only_pos=True) networkx matrix"""

import networkx as  nx
import networkx.algorithms.approximation as app
import networkx.algorithms as alg
def create_graph(W,undirected=True):
    G = nx.Graph()
    G.add_nodes_from(range(W.shape[0]))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if i!=j and W[i,j] > 0:
                G.add_edge(i,j)
    return G

def evaluate_avalanche_plots(sim_res):
    plt.ioff()
    e = load_module('ehe_detailed').EHE()
    N_s,N_e,K = sim_res['N_s'],sim_res['N_e'],50
    all_avs_sizes_a,all_avs_sizes_b,all_avs_sizes_ref = [],[],[]
    num_ens,num_act = 10,10
    figures = np.empty((10,10),dtype=object)
    for ens in range(num_ens):
        for act in range(num_act):
            f,ax = plt.subplots(1,1,figsize=(16,10))
            avs_a,avs_inds_a,act_inds_a = sim_res['sim_res'][ens]['sim_res_a'][act]
            avs_sizes_a,avs_durations_a = e.get_avs_size_and_duration(avs_a,avs_inds_a)
            avs_b,avs_inds_b,act_inds_b = sim_res['sim_res'][ens]['sim_res_b'][act]
            avs_sizes_b,avs_durations_b = e.get_avs_size_and_duration(avs_b,avs_inds_b)
            e.simulate_model_const(np.random.random(N_s),len(avs_sizes_a),(1-1/np.sqrt(N_s))/N_s,deltaU)
            avs_sizes_ref = e.avs_sizes
            loglogplot(avs_sizes_a,ax=ax,label='avalanche sizes figure')
            loglogplot(avs_sizes_b,ax=ax,label='avalanche sizes background')
            loglogplot(avs_sizes_ref,ax=ax,label='reference N_s critical distribution')
            ax.set_title('avs distributions 2afc NS='+str(N_s)+', Ne='+str(N_e)
                         + ', K='+str(50)+', ensemble,act='+str(ens)+','+str(act))
            all_avs_sizes_a.extend(avs_sizes_a)
            all_avs_sizes_b.extend(avs_sizes_b)
            all_avs_sizes_ref.extend(avs_sizes_ref)
            ax.legend()
            figures[ens,act] = f
    f,ax = plt.subplots(1,1,figsize=(16,10))
    loglogplot(avs_sizes_a,ax=ax,label='avalanche sizes figure')
    loglogplot(avs_sizes_b,ax=ax,label='avalanche sizes background')
    loglogplot(avs_sizes_ref,ax=ax,label='reference N_s critical distribution')
    ax.set_title('all avs distributions 2afc NS='+str(N_s)+', Ne='+str(N_e)
                 + ', K='+str(50))
    ax.legend()
    plt.ion()
    return {'all_avalanche_figure':f,'ens+act_figures':figures}

def evaluate_population_activity(sim_res):
    e = load_module('ehe_detailed').EHE()
    N_s,N_e,K= sim_res['N_s'],sim_res['N_e'],sim_res['K']
    all_pop_act_a,all_pop_act_b = [],[]
    pop_acts_a = []; pop_acts_b = []
    num_ens,num_act = 5,5
    figures = np.empty((10,10),dtype=object)
    for ens in range(num_ens):
        for act in range(num_act):
            f,(ax_a,ax_b) = plt.subplots(2,1,figsize=(16,20))
            avs_a,avs_inds_a,act_inds_a = sim_res['sim_res'][ens]['sim_res_a'][act]
            avs_sizes_a,avs_durations_a = e.get_avs_size_and_duration(avs_a,avs_inds_a)
            avs_b,avs_inds_b,act_inds_b = sim_res['sim_res'][ens]['sim_res_b'][act]
            avs_sizes_b,avs_durations_b = e.get_avs_size_and_duration(avs_b,avs_inds_b)
            pop_act_a = np.zeros(act_inds_a[-1])
            pop_act_a[act_inds_a[1:-1]] = avs_sizes_a
            ax_a.plot(pop_act_a)
            ax_a.set_title('population activity figure Ns='+str(N_s)+', Ne='+str(N_e)
                         + ', K='+str(K)+', ensemble,act='+str(ens)+','+str(act))
            pop_act_b = np.zeros(act_inds_b[-1])
            pop_act_b[act_inds_b[1:-1]] = avs_sizes_b
            ax_b.plot(pop_act_b)
            ax_b.set_title('population activity background Ns='+str(N_s)+', Ne='+str(N_e)
                         + ', K='+str(K)+', ensemble,act='+str(ens)+','+str(act))
            all_pop_act_a.extend(pop_act_a)
            all_pop_act_b.extend(pop_act_b)
            pop_acts_a.append(pop_act_a)
            pop_acts_b.append(pop_act_b)
    # f,(ax_a,ax_b) = plt.subplots(2,1,figsize=(16,20))
    # ax_a.plot(all_pop_act_a)
    # ax_a.set_title('all pop act figure NS='+str(N_s)+', Ne='+str(N_e) + ', K='+str(K))
    # ax_b.plot(all_pop_act_b)
    # ax_b.set_title('all pop act background NS='+str(N_s)+', Ne='+str(N_e) + ', K='+str(K))
    return {'all_pop_act_a':all_pop_act_a,'all_pop_act_b':all_pop_act_b,'pop_acts_a':pop_acts_a,'pop_acts_b':pop_acts_u}

def analyze_results(task_id,suffix=''):
    # load dataset
    avs,avs_inds = [np.load("../avalanches/simple_submatrix"+suffix+"/"+s+str(task_id)+".npy") for s in ['avs','avs_inds']]
    # load ensemble and activation information
    (ens_id,act_id) = [(ens_id,act_id) for ens_id in range(10) for act_id in range(10)][task_id-1]
    ensembles = pickle.load(open('ensembles'+suffix+'.pickle','rb'))
    ens = ensembles[ens_id]
    activations = pickle.load(open('activated_ensembles'+suffix+'.pickle','rb'))
    act = activations[act_id]
    # look at avalanche size statistics in each subnet
    e = load_module('ehe_detailed').EHE()
    detailed_sub_avs = []
    sub_avs = []
    for ensemble in ens:
        (sub_avs_detailed,sub_avs_inds) = e.subnetwork_avalanches(avs,avs_inds,set(ensemble))
        (sub_avs_size,sub_avs_duration) = e.get_avs_size_and_duration(sub_avs_detailed,sub_avs_inds)
        detailed_sub_avs.append((sub_avs_detailed,sub_avs_inds))
        sub_avs.append((sub_avs_size,sub_avs_duration))
    return {"ens":ens,"act":act,"sub_avs":sub_avs,"detailed_sub_avs":detailed_sub_avs,'avs':avs,'avs_inds':avs_inds}


def analyze_results_random(task_id,suffix=''):
    # load dataset
    avs,avs_inds = [np.load("../avalanches/simple_submatrix_random"+suffix+"/"+s+str(task_id)+".npy") for s in ['avs','avs_inds']]
    # load ensemble and activation information
    (ens_id,act_id) = [(ens_id,act_id) for ens_id in range(10) for act_id in range(10)][task_id-1]
    ensembles = pickle.load(open('ensembles'+suffix+'.pickle','rb'))
    ens = ensembles[ens_id]
    activations = pickle.load(open('background_units'+suffix+'.pickle','rb'))
    act = activations[act_id]
    # look at avalanche size statistics in each subnet
    e = load_module('ehe_detailed').EHE()
    detailed_sub_avs = []
    sub_avs = []
    for ensemble in ens:
        (sub_avs_detailed,sub_avs_inds) = e.subnetwork_avalanches(avs,avs_inds,set(ensemble))
        (sub_avs_size,sub_avs_duration) = e.get_avs_size_and_duration(sub_avs_detailed,sub_avs_inds)
        detailed_sub_avs.append((sub_avs_detailed,sub_avs_inds))
        sub_avs.append((sub_avs_size,sub_avs_duration))
    (act_avs_detailed,act_avs_inds) = e.subnetwork_avalanches(avs,avs_inds,set(act))
    (act_avs_size,act_avs_duration) = e.get_avs_size_and_duration(act_avs_detailed,act_avs_inds)
    return {"ens":ens,"act":act,"sub_avs":sub_avs,"detailed_sub_avs":detailed_sub_avs,'avs':avs,'avs_inds':avs_inds,
            'act_detailed':(act_avs_detailed,act_avs_inds),'act_avs':(act_avs_size,act_avs_duration)}



def plot_avs_subnets(res):
    f,ax = plt.subplots(figsize=(16,12))
    for sub_avs,sub_avd in res['sub_avs']:
        loglogplot(sub_avs,ax=ax)
    act_avs,act_avd = res['sub_avs'][res['act']]
    #plot reference power law to expect from recurrent ehe network in just the activated subnetwork
    e = load_module('ehe_detailed').EHE()
    N_sub = len(res['ens'][res['act']])
    e.simulate_model_const(np.random.random(N_sub),len(act_avs),(1-1/np.sqrt(N_sub))/N_sub,deltaU)
    loglogplot(e.avs_sizes,ax=ax)


def plot_avs_subnets_random(res):
    f,ax = plt.subplots(figsize=(16,12))
    for sub_avs,sub_avd in res['sub_avs']:
        loglogplot(sub_avs,ax=ax)
    act_avs,act_avd = res['act_avs']
    loglogplot(act_avs,ax=ax)
    # plot reference power law to expect from recurrent ehe network in just the activated subnetwork
    e = load_module('ehe_detailed').EHE()
    N_sub = len(res['act'])
    e.simulate_model_const(np.random.random(N_sub),len(act_avs),(1-1/np.sqrt(N_sub))/N_sub,deltaU)
    loglogplot(e.avs_sizes,ax=ax)



def analyze_all_results(suffix=''):
    # produziere 1 Bild. bei dem die aktivierte Kurve die Mittelung
    # über alle aktiven kurven in den 100 realisierungen ist und die
    # nichtaktivierte Kurve ebenfalls über alle nichtaktivierten
    # kurven aller realisierungen gemittelt ist
    mean_avs_activated = []
    mean_avs_nonactivated = []
    N_sub = 0
    for task_id in range(1,101):
        res_tid = analyze_results(task_id)
        sub_avs = res_tid['sub_avs']
        N_sub = len(res_tid['ens'][0])
        mean_avs_nonactivated += sum([sub_avs[i][0] for i in range(len(sub_avs)) if i != res_tid['act']],[])
        mean_avs_activated.extend(sub_avs[res_tid['act']][0])
    f,ax = plt.subplots(figsize=(16,12))
    loglogplot(mean_avs_activated,ax=ax)
    loglogplot(mean_avs_nonactivated,ax=ax)
    e = load_module('ehe_detailed').EHE()
    e.simulate_model_const(np.random.random(N_sub),len(mean_avs_activated),(1-1/np.sqrt(N_sub))/N_sub,deltaU)
    return {'mean_avs_activated':mean_avs_activated,'mean_avs_nonactivated':mean_avs_nonactivated,'f':f,'ax':ax}

import itertools
def run_simulation(Ns,Ne,K,Nu=1000,Nl=10000,M_ens=10,M_act=10,beta=2):
    ensembles = [embed_ensembles(N_tot=Nu,N_subunits=Ns,N_ensembles=Ne) for i in range(M_ens)]
    acts = np.random.choice(Ne,M_act,replace=False)
    sim_res = []
    mean_avs_activated = []
    mean_avs_nonactivated = []
    sim_idx = 0
    for (ens,act) in itertools.product(ensembles,acts):
        W = simple_submatrix(Nu,ens,beta=beta)
        e = load_module('ehe_detailed').EHE()
        background = np.random.choice(list(set(range(Nu))-set(ens[act])),K,replace=False)
        e.simulate_model_mat(np.random.random(Nu),10*Nu,W,deltaU,
                             np.hstack((np.array(ens[act]),background)))
        print('simulation '+str(sim_idx),flush=True)
        avs,avs_inds,act_inds = e.avs,e.avs_inds,e.act_inds
        sub_avs = []
        for i,ensemble in enumerate(ens):
            (sub_avs_detailed,sub_avs_inds) = e.subnetwork_avalanches(avs,avs_inds,set(ensemble))
            if len(sub_avs_inds) > 0:
                (sub_avs_size,sub_avs_duration) = e.get_avs_size_and_duration(sub_avs_detailed,sub_avs_inds)
                sub_avs.append((sub_avs_size,sub_avs_duration))
                if i == act:
                    mean_avs_activated.extend(sub_avs_size)
                else:
                    mean_avs_nonactivated.extend(sub_avs_size)
        print('analysis '+str(sim_idx),flush=True)
        # Also random assignment...
        sim_res.append({"avs":avs,"avs_inds":avs_inds,'ens':ens,'act':act,'act_inds':act_inds,'sub_avs':sub_avs,
                        "background":background})
    f,ax = plt.subplots(figsize=(16,12))
    loglogplot(mean_avs_activated,ax=ax)
    loglogplot(mean_avs_nonactivated,ax=ax)
    e = load_module('ehe_detailed').EHE()
    print("start_const_simulation",flush=True)
    e.simulate_model_const(np.random.random(Ns),len(mean_avs_activated),(1-1/np.sqrt(Ns))/Ns,deltaU)
    loglogplot(e.avs_sizes,ax=ax)
    ax.set_title('Ns='+str(Ns)+",Ne="+str(Ne)+",K="+str(K)+",beta="+str(beta))
    return {'mean_avs_activated':mean_avs_activated,'mean_avs_nonactivated':mean_avs_nonactivated,"sim_res":sim_res,"f":f,"ax":ax}

import itertools
def run_simulation_random(Ns,Ne,Nu=1000,Nl=10000,M_ens=10,M_act=10):
    ensembles = [embed_ensembles(N_tot=Nu,N_subunits=Ns,N_ensembles=Ne) for i in range(M_ens)]
    backgrounds = [np.random.choice(Nu,Ns,replace=False) for i in range(M_act)]
    sim_res = []
    mean_avs_activated = []
    mean_avs_nonactivated = []
    sim_idx = 0
    for (ens,background) in itertools.product(ensembles,backgrounds):
        W = simple_submatrix(Nu,ens)
        e = load_module('ehe_detailed').EHE()
        e.simulate_model_mat(np.random.random(Nu),10*Nu,W,deltaU,np.array(background))
        print('simulation '+str(sim_idx),flush=True)
        avs,avs_inds,act_inds = e.avs,e.avs_inds,e.act_inds
        sub_avs = []
        for i,ensemble in enumerate(ens):
            (sub_avs_detailed,sub_avs_inds) = e.subnetwork_avalanches(avs,avs_inds,set(ensemble))
            if len(sub_avs_inds) > 0:
                (sub_avs_size,sub_avs_duration) = e.get_avs_size_and_duration(sub_avs_detailed,sub_avs_inds)
                sub_avs.append((sub_avs_size,sub_avs_duration))
                mean_avs_nonactivated.extend(sub_avs_size)
        (background_avs_detailed,background_avs_inds) = e.subnetwork_avalanches(avs,avs_inds,set(background))
        if len(background_avs_inds) > 0:
            (background_avs_size,background_avs_duration) = e.get_avs_size_and_duration(background_avs_detailed,background_avs_inds)
            mean_avs_activated.extend(background_avs_size)
        print('analysis '+str(sim_idx),flush=True)
        sim_res.append({"avs":avs,"avs_inds":avs_inds,'ens':ens,'background':background,'act_inds':act_inds,'sub_avs':sub_avs,
                        "background":background})
    f,ax = plt.subplots(figsize=(16,12))
    loglogplot(mean_avs_activated,ax=ax)
    loglogplot(mean_avs_nonactivated,ax=ax)
    e = load_module('ehe_detailed').EHE()
    print("start_const_simulation",flush=True)
    e.simulate_model_const(np.random.random(Ns),max(len(mean_avs_activated),10*Nu),(1-1/np.sqrt(Ns))/Ns,deltaU)
    loglogplot(e.avs_sizes,ax=ax)
    ax.set_title('Ns='+str(Ns)+",Ne="+str(Ne))
    return {'mean_avs_activated':mean_avs_activated,'mean_avs_nonactivated':mean_avs_nonactivated,"sim_res":sim_res,"f":f,"ax":ax}

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline

class TimeHorizonAvalancheExtractor(BaseEstimator):
    """Takes the avalanches during the simulation as input and transforms them to
       the largest avalanche size in the observed external input time steps num_obs"""

    def  __init__(self,t_hor=100,max_size=100000):
        self.t_hor = t_hor
        self.max_size=max_size

    def fit(self,trials,y=None):
        return self

    def sample(self,sim_results,labels=None):
        X,y = ([],[])
        for sim_res in sim_results:
            # split the simulation results into bins of t_hor external time steps
            # see if it is a background or a figure activation
            figure_simulation = 'act' in sim_res.keys()
            avs,avs_inds,act_inds = (sim_res['avs'],sim_res['avs_inds'],sim_res['act_inds'])
            e = load_module('ehe_detailed').EHE()
            avs_sizes,avs_durations = e.get_avs_size_and_duration(avs,avs_inds)
            duration_between_avs = np.diff(act_inds)
            time = np.cumsum(duration_between_avs)
            t_hor_rem = (time/self.t_hor).astype(int)
            splitpoints = np.searchsorted(t_hor_rem,np.arange(1,t_hor_rem[-1]))
            # split the avs_sizes at each point where time becomes another multiple of t_hor
            observations = np.split(avs_sizes,splitpoints)
            X.extend(observations)
            y.extend([figure_simulation]*len(observations))
            if len(y) > self.max_size:
                return (X,y)
        return (X,y)

    def fit_sample(self, X, y=None):
        return self.fit(X,y).sample(X,y)

class MaxAvalancheExtractor(BaseEstimator,TransformerMixin):
    """ Transforms the given avalanche sizes to their maximum value"""

    def __init__(self):
        pass

    def transform(self,X,y=None):
        return [[np.max(np.hstack(([0],avs_sizes)))] for avs_sizes in X]

    def fit(self,X,y=None):
        print("hi from fit, self is ",self)
        return self

    #def fit_transform(self,X,y=None):
    #    print("hi from fit_transform, self is ",self)
    #    return self.fit(X,y).transform(X,y)

def figure_detector(t_hor):
    return Pipeline([('extract_t_hor',TimeHorizonAvalancheExtractor(t_hor)),
                     ('max_avs_size',MaxAvalancheExtractor()),
                     ('classification',GaussianNB())])

from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics
def plot_roc(fpr,tpr,auc=None,ax=None):
    lw = 2
    ax = plt.subplots()[1] if ax is None else ax
    auc = auc if auc is not None else metrics.auc(fpr,tpr)
    ax.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = ' + str(auc) + " )")
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

def extract_data(t_hor,sim_res_a,sim_res_b,exclude_beginning=1000,max_size=None):
    print('max_size',max_size)
    avalanches_a,labels_a = TimeHorizonAvalancheExtractor(t_hor,max_size=max_size).fit_sample(sim_res_a)
    avalanches_b,labels_b = TimeHorizonAvalancheExtractor(t_hor,max_size=max_size).fit_sample(sim_res_b)
    start_idx = int(np.ceil(exclude_beginning/t_hor))
    min_size = min(len(avalanches_a),len(avalanches_b))
    # achtung geschummelt, nicht unbedingt über gleiche realisierung gegangen etc....
    # X,y = ([],[])
    # 
    # for a,b in zip(avalanches_a[start_idx:],avalanches_b[start_idx:]):
    #     pos = np.random.rand() > 0.5
    #     X.append((a,b) if pos  else (b,a))
    #     y.append(0 if pos else 1)
    # return (X,y)
    if max_size is None:
        max_size = min_size
    return (avalanches_a[start_idx:min(min_size,max_size)],avalanches_b[start_idx:min(min_size,max_size)])

def get_dists(avs_a,avs_b,s_0):
    return ([len(av_a[av_a >= s_0]) for av_a in avs_a],
            [len(av_b[av_b >= s_0]) for av_b in avs_b])


def rate_dists(avs_a,avs_b):
    return ([np.sum(av_a) for av_a in avs_a],[np.sum(av_b) for av_b in avs_b])

def acc_2afc(samples1,samples2):
    ma = max(np.max(samples1),np.max(samples2))
    mi = min(np.min(samples1),np.min(samples2))
    h1 = np.cumsum(np.histogram(samples1,np.arange(mi,ma+2))[0]/len(samples1))
    h2 = np.cumsum(np.histogram(samples2,np.arange(mi,ma+2))[0]/len(samples2))
    x = np.hstack((1,1-h1))
    y = np.hstack((1,1-h2))
    Perf = np.sum(-np.diff(x)*(y[1:]+y[:-1])/2)
    return Perf

def get_acc_2afc(t_hor,s_0s,sim_res_a,sim_res_b,max_size=None):
    print('start',flush=True)
    avs_a,avs_b = extract_data(t_hor,sim_res_a,sim_res_b,max_size=max_size)
    print('extracted',len(avs_a),len(avs_b),flush=True)
    accs = []
    rate_dist_na,rate_dist_nb = rate_dists(avs_a,avs_b)
    rate_acc = acc_2afc(rate_dist_nb,rate_dist_na)
    for s_0 in s_0s:
        if s_0 > t_hor:
            accs.append(0)
            continue
        dist_na,dist_nb = get_dists(avs_a,avs_b,s_0)
        print('dists na and nb',len(avs_a),len(avs_b),flush=True)
        accs.append(acc_2afc(dist_nb,dist_na))
    return accs,rate_acc

class Classifier2AFC(BaseEstimator,ClassifierMixin):
    def __init__(self,s_0):
        self.s_0 = s_0;

    def fit(self,X,y=None):
        return self # in future fit s_0 here

    def predict(self,X):
        y = []
        for (avs_0,avs_1) in X:
           if len(avs_0[avs_0 > self.s_0]) > len(avs_1[avs_1 > self.s_0]):
               y.append(0)
           #elif len(avs_0[avs_0 > self.s_0]) == len(avs_1[avs_1 > self.s_0]):
           #    y.append(0 if np.random.rand() > 0.5 else 1)
           else: 
               y.append(1)
        return y

import numpy as np
def roc_analysis(X,y,thresholds=np.arange(0,11)):
    tpr,fpr = [],[]
    for th in thresholds: 
        print('threshold ',th)
        cl = Classifier2AFC(th)
        y_pred = cl.predict(X)
        cm = metrics.confusion_matrix(y,y_pred)
        tp,fp = cm[1,1],cm[0,1]
        tpr.append(tp/len(y))
        fpr.append(fp/len(y))
    return tpr,fpr

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from imblearn.pipeline import Pipeline

class TimeHorizonAvalancheExtractor(BaseEstimator):
    """Takes the avalanches during the simulation as input and transforms them to
       the largest avalanche size in the observed external input time steps num_obs"""

    def  __init__(self,t_hor=100):
        self.t_hor = t_hor

    def fit(self,trials,y=None):
        return self

    def sample(self,sim_results,labels=None):
        X,y = ([],[])
        for sim_res in sim_results:
            # split the simulation results into bins of t_hor external time steps
            # see if it is a background or a figure activation
            figure_simulation = 'act' in sim_res.keys()
            avs,avs_inds,act_inds = (sim_res['avs'],sim_res['avs_inds'],sim_res['act_inds'])
            e = load_module('ehe_detailed').EHE()
            avs_sizes,avs_durations = e.get_avs_size_and_duration(avs,avs_inds)
            duration_between_avs = np.diff(act_inds)
            time = np.cumsum(duration_between_avs)
            t_hor_rem = (time/self.t_hor).astype(int)
            splitpoints = np.searchsorted(t_hor_rem,np.arange(1,t_hor_rem[-1]))
            # split the avs_sizes at each point where time becomes another multiple of t_hor
            observations = np.split(avs_sizes,splitpoints)
            X.extend(observations)
            y.extend([figure_simulation]*len(observations))
        return (X,y)

    def fit_sample(self, X, y=None):
        return self.fit(X,y).sample(X,y)

class MaxAvalancheExtractor(BaseEstimator,TransformerMixin):
    """ Transforms the given avalanche sizes to their maximum value"""
    def __init__(self):
        pass

    def transform(self,X,y=None):
        return [[np.max(np.hstack(([0],avs_sizes)))] for avs_sizes in X]

    def fit(self,X,y=None):
        print("hi from fit, self is ",self)
        return self

def figure_detector(t_hor):
    return Pipeline([('extract_t_hor',TimeHorizonAvalancheExtractor(t_hor)),
                     ('max_avs_size',MaxAvalancheExtractor()),
                     ('classification',GaussianNB())])

def simple_overlap_matrix(N,overlap_inds,w_ovl):
    W = np.zeros((N,N))
    N_alpha_crit = (1 - 1/np.sqrt(N))
    w_nonovl = (N_alpha_crit - len(overlap_inds)*w_ovl)/(N - len(overlap_inds))
    W[:,overlap_inds] = w_ovl;
    W[:,np.setdiff1d(np.arange(N),overlap_inds)] = w_nonovl
    return W

import numpy as np
def const_critical_subnetwork_weights(N_whole,inds_sub,w = None):
    W = np.zeros((N_whole,N_whole))
    inds = np.array(inds_sub)
    if w is None:
        w = (1 - 1/np.sqrt(len(inds)))/len(inds)
    W[np.ix_(inds,inds)] = w
    return W

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def test_supercriticality(N_u,N_s,N_e,beta=2,num_tests=20,max_duration_factor=0.1):
    ehe = load_module('ehe_detailed').EHE()
    ehe.max_duration_factor = max_duration_factor
    ret = 0;
    for i in range(num_tests):
        ensemble = embed_ensembles(N_u,N_s,N_e)
        W = simple_submatrix(N_u,ensemble,beta=beta)
        ext_support = np.zeros(N_u)
        ext_support[np.unique(np.concatenate(ensemble))] = 1
        ehe.simulate_model_mat(np.ones(N_u)-10**-6,1,W,deltaU,ext_support)
        ret += len(ehe.avs_sizes)
    del ehe
    return ret/num_tests

def sample_row(N_s,start_N_e,beta=2,N_u=1000,num_tests=20):
    res_start = test_supercriticality(N_u,N_s,start_N_e,beta=beta,num_tests=num_tests )
    lower_Ne,higher_Ne = start_N_e,start_N_e
    lower_res,higher_res = [],[]
    # move left until always subcritical
    while True:
        lower_Ne -= 1
        if lower_Ne < 1:
            break;
        r = test_supercriticality(N_u,N_s,lower_Ne,beta=beta,num_tests=num_tests)
        if r == 1:
            break
        lower_res.append(r)
    lower_res.reverse()
    # move right until always supercritical
    while True:
        higher_Ne += 1
        r = test_supercriticality(N_u,N_s,higher_Ne,beta=beta,num_tests=num_tests)
        if r == 0:
            break
        higher_res.append(r)
    print((lower_res,[res_start],higher_res))    
    return (lower_Ne,higher_Ne),np.concatenate((lower_res,[res_start],higher_res))

def sample_on_and_on(start_N_s,start_N_e,beta=2,N_u=1000,num_tests=20,rows_store=5,direction=1):
    N_s = start_N_s
    N_e = start_N_e
    results = []
    while 0 < N_s <= N_u:
        for i in range(rows_store):
            print("explore row "+str(N_s)+' '+str(N_e))
            (l,u),trans = sample_row(N_s,N_e,beta=beta,N_u=N_u,num_tests=num_tests)
            results.append((N_s,(l,u),trans))
            N_s += direction;
            N_e = l; 
        pickle.dump(results,open('../avalanches/phase_spaces/psm_up_'+str(N_s)+'_beta='+str(beta),'wb'))

from collections import defaultdict  
import matplotlib.pyplot as plt
import numpy as np
from ehe import *
from utils import *
def subnetwork_assignments(N,O):
      """generates subnetwork assignment of N units to M subnetworks with
         overlap matrix O of shape (M,M). Returns a vector A of tuples
         with A[i] := units belonging to subnetwork i and a dictionary
         U with U[i] := list of subnetworks that unit i belongs to
      """
      O = np.array(O)
      assert ((O.sum(axis=1) - 2*np.diag(O)) <= 0).all(),'Not implemented yet'
      A = [()]*O.shape[0]
      U = defaultdict(lambda: ())
      remaining_units = np.arange(N)
      for i in range(O.shape[0]):
            #print("i",i)
            sel = []
            # choose number of units overlapping with network j < i
            for j in range(i):
                  #print("select ",O[i,j], " units from ",A[j])
                  sel.extend(np.random.choice(A[j],size=O[i,j],replace=False))
                  # choose remaining units from units not chosen jet
            #print('select ',O[i,i]-len(sel),' units from remaining ',remaining_units)
            sel.extend(np.random.choice(remaining_units,size=O[i,i]-len(sel),replace=False))
            #print("sel for i ",i," is ",sel)
            A[i] = sel
            remaining_units = np.setdiff1d(remaining_units,sel)
      return A

print(subnetwork_assignments(10,np.array([[4,2,2],[2,4,2],[2,2,5]])))

def subnetwork_connectivity_matrix(N,O,A=None,c=1):
    """Generates from the subnetwork assignment A a connectivity matrix
       TODO Works only for subnetworks of same size in the moment"""
    W = np.zeros((N,N))
    O = np.array(O)
    if A is None:
        A = subnetwork_assignments(N,O)
    # first recurrent critical weights for all subnetworks
    for i,S_i in enumerate(A):
        W[np.ix_(S_i,S_i)] = ehe_critical_weights(O[i,i])
    # Now construct inhibitory connections between pairwise overlapping subnetworks
    for i in range(len(A)):
        for j in range(len(A)):
            if i==j:
                continue
            elif O[i,j] > 0:
                units_not_in_j = np.setdiff1d(A[i],A[j])
                units_not_in_i = np.setdiff1d(A[j],A[i])
                #print(i,j,units_not_in_j)
                W[np.ix_(units_not_in_i,units_not_in_j)] = - c*(O[i,j]/(O[i,i]-O[i,j]))*(1 - 1/np.sqrt(O[i,i]))/O[i,i]
    return W

def subnetwork_connectivity_matrix_2(N,O,A=None,c=1):
    """Generates from the subnetwork assignment A a connectivity matrix
       TODO Works only for subnetworks of same size in the moment"""
    class Wrapper():
        def __init__(self,t):
            self.t = t
        def __repr__(self):
            return t.__repr__()
        def __add__(self,other):
            if type(other) == type(self):
                return Wrapper(self.t + other.t)
            return Wrapper(self.t + (other,))
        def extract_mean(self):
            if len(self.t) == 0:
                return 0
            else:
                return np.mean(self.t)

    vmean = np.vectorize(lambda x:x.extract_mean())
    W = np.full((N,N),Wrapper(()))
    O = np.array(O)
    if A is None:
        A = subnetwork_assignments(N,O)
        # first recurrent critical weights for all subnetworks
    for i,S_i in enumerate(A):
        W[np.ix_(S_i,S_i)] += ehe_critical_weights(O[i,i])
        # Now construct inhibitory connections between pairwise overlapping subnetworks
    W = vmean(W)
    for i in range(len(A)):
        for j in range(len(A)):
            if i==j:
                continue
            elif O[i,j] > 0:
                units_not_in_j = np.setdiff1d(A[i],A[j])
                units_not_in_i = np.setdiff1d(A[j],A[i])
                #print(i,j,units_not_in_j)
                overlap = list(set(A[i]).intersection(set(A[j])))
                W[np.ix_(units_not_in_i,units_not_in_j)] += - c*(O[i,j]/(O[i,i]-O[i,j]))*np.mean(W[np.ix_(overlap,overlap)])
    return W

def simulate_subnetwork(W,A,activated=[0],num_background=30,num_avs=10000,deltaU=deltaU,background_contrast=1):
    """ A is the subnetwork assignment """
    e = load_module('ehe_detailed').EHE()
    activated_inds = np.hstack(A[i] for i in activated)
    background_inds = np.random.choice(np.setdiff1d(np.arange(W.shape[0]),activated_inds),num_background)
    external_weights = np.zeros(W.shape[0],dtype=int)
    external_weights[activated_inds ] = 1
    external_weights[background_inds] = background_contrast
    print('external_weights ',external_weights,flush=True)
    e.simulate_model_mat(np.random.random(W.shape[0]),num_avs,W,deltaU,external_weights)
    return {'ehe':e,'activated_inds':activated_inds,'background_inds':background_inds}

def subnet_avs_inspector(A,avs):
    res = np.empty(avs.shape,dtype=object)
    for i,a in enumerate(avs):
        if a == np.iinfo(avs.dtype).max:
            res[i] = a
        else:
            res[i] = tuple(k for k in range(len(A)) if a in A[k])
    return res

import matplotlib.pyplot as plt
from ehe import ehe_critical_weights,deltaU
from utils import load_module
def test_subnet_connectivity(N_tot,num_avs,O,activated,num_background,c=1):
    f,ax = plt.subplots(O.shape[0],2,figsize=(8*O.shape[0],12))
    A = subnetwork_assignments(N_tot,O)
    W = subnetwork_connectivity_matrix(N_tot,O,A,c)
    res = simulate_subnetwork(W,A,activated=activated,num_avs=num_avs,num_background=num_background)
    res['A'] = A
    res['W'] = W
    for i in range(O.shape[0]):
        E1 = res['ehe']
        (sub_avs_detailed_i,sub_avs_indices_i) = E1.subnetwork_avalanches(set(A[i]))
        (sub_avs_sizes_i,sub_avs_durations_i) = E1.get_avs_size_and_duration(sub_avs_detailed_i,sub_avs_indices_i)
        loglogplot(sub_avs_sizes_i,ax=ax[i,0])
        ax[i,0].set_title('avs distribution in '+ ('activatet' if i in activated else 'non-activated') +' subnetwork '+str(i))
        E2 = load_module('ehe_detailed').EHE()
        N=O[i,i];E2.simulate_model_const(np.random.random(N),len(sub_avs_sizes_i),(1-1/np.sqrt(N))/N,deltaU)
        loglogplot(E2.avs_sizes,ax=ax[i,1])
        ax[0,1].set_title('corresponding avs distribution')
    return f,res
