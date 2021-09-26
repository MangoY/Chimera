import numpy as np
import pandas as pd
import os
import logging
import math
from sklearn.model_selection import KFold
from timeit import default_timer as timer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.special import beta
from math import factorial as fact
from run_hls import get_perf, execute_hls
from pareto import getParetoFrontier, checkParetoOptimal, getProbabilityOfEval
from generate_directives import RandomDirectiveGenerator, DirectiveCrossover, DirectiveMutator, DirectiveWriter
from multiprocessing import Process
import re
import random
import scipy
import pprint
import pickle

def get_row(df, row):
    if len(df) == 0:
        # return empty if input is empty
        return pd.DataFrame()
    else:
        srs = pd.Series(np.full(len(df), True))
        for col in row.keys():
            val = row[col]
            srs = srs & (df[col] == val)
        return df[srs]
    
def getPopulation(dataset, threshold=1.05, exclude_infeasible=True):
    pareto_frontier = getParetoFrontier(dataset, exclude_infeasible=exclude_infeasible)
    population = pd.DataFrame(columns=dataset.columns)
    for i,row in enumerate(dataset.iloc):
        is_near_optimal = checkParetoOptimal(row, pareto_frontier, threshold)
        # 3/8 add this, since in some cases infeasible points can also provide useful information
        if exclude_infeasible:
            is_feasible = row.is_feasible
        else:
            is_feasible = True
        is_error = row.is_error
        if(is_near_optimal and is_feasible and (not is_error)):
            population = population.append(row)
    return population

def importanceAnalysis(models, dataset, weights=[1, 0.4, 0.1, 0.1, 0.4]):
    columns = dataset.columns[:-7]
    importances_raw = np.zeros_like(regr_lat.feature_importances_)
    
    for i, regr in enumerate(models):
        importances_raw = importances_raw + regr.feature_importances_ * weights[i]
    
    importances = {}
    pos = 0
    for i,variable_name in enumerate(columns):
        if (re.match('^loop_.+_type$' ,variable_name)):
            importances[variable_name] = np.sum(importances_raw[pos:pos+3])/np.sum(weights)
            pos = pos + 3
        elif (re.match('^loop_.+_factor$' ,variable_name)):
            importances[variable_name] = np.sum(importances_raw[pos:pos+1])/np.sum(weights)
            pos = pos + 1
        elif (re.match('^array_.+_type$' ,variable_name)):
            importances[variable_name] = np.sum(importances_raw[pos:pos+3])/np.sum(weights)
            pos = pos + 3
        elif (re.match('^array_.+_factor$' ,variable_name)):
            importances[variable_name] = np.sum(importances_raw[pos:pos+1])/np.sum(weights)
            pos = pos + 1
            
    return importances

def importanceAdjustment(importances, gamma, prob_scale=1.5):
    def getMutationProb(normalized_importance, gamma):
        x = gamma
        mean = normalized_importance
        stdev = 0.3 # let's test fixed stdev first
        y = (x - mean)/stdev
        return scipy.stats.norm.pdf(y)*np.sqrt(2*np.pi)/2
    
    adjusted_importances = importances.copy()
    params = importances.keys()
    vals = list(importances.values())
    max_val = np.max(vals)
    min_val = np.min(vals)
    
    for param in params:
        importance = float(importances[param])
        
        # normalized importance
        normalized_importance = (importance - min_val)/(max_val - min_val)
        
        # get adjusted importance/ probability of mutation
        adjusted_importance = getMutationProb(normalized_importance, gamma) * prob_scale
        # adjusted_importance = 0.2
        adjusted_importances.update({param: adjusted_importance})
    
    return adjusted_importances

class BetaDistCounter():
    def __init__(self):
        self.n = 0
        self.a = 1  # the number of times this socket returned a charge        
        self.b = 1  # the number of times no charge was returned     
    
    def reset(self):
        self.__init__()
    
    def update(self,R):
        self.n += 1    
        self.a += R
        self.b += (1-R)
        
    def sample(self):
        return np.random.beta(self.a,self.b)
    
def selectMethod(method_records, view_boundary=30):
    # initialize the beta distributions of them
    random_beta = BetaDistCounter()
    evo_beta = BetaDistCounter()
    mut_beta = BetaDistCounter()
    records_in_view = method_records[-view_boundary:]
    for method, result in method_records:
        if(method == 'rand'):
            random_beta.update(result)
        elif(method == 'evo'):
            evo_beta.update(result)
        elif(method == 'mut'):
            mut_beta.update(result)
        else:
            raise(AssertionError('Unknown proposing method'))
    
    rand_beta_sample = random_beta.sample()
    evo_beta_sample = evo_beta.sample()
    mut_beta_sample = mut_beta.sample()
    print('Rand beta sample: '+str(rand_beta_sample))
    print('Evo beta sample: '+str(evo_beta_sample))
    print('Mut beta sample: '+str(mut_beta_sample))

    select = np.argmax([rand_beta_sample, evo_beta_sample, mut_beta_sample])
    print([rand_beta_sample, evo_beta_sample, mut_beta_sample])
    methods = ['rand', 'evo', 'mut']
    print(methods[select])
    return methods[select]

def predict_perf(parameters, models):
    regr_lat, regr_dsp, regr_ff, regr_lut, regr_bram, clss_timeout = models
    
    encoded_features = preprocessing(pd.Series(parameters).to_frame().T.to_numpy(), feature_columns)

    pred_lat = regr_lat.predict(encoded_features)
    pred_dsp = regr_dsp.predict(encoded_features)
    pred_ff = regr_ff.predict(encoded_features)
    pred_lut = regr_lut.predict(encoded_features)
    pred_bram = regr_bram.predict(encoded_features)
    proba_timeout = clss_timeout.predict_proba(encoded_features)[0,0]

    predicted_perf = {'latency':pred_lat,
                      'dsp_perc':pred_dsp,
                      'ff_perc':pred_ff, 
                      'lut_perc':pred_lut,
                      'bram_perc':pred_bram}
    return predicted_perf, proba_timeout

def randProposal(directives_path, no_partitioning):
    _, parameters = dir_gen.generate_directives(out_file_path=None, no_partitioning=no_partitioning)

    return parameters

def evoProposal(no_partitioning, importances, gamma, dataset, models, pareto_frontier, n_families=3, n_offsprings=3, threshold=1.0, exclude_infeasible=True):
    pareto_set = getPopulation(dataset, threshold=1.0, exclude_infeasible=exclude_infeasible).sort_values('latency')
    
    # select only 1 point for each unique latency randomly
    unique_latencies = pareto_set['latency'].unique()
    unique_pareto_points_idx = []
    for lat in unique_latencies:
        eq_lat_points = pareto_set[pareto_set['latency'] == lat]
        rand_idx = random.randint(0, len(eq_lat_points)-1)
        selected = eq_lat_points.index[rand_idx]
        unique_pareto_points_idx.append(selected)
    pareto_points = pareto_set.loc[unique_pareto_points_idx]
    
    latency = pareto_frontier['latency']
    min_lat = min(latency)
    max_lat = max(latency)

    # get the population
    population = getPopulation(dataset, threshold=1.2, exclude_infeasible=exclude_infeasible)

    list_params = []
    list_probs = []

    for i in range(n_families):
        parent_idx = np.random.randint(0, len(population),size=1)
        parent_rand = population.iloc[parent_idx]
        latency_ranking = latency.append(parent_rand['latency'])
        ranking = latency_ranking.rank(method='min')
        rank = int(ranking.iloc[-1])

        parent_lat = parent_rand['latency'].to_numpy()[0]

        if(parent_lat <= min_lat): # unlikely to happen, just in case
            print('case 1: selected the fastest')
            if(len(pareto_points) == 1): # edge case, only 1 in pareto set
                parent_pareto = pareto_points.iloc[0] # other parent is the next fastest one
            else:
                parent_pareto = pareto_points.iloc[1] # other parent is the next fastest one
        elif(parent_lat >= max_lat): # it's worse than the pareto points
            print('case 2: selected the slowest')
            parent_pareto = pareto_points.iloc[-1] # other parent is the last one
        else:
            print('case 3')
            upper = pareto_points.iloc[rank-1] # neighboring point on the frontier with higher latency
            lower = pareto_points.iloc[rank-2] # neighboring point on the frontier with lower latency
            parent_pareto = random.choice([upper, lower])
        
        # parent_rand is a DF, parent pareto is a series
        parents = parent_rand.append(parent_pareto)

        for j in range(n_offsprings):
            _, offspring_parameters = crossover.generate_directives(out_file_path=None, 
                                                                    no_partitioning=no_partitioning, 
                                                                    context=parents)
            offspring_perf,_ = predict_perf(offspring_parameters, models)
            list_params.append(offspring_parameters)
            list_probs.append(getProbabilityOfEval(offspring_perf, pareto_frontier, threshold=threshold))

            _, mutant_parameters = mutator.generate_directives(out_file_path=None, 
                                                        no_partitioning=no_partitioning, 
                                                        context=(offspring_parameters, importances))
            mutant_perf,_ = predict_perf(mutant_parameters, models)
            list_params.append(mutant_parameters)
            list_probs.append(getProbabilityOfEval(mutant_perf, pareto_frontier, threshold=threshold))
            
    for i in range(1, len(list_params)+1):
        best = np.argsort(list_probs)[-i]
        proba_eval = list_probs[best]
        parameters = list_params[best]
        if (get_row(dataset, parameters).empty): 
            break
    return parameters, proba_eval

def mutProposal(no_partitioning, importances, gamma, dataset, models, pareto_frontier, n_mutants=3, threshold=1.2, exclude_infeasible=True):
    pareto_set = getPopulation(dataset, threshold=1.0, exclude_infeasible=exclude_infeasible)
    
    # select only 1 point for each unique latency randomly
    unique_latencies = pareto_set['latency'].unique()
    unique_pareto_points_idx = []
    for lat in unique_latencies:
        eq_lat_points = pareto_set[pareto_set['latency'] == lat]
        rand_idx = random.randint(0, len(eq_lat_points)-1)
        selected = eq_lat_points.index[rand_idx]
        unique_pareto_points_idx.append(selected)
    pareto_points = pareto_set.loc[unique_pareto_points_idx]

    rand_idx = random.randint(0, len(pareto_points)-1)
    _, mutant_parameters = mutator.generate_directives(out_file_path=None, 
                                            no_partitioning=no_partitioning, 
                                            context=(pareto_points.iloc[rand_idx], importances))
    mutant_perf,_ = predict_perf(mutant_parameters, models)
    prob_eval = getProbabilityOfEval(mutant_perf, pareto_frontier, threshold=threshold)
    return mutant_parameters, prob_eval

def update_models(models, dataset):
    regr_lat, regr_dsp, regr_ff, regr_lut, regr_bram, clss_timeout = models
    
    # extract features and labels from the feature set
    feature_columns = dataset.columns[:len(dataset.columns)-7]
    label_columns = dataset.columns[len(dataset.columns)-7:]
    features = dataset[feature_columns].to_numpy()
    labels = dataset[label_columns].to_numpy()
    features_encoded = preprocessing(features, feature_columns)

    # first determine the fesibility and timeout 
    timeout= labels[:,-1].astype('bool')

    lat = labels[:, 0]
    dsp_perc = labels[:, 1]
    ff_perc = labels[:, 2]
    lut_perc = labels[:, 3]
    bram_perc = labels[:, 4]

    not_timeout = np.logical_not(labels[:,-1].astype('bool'))
    
    # Notice that the regressions are trained only on points that do not timeout
    # Otherwise, these points will disturb the prediction 
    regr_lat.fit(features_encoded[not_timeout], lat[not_timeout])
    regr_dsp.fit(features_encoded[not_timeout], dsp_perc[not_timeout])
    regr_ff.fit(features_encoded[not_timeout], ff_perc[not_timeout])
    regr_lut.fit(features_encoded[not_timeout], lut_perc[not_timeout])
    regr_bram.fit(features_encoded[not_timeout], bram_perc[not_timeout])
    
    # timeout prediction will be trained on all points
    clss_timeout.fit(features_encoded, timeout)

def preprocessing(features, columns):
    # define the categories and encoders
    # the reason to use predefined categories is to avoid special cases, where some of the input features
    # only have one category present in the current dataset
    loop_directive_types = ['pipeline','unroll','none']
    array_directive_types = ['cyclic','block','complete','none']
    #enc_loop = OneHotEncoder(categories=[loop_directive_types], drop='first', sparse=False)
    #enc_array = OneHotEncoder(categories=[array_directive_types], drop='first', sparse=False)
    # for now, we do not drop the category
    enc_loop = OneHotEncoder(categories=[loop_directive_types], sparse=False)
    enc_array = OneHotEncoder(categories=[array_directive_types], sparse=False)
    
    #
    list_features_encoded = []
    #for i in range(features.shape[1]):
    for i,col in enumerate(columns):
        feature = features[:,i].reshape(1,-1).transpose()
        
        # detect data type of a feature
        if isinstance(feature[0][0], str):
            # identify the type of feature
            if (re.match('^loop_.+_type$' ,col)):
                encoder = enc_loop
            elif (re.match('^array_.+_type$' ,col)):
                encoder = enc_array
            else:
                raise AssertionError('unknown directive types')
            
            # encode the feature
            encoded = encoder.fit_transform(feature).astype('int')
            list_features_encoded.append(encoded)
        else:
            list_features_encoded.append(feature)

    encoded_features = np.concatenate(list_features_encoded, axis=1)
    return encoded_features