# -*- coding: utf-8 -*-
"""
Created on Agu 29 2019

@author: Camilo Montenegro
"""

import sys
import re
import os
import subprocess
import configparser as cp
import time
import resourceAllocationOptimizer as resourceOptimization
from shutil import copyfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import  hp

#Módulos de SIMOD
from support_modules import support as sup 
from support_modules.readers import bpmn_reader as br #FIJO
from support_modules.readers import process_structure as gph #FIJO
from support_modules.writers import xml_writer_scylla as xml_scylla
from support_modules.writers import assets_writer as assets_writer
from extraction import parameter_extraction as par
from extraction import log_replayer as rpl
from support_modules.readers import log_reader as lr


def mining_structure(settings, epsilon, eta):
    """Execute splitminer for bpmn structure mining.
    Args:
        settings (dict): Path to jar and file names
        epsilon (double): Parallelism threshold (epsilon) in [0,1]
        eta (double): Percentile for frequency threshold (eta) in [0,1]
    """
    print(" -- Mining Process Structure --")
    print(settings['miner_path'])
    args = ['java', '-jar', settings['miner_path'],
            str(epsilon), str(eta),
            os.path.join(settings['output'], settings['file']),
            os.path.join(settings['output'], settings['file'].split('.')[0])]
    subprocess.call(args)


def simulate(settings, rep):
    """Executes SCYLLA Simulations.
    Args:
        settings (dict): Path to jar and file names
        java -jar scylla_V{}.jar --config=<your config file> --bpmn=<your first bpmn file> --sim=<your first sim file> --output=<your output path + rep> --enable-bps-logging
    """
    print("-- Executing SCYLLA Simulations --")
    args = ['java', '-jar', settings['scylla_path'],
            "--config=" + os.path.join(settings['output'], settings['file'].split('.')[0] + 'ScyllaGlobalConfig.xml'),
            "--bpmn=" +'inputs/'+settings['file'].split('.')[0] + '.bpmn',
            "--sim=" + os.path.join(settings['output'], settings['file'].split('.')[0] + 'ScyllaSimuConfig.xml'),
            "--output=" + os.path.join(settings['output'], "scyllaSim_rep_" + str(rep), ""),
            "--enable-bps-logging"]

    subprocess.call(args)


def measure_stats(settings, bpmn, rep, resource_table):
    """Executes BIMP Simulations.
    Args:
        settings (dict): Path to jar and file names
        rep (int): repetition number
    """
    timeformat = '%Y-%m-%d %H:%M:%S.%f'
    temp = lr.LogReader(os.path.join(settings['output'], 'sim_data',
                                     settings['file'].split('.')[0] + '_' + str(rep + 1) + '.csv'),
                        timeformat)
    process_graph = gph.create_process_structure(bpmn)
    _, _, temp_stats = rpl.replay(process_graph, temp, resource_table=resource_table, source='simulation',
                                  run_num=rep + 1)
    temp_stats = pd.DataFrame.from_records(temp_stats)
    role = lambda x: x['resource']
    temp_stats['role'] = temp_stats.apply(role, axis=1)
    return temp_stats

def reformat_path(raw_path):
    """Provides path support to different OS path definition"""
    route = re.split(chr(92) + '|' + chr(92) + chr(92) + '|' +
                     chr(47) + '|' + chr(47) + chr(47), raw_path)
    return os.path.join(*route)


def read_settings(settings):
    """Catch parameters fron console or code defined"""
    config = cp.ConfigParser(interpolation=None)
    config.read("./config.ini")
    # Basic settings
    settings['input'] = config.get('FOLDERS', 'inputs')
    settings['file'] = config.get('EXECUTION', 'filename')
    settings['output'] = os.path.join(config.get('FOLDERS', 'outputs'), sup.folder_id())
    settings['timeformat'] = config.get('EXECUTION', 'timeformat')
    settings['simulation'] = config.get('EXECUTION', 'simulation')
    settings['analysis'] = config.get('EXECUTION', 'analysis')
    settings['role_optimization'] = config.get('EXECUTION', 'role_optimization')
    settings['resource_optimization'] = config.get('EXECUTION', 'resource_optimization')
    
    settings['flag'] = config.get('EXECUTION', 'flag')
    settings['cooperation_policy'] = config.get('EXECUTION', 'cooperation_policy')
    settings['preference_policy'] = config.get('EXECUTION', 'preference_policy')
    
    settings['k'] = config.get('EXECUTION', 'k')
    settings['sim_percentage'] = config.get('EXECUTION', 'sim_percentage')
    settings['quantity_by_cost'] = config.get('EXECUTION', 'quantity_by_cost')
    settings['reverse'] = config.get('EXECUTION', 'reverse')
    settings['happy_path'] = config.get('EXECUTION', 'happy_path')
    settings['graph_roles_flag'] = config.get('EXECUTION','graph_roles_flag')
    
    
    

    # Conditional settings
    settings['miner_path'] = reformat_path(config.get('EXTERNAL', 'splitminer'))
    if settings['alg_manag'] == 'repairment':
        settings['align_path'] = reformat_path(config.get('EXTERNAL', 'proconformance'))
        settings['aligninfo'] = os.path.join(settings['output'],
                                             config.get('ALIGNMENT', 'aligninfo'))
        settings['aligntype'] = os.path.join(settings['output'],
                                             config.get('ALIGNMENT', 'aligntype'))
    if settings['simulation']:
        settings['repetitions'] = config.get('EXECUTION', 'repetitions')
        settings['scylla_path'] = reformat_path(config.get('EXTERNAL', 'scylla'))
        settings['simulator'] = config.get('EXECUTION', 'simulator')
    if settings['role_optimization']:
        settings['objective'] = config.get('OPTIMIZATION', 'objective')
        settings['criteria'] = config.get('OPTIMIZATION', 'criteria')
        settings['graph_optimization'] = config.get('OPTIMIZATION', 'graph_optimization')
    if settings['resource_optimization']:
        settings['cost'] = config.get('OPTIMIZATION', 'cost')
        settings['workload'] = config.get('OPTIMIZATION', 'workload')
        settings['flow_time'] = config.get('OPTIMIZATION', 'flow_time')
        settings['waiting_time'] = config.get('OPTIMIZATION', 'waiting_time')
        settings['log'] = config.get('OPTIMIZATION', 'log')
        settings['non_repeated_resources'] = config.get('OPTIMIZATION', 'non_repeated_resources')
        settings['generations'] = config.get('OPTIMIZATION', 'generations')
        settings['initial_population'] = config.get('OPTIMIZATION', 'initial_population')
        settings['min_population'] = config.get('OPTIMIZATION', 'min_population')
        settings['max_population'] = config.get('OPTIMIZATION', 'max_population')
    return settings


def objective(params):
    settings = read_settings(params) #Separar
    kpis = ['cost_total', 'flowTime_avg', 'waiting_avg', 'time_workload_avg']
    happy_path = True if settings['happy_path'] == 'True' else False
    simulation = True if settings['simulation'] == 'True' else False
    analysis = True if settings['analysis'] == 'True' else False
    optimization = True if settings['role_optimization'] == 'True' else False
    resource_optimization = True if settings['resource_optimization'] == 'True' else False

    graph_opti = True if settings['graph_optimization'] == 'True' else False
    graph_roles_flag = True if settings['graph_roles_flag'] == 'True' else False
    
    for f in settings['flag'].split(","):
        f = int(f)
        if f == 1:
            time_start = time.time()

            if optimization and settings['objective'] in kpis:
                opti_folder_id = sup.folder_id()
                global_results = dict()
                print("--- Optimization execution started at time:", str(time_start), "---")
            else:
                print("--- Execution started at time:", str(time_start), "---")
            ks = settings['k'].split(",")
            for k in range(int(ks[0]), int(ks[1]) + 1):
                print('Using the k = ' + str(k) + " frequent resources per activity")
                settings = read_settings(params)
                params['output'] = settings['output']
                if optimization:
                    temp = list(os.path.split(settings['output']))
                    folder_iter_id = temp[1]
                    temp = [temp[0], 'optimization_' + opti_folder_id, temp[1]]
                    settings['output'] = os.path.join(*temp)
                # Output folder creation
                if not os.path.exists(settings['output']):
                    os.makedirs(settings['output'])
                    os.makedirs(os.path.join(settings['output'], 'sim_data'))
                copyfile(os.path.join(settings['input'], settings['file']),
                         os.path.join(settings['output'], settings['file']))
                log = lr.LogReader('inputs/'+settings['file'].split('.')[0] + '.xes',
                                   settings['timeformat'])

                bpmn = br.BpmnReader('inputs/'+settings['file'].split('.')[0] + '.bpmn')
                print(bpmn)
                process_graph = gph.create_process_structure(bpmn)

                print("-- Mining Simulation Parameters --")
                parameters, process_stats = par.extract_parameters(log, bpmn, process_graph,
                                                                   flag=f, k=int(k),
                                                                   simulator=settings['simulator'].split(","),
                                                                   sim_percentage=0,
                                                                   quantity_by_cost=int(settings['quantity_by_cost']),
                                                                   reverse_cost=settings['reverse'],
                                                                   happy_path=happy_path, graph_roles_flag=graph_roles_flag)

                xml_scylla.print_parameters(os.path.join(settings['output'],
                                                         settings['file'].split('.')[0] + '.bpmn'),
                                            os.path.join(settings['output'],
                                                         settings['file'].split('.')[0] + 'Scylla.bpmn'),
                                            parameters['scylla'])

                if simulation:

                    for rep in range(int(settings['repetitions'])):
                        print("Experiment #" + str(rep + 1))
                        try:
                            simulate(settings, rep + 1)
                            if analysis:
                                file_name = settings['file'].split(".")[0]
                                file_global_config_name = file_name + "ScyllaGlobalConfig_resourceutilization.xml"
                                scylla_output_path = os.path.join(settings['output'], "scyllaSim_rep_" + str(rep + 1),
                                                                  file_global_config_name)
                                new_path = os.path.join(settings['output'], "ResourceUtilizationResults")
                                if not os.path.exists(new_path):
                                    os.mkdir(new_path)
                                new_path = os.path.join(new_path, file_name + "_rep" + str(rep + 1))
                                if optimization:
                                    kpi = dict()
                                    if settings['objective'] in kpis:
                                        kpi = assets_writer.processMetadata(scylla_output_path, new_path,
                                                                            settings['objective'])
                                        kpi['time_workload_avg'] = assets_writer. \
                                            readResourcesUtilization(scylla_output_path, new_path, 'None')
                                        global_results[folder_iter_id] = (kpi, k)
                                else:
                                    _ = assets_writer.processMetadata(scylla_output_path, new_path, 'None')
                                    _ = assets_writer.readResourcesUtilization(scylla_output_path, new_path, 'None')
                                assets_writer.instancesData(scylla_output_path, new_path)

                        except Exception as e:
                            print('Failed ' + str(e))

                            break
                if not optimization:
                    break
            if optimization:
                # Find global optimal
                min_val = 1000000000000000000
                optimal_key = 0
                max_val = 0
                optimal_val = 0
                optimal_k = 0
                labels = []
                values = []
                unit = 'Seconds' if settings['objective'] in ['flowTime_avg', 'waiting_avg', 'time_workload_avg'] \
                    else 'Price units'
                for k, v in global_results.items():
                    values.append((v[0]))
                    labels.append(float(v[1]))
                    if settings['criteria'] == 'min':
                        if min_val > float(v[0][settings['objective']]):
                            min_val = float(v[0][settings['objective']])
                            optimal_key = k
                            optimal_val = min_val
                            optimal_k = v[1]
                    elif settings['criteria'] == 'max':
                        if max_val < float(v[0][settings['objective']]):
                            max_val = float(v[0][settings['objective']])
                            optimal_key = k
                            optimal_val = max_val
                            optimal_k = v[1]
                print('--- Global optimal', settings['criteria'], 'for', settings['objective'], 'is:',
                      str(optimal_val), unit, 'found in configuration k =', str(optimal_k), 'and in iteration:',
                      str(optimal_key), '---')
                if graph_opti:
                    graph_kpi_path = list(os.path.split(settings['output']))
                    csv_kpi_path = [graph_kpi_path[0], 'kpiResultsTable.csv']
                    csv_kpi_path = os.path.join(*csv_kpi_path)
                    graph_kpi_path = [graph_kpi_path[0], "kpiResultsGraph.png"]
                    graph_kpi_path = os.path.join(*graph_kpi_path)
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    x = np.arange(len(labels))  # the label locations
                    width = 0.25  # the width of the bars
                    pd_results = pd.DataFrame(columns=['K-Value', 'cost_total', 'flowTime_avg', 'waiting_avg',
                                                       'time_workload_avg'])
                    kpi_index = 0
                    for value in values:
                        value['K-Value'] = labels[kpi_index]
                        kpi_index += 1
                        pd_results = pd_results.append(value, ignore_index=True)
                    pd_results.to_csv(csv_kpi_path, sep=',')
                    kpi_index = 0
                    for i in range(0, 2):
                        for j in range(0, 2):
                            lis = []
                            for value in values:
                                if kpi_index == 0:
                                    lis.append(round(float(value[kpis[kpi_index]])))
                                elif kpi_index == 3:
                                    lis.append(float("{0:.2f}".format(float(value[kpis[kpi_index]]))))
                                else:
                                    lis.append(round(float(value[kpis[kpi_index]])/86400))
                            rects1 = axes[i, j].bar(x, lis, width, label=kpis[kpi_index])
                            # Add some text for labels, title and custom x-axis tick labels, etc.
                            axes[i, j].set_ylabel(kpis[kpi_index])
                            t = (kpis[kpi_index] + ' per K Value')
                            axes[i, j].set_title(t)
                            axes[i, j].set_xticks(x)
                            axes[i, j].set_xticklabels(labels)
                            axes[i, j].set_xlabel('K Value')
                            if kpi_index == 0:
                                axes[i, j].set_ylabel('Price units')
                            elif kpi_index == 3:
                                axes[i, j].set_ylabel('Seconds')
                            else:
                                axes[i, j].set_ylabel('Days')
                            axes[i, j].legend()

                            def autolabel(rects, axe):
                                """Attach a text label above each bar in *rects*, displaying its height."""
                                for rect in rects:
                                    height = rect.get_height()
                                    axe.annotate('{}'.format(height),
                                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                                 xytext=(0, 3),  # 3 points vertical offset
                                                 textcoords="offset points",
                                                 ha='center', va='bottom')

                            autolabel(rects1, axes[i, j])
                            fig.tight_layout()
                            kpi_index += 1
                    fig.savefig(graph_kpi_path)
            print("--- Execution total duration", str(time.time() - time_start), "seconds---")
            if resource_optimization:
                print("Resource optimization")
                resourceOptimization.read_parameters(settings['log'],settings['non_repeated_resources'],settings['cost'],
                                                    settings['workload'], settings['flow_time'],settings['waiting_time'], settings['preference_policy'], settings['cooperation_policy'])
                resourceOptimization.main_NSGA2(settings['initial_population'],settings['max_population'],settings['min_population'],settings['generations'])

        elif f == 2:
            time_start = time.time()

            if optimization and settings['objective'] in kpis:
                opti_folder_id = sup.folder_id()
                global_results = dict()
                print("--- Optimization execution started at time:", str(time_start), "---")
            else:
                print("--- Execution started at time:", str(time_start), "---")
            sim_percentage_start = int(settings['sim_percentage'])
            for sim_percentageRaw in range(sim_percentage_start, 110, 10):
                folder_iter_id = 0
                sim_percentage = sim_percentageRaw / 100
                print('sim_percentage ' + str(sim_percentage))
                settings = read_settings(params)
                if optimization:
                    temp = list(os.path.split(settings['output']))
                    folder_iter_id = temp[1]
                    temp = [temp[0], 'optimization_' + opti_folder_id, temp[1]]
                    settings['output'] = os.path.join(*temp)
                # Output folder creation
                if not os.path.exists(settings['output']):
                    os.makedirs(settings['output'])
                    os.makedirs(os.path.join(settings['output'], 'sim_data'))
                copyfile(os.path.join(settings['input'], settings['file']),
                         os.path.join(settings['output'], settings['file']))
                log = lr.LogReader('inputs/'+settings['file'].split('.')[0] + '.xes',
                                   settings['timeformat'])

                bpmn =  br.BpmnReader('inputs/'+settings['file'].split('.')[0] + '.bpmn')
                process_graph = gph.create_process_structure(bpmn)

                print("-- Mining Simulation Parameters --")
                parameters, process_stats = par.extract_parameters(log, bpmn, process_graph,
                                                                   flag=f, k=0,
                                                                   simulator=settings['simulator'].split(","),
                                                                   sim_percentage=sim_percentage,
                                                                   quantity_by_cost=int(settings['quantity_by_cost']),
                                                                   reverse_cost=settings['reverse'],
                                                                   happy_path=happy_path, graph_roles_flag=graph_roles_flag)
                xml_scylla.print_parameters(os.path.join(settings['output'],
                                                         settings['file'].split('.')[0] + '.bpmn'),
                                            os.path.join(settings['output'],
                                                         settings['file'].split('.')[0] + 'Scylla.bpmn'),
                                            parameters['scylla'])

                if simulation:
                    for rep in range(int(settings['repetitions'])):
                        print("Experiment #" + str(rep + 1))
                        try:
                            simulate(settings, rep + 1)
                            if analysis:
                                file_name = settings['file'].split(".")[0]
                                file_global_config_name = file_name + "ScyllaGlobalConfig_resourceutilization.xml"
                                scylla_output_path = os.path.join(settings['output'], "scyllaSim_rep_" + str(rep + 1),
                                                                  file_global_config_name)
                                new_path = os.path.join(settings['output'], "ResourceUtilizationResults")
                                if not os.path.exists(new_path):
                                    os.mkdir(new_path)
                                new_path = os.path.join(new_path, file_name + "_rep" + str(rep + 1))
                                if optimization:
                                    kpi = dict()
                                    if settings['objective'] in kpis:
                                        kpi = assets_writer.processMetadata(scylla_output_path, new_path,
                                                                            settings['objective'])
                                        kpi['time_workload_avg'] = assets_writer. \
                                            readResourcesUtilization(scylla_output_path, new_path, 'None')
                                        global_results[folder_iter_id] = (kpi, sim_percentage)
                                else:
                                    _ = assets_writer.processMetadata(scylla_output_path, new_path, 'None')
                                    _ = assets_writer.readResourcesUtilization(scylla_output_path, new_path, 'None')
                                assets_writer.instancesData(scylla_output_path, new_path)
                        except Exception as e:
                            print('Failed ' + str(e))

                            break
                if not optimization:
                    break

            if optimization:
                # Find global optimal
                min_val = 1000000000000000000
                optimal_key = 0
                max_val = 0
                optimal_val = 0
                optimal_perc = 0
                labels = []
                values = []
                unit = 'Seconds' if settings['objective'] in ['flowTime_avg', 'waiting_avg', 'time_workload_avg'] \
                    else 'Price units'
                for k, v in global_results.items():
                    values.append((v[0]))
                    labels.append(float(v[1]))
                    if settings['criteria'] == 'min':
                        if min_val > float(v[0][settings['objective']]):
                            min_val = float(v[0][settings['objective']])
                            optimal_key = k
                            optimal_val = min_val
                            optimal_perc = v[1]
                    elif settings['criteria'] == 'max':
                        if max_val < float(v[0][settings['objective']]):
                            max_val = float(v[0][settings['objective']])
                            optimal_key = k
                            optimal_val = max_val
                            optimal_perc = v[1]
                print('--- Global optimal', settings['criteria'], 'for', settings['objective'], 'is:',
                      str(optimal_val), unit, 'found in configuration with similitude percentage =', str(optimal_perc),
                      'and in iteration:', str(optimal_key), '---')
                if graph_opti:
                    graph_kpi_path = list(os.path.split(settings['output']))
                    csv_kpi_path = [graph_kpi_path[0], 'kpiResultsTable.csv']
                    csv_kpi_path = os.path.join(*csv_kpi_path)
                    graph_kpi_path = [graph_kpi_path[0], "kpiResultsGraph.png"]
                    graph_kpi_path = os.path.join(*graph_kpi_path)
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    x = np.arange(len(labels))  # the label locations
                    width = 0.25  # the width of the bars
                    pd_results = pd.DataFrame(columns=['Sim_Percentage', 'cost_total', 'flowTime_avg', 'waiting_avg',
                                                       'time_workload_avg'])
                    kpi_index = 0
                    for value in values:
                        value['Sim_Percentage'] = labels[kpi_index]
                        kpi_index += 1
                        pd_results = pd_results.append(value, ignore_index=True)
                    pd_results.to_csv(csv_kpi_path, sep=',')
                    kpi_index = 0
                    for i in range(0, 2):
                        for j in range(0, 2):
                            lis = []
                            for value in values:
                                if kpi_index == 0 or kpi_index == 3:
                                    lis.append(round(float(value[kpis[kpi_index]])))
                                else:
                                    lis.append(round(float(value[kpis[kpi_index]]) / 86400))
                            rects1 = axes[i, j].bar(x, lis, width, label=kpis[kpi_index])
                            axes[i, j].set_ylabel(kpis[kpi_index])
                            t = (kpis[kpi_index] + ' per similitude percentage')
                            axes[i, j].set_title(t)
                            axes[i, j].set_xticks(x)
                            axes[i, j].set_xticklabels(labels)
                            axes[i, j].set_xlabel('Similitude percentage')
                            if kpi_index == 0:
                                axes[i, j].set_ylabel('Price units')
                            elif kpi_index == 3:
                                axes[i, j].set_ylabel('Seconds')
                            else:
                                axes[i, j].set_ylabel('Days')
                            axes[i, j].legend()

                            def autolabel(rects, axe):
                                """Attach a text label above each bar in *rects*, displaying its height."""
                                for rect in rects:
                                    height = rect.get_height()
                                    axe.annotate('{}'.format(height),
                                                 xy=(rect.get_x() + rect.get_width() / 2, height),
                                                 xytext=(0, 3),  # 3 points vertical offset
                                                 textcoords="offset points",
                                                 ha='center', va='bottom')

                            autolabel(rects1, axes[i, j])
                            fig.tight_layout()
                            kpi_index += 1
                    fig.savefig(graph_kpi_path)
                    plt.close()
            if resource_optimization:
                print("Resource optimization")
                resourceOptimization.read_parameters(settings['log'],settings['non_repeated_resources'],settings['cost'],
                                                    settings['workload'], settings['flow_time'],settings['waiting_time'], settings['preference_policy'], settings['cooperation_policy'])
                resourceOptimization.main_NSGA2(settings['initial_population'],settings['max_population'],settings['min_population'],settings['generations'])
            print("--- Execution total duration ", str(time.time() - time_start), " seconds---")


def main(argv):
    space = {
        'epsilon': 1,
        'eta': 1,
        'alg_manag': hp.choice('alg_manag', ['replacement',
                                             'trace_alignment',
                                             'removal'])
    }
    objective(space)

if __name__ == "__main__":
    main(sys.argv[1:])
