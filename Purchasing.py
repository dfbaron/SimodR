import numpy as np
import random as rn
import math
import subprocess
import xml.etree.ElementTree as ET 


import os
from os import path
import gc
import time
import json
from support_modules.writers import xml_writer_scylla_resources

import pandas as pd 
import datetime


def get_resource(resource_name,resources_dictionary):
  for id, name in resources_dictionary.items(): 
    if name == resource_name:
        return id


"REVISAR LOS INDICEEEES.Restricciones Qué roles están habilitados para qué actividad->Scylla. CORREO: 1. Cómo se manejan las restricciones sobre recursos Mutación con sentido. 2. Simulador asignacion por recurso o por rol. "

"""
This function generates random initial solutions.

Parameters:
The chromosome length equals the number of activities in the process.
Number of resources to be created
The number of resources must be greater than zero
The number of resources must be greater than the number of activities (chromosome length)

Se puede mejorar teniendo una lista de los que aún no se han metido.
VERIFIED
"""

def create_solutions(chromosome_length, solutions_number,resources_number):
  solutions=np.zeros((solutions_number,chromosome_length))

  for solution in range(solutions_number): 
    list=[]
    i=1
    while i <= chromosome_length:  
        r=rn.randint(1,resources_number)
        if nonrepeated_resources:
            if r not in list: 
              list.append(r)
              i=i+1
        else:
            list.append(r)
            i=i+1
   
    
    solutions[solution,:]=make_feasible(list)
  
    
  return solutions


"""
This function saves the scores for the whole population for all the objectives.

Parameters:
A vector containing the entire solution

VERIFIED
"""
def score_population(population):
  scores=np.zeros((population.shape[0],metrics_number))
  for i in range(len(population)):
    #SIMULAR UNA VEZ
    #Tomar las 5 métricas
    metrics=calculate_fitness(population[i])
    printProgressBar(((i+1)/len(population))*100, 100, prefix = 'Progress:', suffix = 'Complete', length = 100)
    scores[i][0]=float(metrics['cost_average'])*cost
    scores[i][1]=float(metrics['flow_time_average'])*flow_time
    scores[i][2]=float(metrics['waiting_time_average'])*waiting_time
    scores[i][3]=float(metrics['workload_average'])*workload
    scores[i][4]=float(metrics['preference'])*preference_policy
    scores[i][5]=float(metrics['cooperation'])*cooperation_policy
  return scores

"""
This function mutates a number of individuals in the population.

Parameters:
The whole population, the rate of mutation. 

VERIFIED
"""
def randomly_mutate_population(population, mutation_probability):
  mutations_number=math.ceil(len(population)*mutation_probability)
  list=[]
  for i in range(mutations_number):
          r=rn.randint(0,len(population)-1)
          if nonrepeated_resources:
            if r not in list: list.append(r)
          else:
             list.append(r) 
  
  for mutation in list:
    mutation_rn=rn.randint(0,1)
    if(mutation_rn==0):
      new_activity=random_swap(population[mutation])
    else:
        new_activity=random_change(population[mutation])
    population[mutation]=new_activity
  return population
  
"""
This function swaps the reources of a pair of activities. This process is random and under a uniform distribution.

Parameters:
The vector that is going to be mutated. 


VERIFIED
"""
def random_swap(solution):
  
  first_position=rn.randint(0,len(solution)-1)
  second_position=rn.randint(0,len(solution)-1)
  auxiliary_position=solution[first_position]
  solution[first_position]=solution[second_position]
  solution[second_position]=auxiliary_position
  return solution

def random_change(solution):
  activity=rn.randint(0,len(solution)-1)
  resource=rn.randint(1,len(resources_dictionary))
  solution[activity]=resource
  solution=make_feasible(solution)
  
  return solution


"""
Both parents must match in length

"""
def breed_by_crossover(parent_1,parent_2):
  chromosome_length=len(parent_1)
  crossover_point=rn.randint(0,(chromosome_length-1))
  child=np.hstack((parent_1[0:crossover_point],parent_2[crossover_point:]))
  i=crossover_point
  if nonrepeated_resources:
   while i<chromosome_length:
     if child[i] in child[0:i]:
       new_resouce=get_pmx_resource()
       resource_in_list=True
       while(resource_in_list):
         if(new_resouce in child[0:i]):
           new_resouce=get_pmx_resource()
         else:
           resource_in_list=False
           child[i]=new_resouce    
     i=i+1
  return child



def get_pmx_resource():

  return rn.randint(1,len(resources_dictionary))
  #r= rn.random()
  #for i in range(len(pmx[:,])):
    #if(pmx[i,metrics_number+2]>=r):
     # return pmx[i-1,1]
#Ser elitista

def breed_population(population):
  new_population=[]
  global hash_dictionary
  global fronts_dictionary
  population_size=population.shape[0]
  
  for i in range(int(population_size/2)):
      print("------Crossover between two parents------")
      #Van a venir de una distribución lognormal sigma=1, miu=0
      selected_front1=int(round(np.random.lognormal(1,0)))
      if selected_front1>len(fronts_dictionary)-1:
          selected_front1=len(fronts_dictionary)-1
      selected_front2=int(round(np.random.lognormal(1,0)))
      if selected_front2>len(fronts_dictionary)-1:
          selected_front2=len(fronts_dictionary)-1
     # selected_front3=rn.randint(0,len(fronts_dictionary)-1)
      
      
      
      selected_parent1=rn.randint(0,len(fronts_dictionary[selected_front1])-1)
      selected_parent2=rn.randint(0,len(fronts_dictionary[selected_front2])-1)
      #selected_parent3=rn.randint(0,len(fronts_dictionary[selected_front3])-1)
      selected_parent3=population[rn.randint(0, population_size-1)]
      selected_parent4=population[rn.randint(0, population_size-1)]
      
      
      solution_hash1=hash(str(population[selected_parent1]))
      solution_hash2=hash(str(population[selected_parent2]))
      metrics_parent1_string=hash_dictionary[solution_hash1]
      
      metrics_parent1= [float(numeric_string) for numeric_string in metrics_parent1_string]

      metrics_parent2_string=hash_dictionary[solution_hash2]
      metrics_parent2= [float(numeric_string) for numeric_string in metrics_parent2_string]
      
   
     
      parent_1=population[selected_parent2]
      if all([metrics_parent1 >=metrics_parent2]) and any([metrics_parent1> metrics_parent2]) :
         # j dominates i. Label 'i' point as not on Pareto front
         parent_1= population[selected_parent1]
         # Stop further comparisons with 'i' (no more comparisons needed)
      print("Created parent 1 ")
      solution_hash3=hash(str(selected_parent3))
      solution_hash4=hash(str(selected_parent4))
      metrics_parent3_string=hash_dictionary[solution_hash3]
      metrics_parent3= [float(numeric_string) for numeric_string in metrics_parent3_string]

      metrics_parent4_string=hash_dictionary[solution_hash4]
      metrics_parent4= [float(numeric_string) for numeric_string in metrics_parent4_string]
      
      parent_2=selected_parent4
      if all([metrics_parent3 >= metrics_parent4]) and any([metrics_parent3 > metrics_parent4]):
         # j dominates i. Label 'i' point as not on Pareto front
         parent_2= selected_parent3
         # Stop further comparisons with 'i' (no more comparisons needed)
         
      print("Created parent 2 ")
      
      child_1 = make_feasible(breed_by_crossover(parent_1,parent_2))
      print("Created child 1 ")

      child_2 = make_feasible(breed_by_crossover(parent_2,parent_1))
      print("Created child 2 ")
      new_population.append(child_1)
      new_population.append(child_2)
      print("Both childs added to population")
      
  population = np.vstack((population, np.array(new_population)))
  population = np.unique(population, axis=0)
  return population



""" La función declarada a continuación fue tomado de: https://pythonhealthcare.org/2019/01/17/117-genetic-algorithms-2-a-multiple-objective-genetic-algorithm-nsga-ii/
Asimismo, el esqueleto del algoritmo está basado en esa solución."""


def calculate_crowding(scores):
    """
    Crowding is based on a vector for each individual
    All scores are normalised between low and high. For any one score, all
    solutions are sorted in order low to high. Crowding for chromsome x
    for that score is the difference between the next highest and next
    lowest score. Total crowding value sums all crowding for all scores
    """
    
    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    normed_scores = (scores - scores.min(0)) / scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])

        sorted_scores_index = np.argsort(normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1:population_size - 1] = \
            (sorted_scores[2:population_size] -
             sorted_scores[0:population_size - 2])

        # resort to orginal order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances

""" La función declarada a continuación fue tomado de: https://pythonhealthcare.org/2019/01/17/117-genetic-algorithms-2-a-multiple-objective-genetic-algorithm-nsga-ii/
Asimismo, el esqueleto del algoritmo está basado en esa solución."""
def reduce_by_crowding(scores, number_to_select):
    """
    This function selects a number of solution d on tournament of
    crowding distances. Two members of the population are picked at
    random. The one with the higher croding dostance is always picked
    """    
    population_ids = np.arange(scores.shape[0])

    crowding_distances = calculate_crowding(scores)

    picked_population_ids = np.zeros((number_to_select))

    picked_scores = np.zeros((number_to_select, len(scores[0, :])))

    for i in range(number_to_select):

        population_size = population_ids.shape[0]

        fighter1ID = rn.randint(0, population_size - 1)

        fighter2ID = rn.randint(0, population_size - 1)

        # If fighter # 1 is better
        if crowding_distances[fighter1ID] >= crowding_distances[fighter2ID]:

            # add solution to picked solutions array
            picked_population_ids[i] = population_ids[fighter1ID]

            # Add score to picked scores array
            picked_scores[i, :] = scores[fighter1ID, :]

            # remove selected solution from available solutions
            population_ids = np.delete(population_ids, (fighter1ID),axis=0)

            scores = np.delete(scores, (fighter1ID), axis=0)

            crowding_distances = np.delete(crowding_distances, (fighter1ID),axis=0)
        else:
            picked_population_ids[i] = population_ids[fighter2ID]

            picked_scores[i, :] = scores[fighter2ID, :]

            population_ids = np.delete(population_ids, (fighter2ID), axis=0)

            scores = np.delete(scores, (fighter2ID), axis=0)

            crowding_distances = np.delete(
                crowding_distances, (fighter2ID), axis=0)

    # Convert to integer 
    picked_population_ids = np.asarray(picked_population_ids, dtype=int)
    
    return (picked_population_ids)

""" La función declarada a continuación fue tomado de: https://pythonhealthcare.org/2019/01/17/117-genetic-algorithms-2-a-multiple-objective-genetic-algorithm-nsga-ii/
Asimismo, el esqueleto del algoritmo está basado en esa solución."""
def identify_pareto(scores, population_ids):

    """
    Identifies a single Pareto front, and returns the population IDs of
    the selected solutions.
    """
    
    population_size = scores.shape[0]
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

""" La función declarada a continuación fue tomado de: https://pythonhealthcare.org/2019/01/17/117-genetic-algorithms-2-a-multiple-objective-genetic-algorithm-nsga-ii/
Asimismo, el esqueleto del algoritmo está basado en esa solución."""
def build_pareto_population(population, scores, minimum_population_size, maximum_population_size):

    """
    As necessary repeats Pareto front selection to build a population within
    defined size limits. Will reduce a Pareto front by applying crowding 
    selection as necessary.    
    """
    unselected_population_ids = np.arange(population.shape[0])
    all_population_ids = np.arange(population.shape[0])
    pareto_front = []
    i=0
    while len(pareto_front) < minimum_population_size:
        temp_pareto_front = identify_pareto(scores[unselected_population_ids, :], unselected_population_ids)
        
        # Check size of total parteo front. 
        # If larger than maximum size reduce new pareto front by crowding
        combined_pareto_size = len(pareto_front) + len(temp_pareto_front)
        if combined_pareto_size > maximum_population_size:
            number_to_select = combined_pareto_size - maximum_population_size
            selected_individuals = (reduce_by_crowding(scores[temp_pareto_front], number_to_select))
            temp_pareto_front = temp_pareto_front[selected_individuals]
        
        # Add latest pareto front to full Pareto front
        
        pareto_front = np.hstack((pareto_front, temp_pareto_front))
        global fronts_dictionary
        fronts_dictionary[i]=temp_pareto_front
        i=i+1
        
        # Update unselected population ID by using sets to find IDs in all
        # ids that are not in the selected front
        unselected_set = set(all_population_ids) - set(pareto_front)
        unselected_population_ids = np.array(list(unselected_set))

    population = population[pareto_front.astype(int)]
    return population


"Number of metrics in the multiobjective function"
def main_NSGA2(initial_population, max_population, min_population, generations):
  gc.enable()
  chromosome_length = len(activities_dictionary)
  #tiene que ser el numero de actividades
  mutation_probability=0.05
  print(log)
  print(nonrepeated_resources)
  starting_population_size = int(initial_population)
  maximum_generation = int(generations)
  minimum_population_size = int(min_population)
  maximum_population_size =int( max_population)
  resources_number=len(resources_dictionary)
  
  population = create_solutions( chromosome_length,starting_population_size, resources_number)
  
  for generation in range(maximum_generation):
  #while convergence<=10:  
     
      print('Generation:'+str(generation)+ ' of: '+str(maximum_generation))
      print('****************************************************************************************************')
      print('****************************************************************************************************')
      print('*******         ***   ****   *************  ****           ****           **********          ******')
      print('******          ***   ****     ********     ***   *******   ***   *******   ********  ********  ****')
      print('******    *********   ****   ***  *   ***   ***   *******   ***   *******   ********  ********  ****')
      print('******   **********   ****   ****   *****   ***   *******   ***   *******   ********  ********  ****')
      print('******    *********   ****   ************   ***   *******   ***   *******   ********  ******  ******')
      print('*******        ****   ****   ************   ***   *******   ***   *******   ********         *******')
      print('************    ***   ****   ************   ***   *******   ***   *******   ********   *   *********')
      print('*************   ***   ****   ************   ***   *******   ***   *******   ********   **   ********')
      print('************    ***   ****   ************   ***   *******   ***   *******   ********   ***   *******')
      print('******          ***   ****   ************   ***   *******   ***   *******   ********   *****   *****')
      print('******         ****   ****   ************   ****           ****           **********   *******   ***')
      print('****************************************************************************************************')
      print('****************************************************************************************************')
      printProgressBar(0, 100, prefix = 'Progress:', suffix = 'Complete', length = 100)

      
      #Factibilidad // Viabilidad
      
      
      
      randomly_mutate_population(population,mutation_probability) 
     

      # Score population
      scores = score_population(population)
      len_unique_scores = len(np.unique(scores, axis=0))
      len_scores=len(scores)
      mutation_probability=0.2*(1-len_unique_scores/len_scores)
      
      print("Scoring completed. Starting building Pareto front")
      # Build pareto front
      population = build_pareto_population(population, scores, minimum_population_size, maximum_population_size)
      
             
      
      # Breed
      print("Starting breeding...")
      population = breed_population(population)
      print("Breeding completed. Starting scoring")
      global fronts_dictionary

      fronts_dictionary={}
      print("Built pareto front")
      
              
      
      gc.collect()
      
      
  print("Finished generations")
  for solution in range(len(population)):
      population[solution]=make_feasible(population[solution])
  scores = score_population(population)
  population_ids = np.arange(population.shape[0]).astype(int)
  pareto_front = identify_pareto(scores, population_ids)
  population = population[pareto_front, :]
  
  scores = scores[pareto_front]
  print(scores)
  print("Population")
  print(population)
  global output_date
  #
  
  np.savetxt('outputs/'+output_date+'/'+log+"_solutions.csv", population, delimiter=",")
  np.savetxt('outputs/'+output_date+'/'+log+"_scores.csv", scores, delimiter=",")
  
  

  
  args = [rpath, '--vanilla', 'support_modules/plot_generation/PostProcessing.R','outputs/'+output_date+'/'+log]
  subprocess.call(args,shell=True,stdout=subprocess.PIPE)
  
  with open('outputs/'+output_date+'/'+log+'_resources_dictionary.json', 'w') as file:
    json.dump(resources_dictionary, file, indent=4, sort_keys=True,
              separators=(', ', ': '), ensure_ascii=False,
              default=myconverter)
  with open('outputs/'+output_date+'/'+log+'_activities_dictionary.json', 'w') as file:
    json.dump(activities_dictionary, file, indent=4, sort_keys=True,
              separators=(', ', ': '), ensure_ascii=False,
              default=myconverter)
  
  with open('utilities/'+log+'_hash.json', 'w', encoding='utf-8') as f:
        json.dump(hash_dictionary, f, ensure_ascii=False, indent=4)
  
    
  
  

def myconverter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

        
def make_feasible(solution):
   
    for i in range(len(solution)):
        activity_id=activities_dictionary[i+1]
        resource_id=resources_dictionary[solution[i]]
        resource_name_map=name_id_resources_dictionary[resource_id]
    
        activity_name_map=name_id_activities_dictionary[activity_id]
       
                
                
        global incidence_matrix
        if resource_name_map!='SYSTEM':
         if incidence_matrix.at[activity_name_map,resource_name_map]==0:
                repaired=False
                while not repaired:
                   
                     resource_randid=rn.randint(1,resources_number)
                     resource_id=resources_dictionary[resource_randid]
                     resource_name_map=name_id_resources_dictionary[resource_id]
                   # print(activity_name_map+":"+resource_name_map)
                     if resource_name_map=='SYSTEM':
                        repaired=True 
                     elif incidence_matrix.at[activity_name_map,resource_name_map]>0:
                        solution[i]=resource_randid
                        
                        repaired=True
                        
    return solution

"""Usage: Scylla --config=<your config file> --bpmn=<your first bpmn file> [--bpmn=<your second bpmn file>] [--bpmn=...] --sim=<your first sim file> [--sim=<your second sim file>] [--sim=...] [--output=<your output path>]"""   
""" INSTANCIAS MANUALES O AUTO. REVISAR EFFECTIVE
Manejo recursos
Simulador"""




def calculate_fitness(solution):
  
  starting_time = int(round(time.time() * 1000))
  solution_hash=hash(str(solution))
  results_dictionary={}
  global hash_dictionary
  if solution_hash in hash_dictionary:
      
      
      cost_average= hash_dictionary[solution_hash][0]
      flow_time_average=hash_dictionary[solution_hash][1]
      waiting_time_average=hash_dictionary[solution_hash][2]
      workload_average=hash_dictionary[solution_hash][3]
      preference=hash_dictionary[solution_hash][4]
      cooperation=hash_dictionary[solution_hash][5]
      
      results_dictionary={
          "waiting_time_average":waiting_time_average,
          "flow_time_average":flow_time_average,
          "cost_average":cost_average,
          "workload_average":workload_average,
          "preference": preference,
          "cooperation": cooperation
       }
      
  else:
    #   print('SIMULATION STARTED at:'+str(starting_time))
       
       output_file_name="scylla_results_"+log
      
        
       if os.path.exists('scylla_results_'):
           os.rmdir("scylla_results_")
    
       if os.path.exists(output_file_name+"ScyllaGlobalConfig_resourceutilization.xml"):
           os.remove(output_file_name+"ScyllaGlobalConfig_resourceutilization.xml")
    
       if os.path.exists(output_file_name+".xes"):
           os.remove(output_file_name+".xes")
    
       if os.path.exists(output_file_name+"ScyllaGlobalConfig_batchactivitystats.txt"):
           os.remove(output_file_name+"ScyllaGlobalConfig_batchactivitystats.txt")    
  
       i=0
       if str(solution)!="BASELINE":
         
         for child in activities:
           if child.get('name')!="Start" and child.get('name')!="End" :
               child.find('{http://bsim.hpi.uni-potsdam.de/scylla/simModel}resources')[0].set('id',resources_dictionary[int(solution[i])])
               i=i+1
       
           
       activities_tree.write('inputs/'+log+'ScyllaSimuConfig.xml')
      
       args = ['java', '-jar', 'external_tools/ScyllaNew/Scylla_V6.jar',"--config="+'inputs/'+log+"ScyllaGlobalConfig.xml","--bpmn="+'inputs/'+log+".bpmn","--sim="+'inputs/'+log+"ScyllaSimuConfig.xml","--output=scylla_results_","--enable-bps-logging"]
       subprocess.call(args,shell=True,stdout=subprocess.PIPE)
       duration= int(round(time.time() * 1000))-starting_time
     #  print('SIMULATION FINISHED, duration='+str(duration))

       
       starting_time = int(round(time.time() * 1000))
       output_tree =ET.parse(output_file_name+"ScyllaGlobalConfig_resourceutilization.xml")
       output_root=output_tree.getroot()
       resources_output_metrics=output_root.find('resources')
       accumulative_workload=0
       for child in resources_output_metrics:
           accumulative_workload=accumulative_workload+float(child.find('time').find('workload').find('avg').text)
          # print(accumulative_workload)
       output_metrics=output_root[1][0]
       
       cost_average=output_metrics.find('cost').find('avg').text
       flow_time_average=output_metrics.find('time').find('flow_time').find('avg').text
       waiting_time_average=output_metrics.find('time').find('waiting').find('avg').text
       workload_average=accumulative_workload/resources_number
       preference=calculate_preference(solution)
       cooperation=calculate_cooperation(solution)
       
       results_dictionary={
        "waiting_time_average":waiting_time_average,
        "flow_time_average":flow_time_average,
        "cost_average":cost_average,
        "workload_average":workload_average,
        "preference": preference,
        "cooperation": cooperation
       }
       
       if str(solution)!="BASELINE":
           hash_dictionary[solution_hash]=[cost_average,flow_time_average,waiting_time_average,workload_average,preference,cooperation]
       else:
           hash_dictionary[solution_hash]=[cost_average,flow_time_average,waiting_time_average,workload_average,str(0),str(0)]

       os.remove(output_file_name+"ScyllaGlobalConfig_resourceutilization.xml")
       os.remove(output_file_name+".xes")
       os.remove(output_file_name+"ScyllaGlobalConfig_batchactivitystats.txt")
       os.rmdir("scylla_results_")
      
          
  return results_dictionary

def repair_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    prev = None
    
    def elements_equal(e1, e2): 
        if type(e1) != type(e2):
            return False
        if e1.tag != e2.tag: return False
        if e1.tag=='{http://bsim.hpi.uni-potsdam.de/scylla/simModel}timetable' and e2.tag=='{http://bsim.hpi.uni-potsdam.de/scylla/simModel}timetable':
         if e1.get('id')==e2.get('id'):
           return True
        if e1.text != e2.text: return False
        if e1.tail != e2.tail: return False
        if e1.attrib != e2.attrib: return False
        if len(e1) != len(e2): return False
        return all([elements_equal(c1, c2) for c1, c2 in zip(e1, e2)])
    
    for page in root:                     # iterate over pages
        elems_to_remove = []
        first_elem=page
        for elem in page:
            
            first=True
            if elements_equal(elem, prev):
                first=False
                
                first_elem.append(list(elem)[0])
                elems_to_remove.append(elem)
                continue
            prev = elem
            if(first):
              first_elem=elem
              
        for elem_to_remove in elems_to_remove:
    
            page.remove(elem_to_remove)
# [...]
    tree.write(path)


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def calculate_preference(solution):
   total_sum=0
   if str(solution)!="BASELINE":
    for i in range(len(solution)):
        activity_id=activities_dictionary[i+1]
        resource_id=resources_dictionary[solution[i]]
        resource_name_map=name_id_resources_dictionary[resource_id]
    
        activity_name_map=name_id_activities_dictionary[activity_id]
       
                
                
        global incidence_matrix
        if resource_name_map!='SYSTEM':
         #if incidence_matrix.at[activity_name_map,resource_name_map]==0:
                   
                   # print(activity_name_map+":"+resource_name_map)
                  num_repetitions=incidence_matrix.at[activity_name_map,resource_name_map]
                  row_sum=sum(incidence_matrix.loc[activity_name_map,:])
                  column_sum=sum(incidence_matrix.loc[:,resource_name_map])
                  if row_sum!=0 and column_sum!=0:
                   total_sum=total_sum+num_repetitions/row_sum*num_repetitions/column_sum
                  
    return str(total_sum)

def calculate_cooperation(solution):
   total_sum=0
   if str(solution)!="BASELINE":
    for i in range(len(solution)-1):
        
        first_resource_id=resources_dictionary[solution[i]]
        second_resource_id=resources_dictionary[solution[i+1]]
        first_resource_name_map=name_id_resources_dictionary[first_resource_id]
        second_resource_name_map=name_id_resources_dictionary[second_resource_id]
        
       
                
                
        global cooperation_matrix
        if first_resource_name_map!='SYSTEM' and second_resource_name_map!='SYSTEM':
         #if incidence_matrix.at[activity_name_map,resource_name_map]==0:
                   
                   # print(activity_name_map+":"+resource_name_map)
                  
                  num_repetitions=cooperation_matrix.at[first_resource_name_map, second_resource_name_map]
                  #num_repetitions=cooperation_matrix.loc[first_resource_name_map, second_resource_name_map].values[0]
                  total_sum=total_sum+num_repetitions
                  
    return str(total_sum)
        
"""---------------------------------------- SANDBOX ----------------------------------------------------"""
#Generalizar las métricas, arreglando el PMX
#tRATAR DE HACER MÁS RÁPIDO
#Hacer interfaz
#Parar cuando converge
#Exportar un csv
#Relajar las restricciones

logs=['Production','PurchasingExample']
rpath='C:/Program Files/R/R-4.0.3/bin/Rscript'


def read_parameters(input_log, input_nonrepeated_resources,input_cost,input_workload,input_flow_time,input_waiting_time,input_preference_policy,input_cooperation_policy):
                    global log
                    log=input_log
                    #Para minimizar poner -1, para maximizxar 1y pa no hacer nada 0
                    global nonrepeated_resources
                    nonrepeated_resources=input_nonrepeated_resources
                    global cost
                    cost=int(input_cost)
                    global workload
                    workload=int(input_workload)
                    global flow_time
                    flow_time=int(input_flow_time)
                    global waiting_time
                    waiting_time=int(input_waiting_time)
                    global metrics_number
                    metrics_number=6
                    global preference_policy
                    preference_policy=int(input_preference_policy)
                    global cooperation_policy
                    cooperation_policy=int(input_cooperation_policy)
                   
                   
                    
                    json_file=open('inputs/'+log+'_canon.json')
                    canon_model=json.load(json_file)
                    dist_dictionary={
                        'EXPONENTIAL':'exponentialDistribution',
                        'GAMMA':'normalDistribution',
                        'NORMAL':'normalDistribution',
                        'FIXED':'constantDistribution',
                        'BINOMIAL':'binomialDistribution',
                        'ERLANG':'erlangDistribution',
                        'EXPONENTIAL':'exponentialDistribution',
                        'POISSON':'poissonDistribution',
                        'TRIANGULAR':'triangularDistribution',
                        'UNIFORM':'uniformDistribution',
                        'LOGNORMAL':'normalDistribution'
                    }
                
                    for activity in canon_model['elements_data']:
                      if(str(activity['resource'])=='nan'):
                        #Cuando hay NaN, seleccionamos el primer recurso del pool ¿Por qué?  no sé :(
                        activity['resource']=canon_model['resource_pool'][0]['id']
                      activity['type']=dist_dictionary[activity['type']]
                      if activity['name']=="Start":
                        start_id=activity['id']
                        del activity
                        break
                      if activity['name']=="End":
                        del activity
                        
                    canon_model['arrival_rate']["startEventId"]=start_id
                    canon_model['arrival_rate']['dname']=dist_dictionary[canon_model['arrival_rate']['dname']]
                    print('Parsing JSON...')
                    resources_list=[]
                    for role in canon_model['resource_pool']:
                        
                      if role['name']!='SYSTEM':
                          #TODO Use real costs instead of random
                          
                          role['instances']=[]
                          
                          i=0
                          
                          for resource in canon_model['rol_user'][role['name']]:
                               
                              if i==0:
                                  actual_resource={
                                  #TODO Use real costs instead of random
                                  'avg_costxhour':role['costxhour'],
                                  'instances':[],
                                  'id':role['id'],
                                  'name':resource,
                                  'timetable_id':role['timetable_id'],
                                  'total_amount':'1'
                                  }
                              else:
                                  actual_resource={
                                  #TODO Use real costs instead of random
                                  'avg_costxhour':role['costxhour'],
                                  'instances':[],
                                  'id':role['id']+str(i),
                                  'name':resource,
                                  'timetable_id':role['timetable_id'],
                                  'total_amount':'1'
                                  }
                              
                              i=i+1
                              resources_list.append(actual_resource)
                          del role
                      else:
                          
                         actual_resource={
                                  #TODO Use real costs instead of random
                                  'avg_costxhour':role['costxhour'],
                                  'instances':[],
                                  'id':role['id'],
                                  'name':role['name'],
                                  'timetable_id':role['timetable_id'],
                                  'total_amount':role['total_amount']
                                  }
                         resources_list.append(actual_resource)
                          
                    canon_model['resource_pool']=resources_list
                      
                    
                    bpmn_tree = ET.parse('inputs/'+log+'.bpmn')
                    bpmn_root=bpmn_tree.getroot()
                
                    bpmn_id=bpmn_root[0].attrib.get('id')
                    canon_model['bpmnId']=bpmn_id
                    canon_model['time_table']
                    
                    table_array=[]
                    
                    for table in canon_model['time_table']["http://www.qbp-simulator.com/Schema201212:timetables"]["http://www.qbp-simulator.com/Schema201212:timetable"]:
                      if table['@name']=="Discovered_CASES_ARRIVAL_CALENDAR":
                      #Cómo hago con los descubiertos?
                       if isinstance(table['http://www.qbp-simulator.com/Schema201212:rules']['http://www.qbp-simulator.com/Schema201212:rule'], list):
                       #   print(table['http://www.qbp-simulator.com/Schema201212:rules']['http://www.qbp-simulator.com/Schema201212:rule'])
                        for time_item in table['http://www.qbp-simulator.com/Schema201212:rules']['http://www.qbp-simulator.com/Schema201212:rule']:
                          
                          actual_dictionary={
                          'id_t':table['@id'],
                          'default':table['@default'],
                          'name':table['@name'],
                          'from_t':time_item['@fromTime'],
                          'to_t':time_item['@toTime'],
                          'from_w':time_item['@fromWeekDay'],
                          'to_w':time_item['@toWeekDay']
                          }
                          table_array.append(actual_dictionary) 
                      else:
                        actual_dictionary={
                          'id_t':table['@id'],
                          'default':table['@default'],
                          'name':table['@name'],
                          'from_t':table['http://www.qbp-simulator.com/Schema201212:rules']['http://www.qbp-simulator.com/Schema201212:rule']['@fromTime'],
                          'to_t':table['http://www.qbp-simulator.com/Schema201212:rules']['http://www.qbp-simulator.com/Schema201212:rule']['@toTime'],
                          'from_w':table['http://www.qbp-simulator.com/Schema201212:rules']['http://www.qbp-simulator.com/Schema201212:rule']['@fromWeekDay'],
                          'to_w':table['http://www.qbp-simulator.com/Schema201212:rules']['http://www.qbp-simulator.com/Schema201212:rule']['@toWeekDay']
                          }
                      table_array.append(actual_dictionary)
                    
                    del canon_model['time_table']
                    canon_model['time_table']=table_array
                    
                    with open('inputs/'+log+'_modified.json', 'w', encoding='utf-8') as f:
                        json.dump(canon_model, f, ensure_ascii=False, indent=4)
                    
                    xml_writer_scylla_resources.print_parameters('inputs/'+log+'Scylla',canon_model)
                    repair_xml('inputs/'+log+'ScyllaGlobalConfig.xml')
                    
                    
    

                    #pmx=np.loadtxt(open("NSGA2_RESOURCES_PMX.csv"),delimiter=",", skiprows=1)
                    resources_tree = ET.parse('inputs/'+log+'ScyllaGlobalConfig.xml')
                    resources_root=resources_tree.getroot()
                    resources=resources_root[3].findall('{http://bsim.hpi.uni-potsdam.de/scylla/simModel}dynamicResource')
                    global hash_dictionary
                    hash_dictionary={}
                    print("--------------------- Looking for existent solutions from file -----------------------")
                    try:
                     with open('utilities/'+log+'_hash.json') as json_file:
                       print(" Importing solutions...")
                       hash_dictionary = json.load(json_file)
                       
                    except IOError:
                        print("------------------------- No previous results --------------------------")
                     
                    global fronts_dictionary
                    fronts_dictionary={}
                    global resources_number
                    resources_number=len(resources)
                    i=1
                    global resources_dictionary
                    resources_dictionary={}
                    global name_id_resources_dictionary
                    name_id_resources_dictionary={}
                    for child in resources:
                      resources_dictionary[i]=child.get('id')
                      name_id_resources_dictionary[child.get('id')]=child.get('name')
                      i=i+1
                    global activities_tree
                    activities_tree = ET.parse('inputs/'+log+'ScyllaSimuConfig.xml')
                    activities_root=activities_tree.getroot()
                    global activities
                    activities=activities_root[0].findall('{http://bsim.hpi.uni-potsdam.de/scylla/simModel}task')
                    activities_number=len(activities)
                    i=1
                    activities_root=activities_tree.getroot()
                    activities=activities_root[0].findall('{http://bsim.hpi.uni-potsdam.de/scylla/simModel}task')
                    
                    i=1
                    global activities_dictionary
                    activities_dictionary={}
                    global allocation_dictionary
                    allocation_dictionary={}
                    global name_id_activities_dictionary
                    
                    name_id_activities_dictionary={}
                    for child in activities:
                        if child.get('name')!="Start" and child.get('name')!="End" :
                          activities_dictionary[i]=child.get('id')
                          name_id_activities_dictionary[child.get('id')]=child.get('name')
                          resource_name=child.find('{http://bsim.hpi.uni-potsdam.de/scylla/simModel}resources')[0].get('id')
                          allocation_dictionary[i]=get_resource(resource_name, resources_dictionary)
                          i=i+1
                    activities_number=len(activities_dictionary)
                    
                    
                    
                    args = [rpath, '--vanilla', 'support_modules/matrix_processing/PreProcessing.R','inputs/'+log]
                    subprocess.call(args,shell=True,stdout=subprocess.PIPE)
                    global incidence_matrix
                    incidence_matrix = pd.read_csv('inputs/'+log+'_incidenceMatrix.csv', index_col=0)
                    
                    
                    
                    if not path.exists('inputs/'+log+'_cooperationMatrix.csv'):
                        print('Calculating cooperation. This could take few minutes.')
                        args = [rpath, '--vanilla', 'support_modules/matrix_processing/CooperationPolicy.R','inputs/'+log]
                        process = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
                        stdout, stderr = process.communicate()
                        
                        #subprocess.call(args,shell=True,stdout=subprocess.PIPE)
                    global cooperation_matrix
                    cooperation_matrix = pd.read_csv('inputs/'+log+'_cooperationMatrix.csv', index_col=0)
                    cooperation_matrix.index = cooperation_matrix.index.map(str)
                    global output_date
                    output_date=datetime.datetime.now().strftime("%B%d%Y%H%M%S")

                    baseline_output=[0,0,0,0,0,0]
                    baseline_metrics=calculate_fitness('BASELINE')
                    baseline_output[0]=float(baseline_metrics['cost_average'])
                    baseline_output[1]=float(baseline_metrics['flow_time_average'])
                    baseline_output[2]=float(baseline_metrics['waiting_time_average'])
                    baseline_output[3]=float(baseline_metrics['workload_average'])                   
                    
                    os.mkdir('outputs/'+output_date)
                    np.savetxt('outputs/'+output_date+'/'+log+"_baseline_scores.csv", baseline_output, delimiter=",")
                    
#read_parameters('PurchasingExample',False,-1,0, 0,0, 0, 0)
#main_NSGA2(100,700,100,50)

read_parameters('PurchasingExample',False,0,1, 0,0, 0, 0)
main_NSGA2(100,400,100,40)

read_parameters('PurchasingExample',False,0,0, -1,0, 0, 0)
main_NSGA2(100,400,100,40)

read_parameters('PurchasingExample',False,0,0,0,-1, 0, 0)
main_NSGA2(100,400,100,40)

read_parameters('PurchasingExample',False,-1,1, -1,-1, 0, 0)
main_NSGA2(100,400,100,40)


#main_NSGA2('PurchasingExample',settings['max_population'],settings['min_population'],settings['generations'])
