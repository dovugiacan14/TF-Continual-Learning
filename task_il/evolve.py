import random
import time
import os
from datetime import datetime

# Auto-detect and change to correct working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != script_dir:
    os.chdir(script_dir)

from evo_utils import StatusUpdateTool, Utils, Log
from genetic.population import Population, Individual
from genetic.mutation import Mutation
import numpy as np
import copy
from genetic.evaluate_synflow import SynflowEvaluate
from genetic.evaluate_zen import ZenEvaluate
from genetic.evaluate_naswot import NaswotEvaluate
from genetic.evaluate_fisher import FisherEvaluate
from genetic.evaluate_gradnorm import GradNormEvaluate
from genetic.evaluate_grasp import GraSPEvaluate
from genetic.evaluate_snip import SnipEvaluate
from genetic.evaluate import FitnessEvaluate

def run_evolve():
    params = {}
    params['pop_size'] = 10   # Population size
    params['max_gen'] = 20    # Maximum number of iteration generations
    params['eval_mode'] = 7     # Evaluation mode: 0=pytorch_train, 1=synflow
    evoCNN = EvolveCNN(params)
    evoCNN.do_work(params)

class EvolveCNN(object):
    def __init__(self, params):
        self.parent_pops = None
        self.params = params
        self.pops = None

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(self.params, 0)
        pops.initialize()
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    # type==0 means 0-th
    def fitness_evaluate(self):
        eval_mode = self.params.get('eval_mode', 0)  # Default to pytorch_train mode

        if eval_mode == 7:
            # use SNIP (connection sensitivity with real data)
            fitness = SnipEvaluate(self.pops.individuals, Log)
            Log.info('Using SNIP evaluation mode (eval_mode=%d)' % eval_mode)
        elif eval_mode == 6:
            # use GraSP (Gradient Signal Preservation with real data)
            fitness = GraSPEvaluate(self.pops.individuals, Log)
            Log.info('Using GraSP evaluation mode (eval_mode=%d)' % eval_mode)
        elif eval_mode == 5:
            # use GradNorm (L2 norm of gradients with random data)
            fitness = GradNormEvaluate(self.pops.individuals, Log)
            Log.info('Using GradNorm evaluation mode (eval_mode=%d)' % eval_mode)
        elif eval_mode == 4:
            # use Fisher Information (activation-gradient product)
            fitness = FisherEvaluate(self.pops.individuals, Log)
            Log.info('Using Fisher evaluation mode (eval_mode=%d)' % eval_mode)
        elif eval_mode == 3:
            # use NASWOT (log-det of activation kernel matrix)
            fitness = NaswotEvaluate(self.pops.individuals, Log)
            Log.info('Using NASWOT evaluation mode (eval_mode=%d)' % eval_mode)
        elif eval_mode == 2:
            # use Zen-NAS (Zen-Score)
            fitness = ZenEvaluate(self.pops.individuals, Log)
            Log.info('Using Zen-NAS evaluation mode (eval_mode=%d)' % eval_mode)
        elif eval_mode == 1:
            # use Synflow
            fitness = SynflowEvaluate(self.pops.individuals, Log)
            Log.info('Using Synflow evaluation mode (eval_mode=%d)' % eval_mode)
        else:
            # use PyTorch training evaluation mode
            fitness = FitnessEvaluate(self.pops.individuals, Log)
            Log.info('Using PyTorch training evaluation mode (eval_mode=%d)' % eval_mode)

        fitness.generate_to_python_file()

        # for indi in self.pops.individuals:
        #     indi.acc = indi.code[1]/indi.code[0]
        # return None

        fitness.evaluate()

    def generate_offspring(self):
        cm = Mutation(Log,self.pops.individuals, _params={'gen_no': self.pops.gen_no})
        offspring = cm.process(mut_offspring_num=self.params['pop_size'])
        
        _str = []
        for ind in offspring:
            _str.append(str(ind))
            _str.append('-' * 100)
        file_name = './populations/offspring_%02d.txt' % self.pops.gen_no
        with open(file_name, 'w') as f:
            f.write('\n'.join(_str))

        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        indi_list = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
        pop_size = self.params['pop_size']
        elitism = 0.2
        e_count = int(pop_size * elitism)
        indi_list.sort(key=lambda x: x.acc, reverse=True)
        # descending order
        next_individuals = indi_list[0:e_count]
        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)

        for i in range(pop_size - e_count):
            idx1 = random.randrange(0, len(left_list))
            indi1 = left_list.pop(idx1)
            idx2 = random.randrange(0, len(left_list))
            indi2 = left_list.pop(idx2)
            if indi1.acc > indi2.acc:
                next_individuals.append(indi1)
            else:
                next_individuals.append(indi2)

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.params, self.pops.gen_no)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops

        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)
        self.pops.gen_no += 1

    def do_work(self, params):
        # Enable quiet mode for Kaggle (reduce terminal output)
        Log.set_quiet_mode(quiet=True)

        max_gen = params['max_gen']
        pop_size = params['pop_size']
        eval_mode = params.get('eval_mode', 0)
        eval_mode_str = {0: 'PyTorch Train', 1: 'Synflow', 2: 'Zen-NAS', 3: 'NASWOT', 4: 'Fisher', 5: 'GradNorm', 6: 'GraSP', 7: 'SNIP'}.get(eval_mode, 'Unknown')

        # START SIGNAL - Important information
        Log.important('='*60)
        Log.important('EVOLUTION STARTED')
        Log.important(f'Mode: {eval_mode_str}')
        Log.important(f'Max Generations: {max_gen}')
        Log.important(f'Population Size: {pop_size}')
        Log.important(f'Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        Log.important('='*60)

        # the step 1
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation'%(gen_no))
                pops = Utils.load_population(prefix='begin', gen_no=gen_no, params=params)
                self.pops = pops
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()

        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(gen_no))
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(gen_no))
        gen_no += 1
        self.pops.gen_no += 1
        for curr_gen in range(gen_no, max_gen):
            # GENERATION PROGRESS - Important information
            Log.important(f'Generation {curr_gen}/{max_gen} completed')

            self.params['gen_no'] = curr_gen
            #step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation'%(curr_gen))
            self.generate_offspring()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation'%(curr_gen))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(curr_gen))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(curr_gen))
            # time.sleep(2)
            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection'%(curr_gen))

        self.params['gen_no'] = max_gen
        self.generate_offspring()

        StatusUpdateTool.end_evolution()

        # DONE SIGNAL - Important information
        Log.important('='*60)
        Log.important('EVOLUTION COMPLETED')
        Log.important(f'End Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        Log.important('='*60)


if __name__ == '__main__':
    run_evolve()
