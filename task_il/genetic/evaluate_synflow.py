"""
Synflow-based Fitness Evaluation
Uses Synaptic Flow instead of training for fast architecture evaluation
"""
from evo_utils import Utils, GPUTools, Log
import importlib
import time
import os
import sys


class SynflowEvaluate(object):
    """
    Evaluate fitness using Synflow score instead of training.
    Much faster than training-based evaluation.
    """

    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log

    def generate_to_python_file(self):
        """Generate Python files using synflow template"""
        self.log.info('Begin to generate synflow python files')
        for indi in self.individuals:
            Utils.generate_synflow_file(indi)
        self.log.info('Finish the generation of synflow python files')

    def evaluate(self):
        """
        Evaluate fitness using Synflow score.
        Runs sequentially (no multiprocessing) for CUDA compatibility.
        """
        self.log.info('Query synflow fitness from cache')
        _map = Utils.load_cache_data()
        _count = 0
        for indi in self.individuals:
            _key, _str = indi.uuid()
            if _key in _map:
                _count += 1
                _score = _map[_key]
                self.log.info('Hit the cache for %s, key:%s, synflow:%.5f, assigned_score:%.5f'%
                            (indi.id, _key, float(_score), indi.acc))
                indi.acc = float(_score)
        self.log.info('Total hit %d individuals for synflow fitness'%(_count))

        has_evaluated_offspring = False
        for indi in self.individuals:
            if indi.acc < 0:
                has_evaluated_offspring = True
                time.sleep(5)
                gpu_id = GPUTools.detect_available_gpu_id()
                while gpu_id is None:
                    time.sleep(5)
                    gpu_id = GPUTools.detect_available_gpu_id()
                if gpu_id is not None:
                    file_name = indi.id
                    self.log.info('Begin to evaluate synflow for %s'%(file_name))
                    module_name = 'scripts.%s'%(file_name)
                    if module_name in sys.modules.keys():
                        self.log.info('Module:%s has been loaded, delete it'%(module_name))
                        del sys.modules[module_name]
                        _module = importlib.import_module('.', module_name)
                    else:
                        _module = importlib.import_module('.', module_name)
                    _class = getattr(_module, 'RunModel')
                    cls_obj = _class()
                    # Run sequentially (no Process) for CUDA compatibility
                    cls_obj.do_work('%d' % gpu_id, file_name)
                    self.log.info('Finished synflow evaluation for %s'%(file_name))
            else:
                file_name = indi.id
                self.log.info('%s has inherited the synflow fitness as %.5f, no need to evaluate'%
                            (file_name, indi.acc))
                f = open('./populations/after_%s.txt'%(file_name[4:6]), 'a+')
                # Write synflow format
                f.write('%s={synflow:%.5f, raw:0.0, norm:0.0, params:0.0}\n'%(file_name, indi.acc))
                f.flush()
                f.close()

        # Wait for all evaluations to finish and collect results
        if has_evaluated_offspring:
            all_finished = False
            while all_finished is not True:
                has_nums = 0
                time.sleep(5)
                file_name = './populations/after_%s.txt' % (self.individuals[0].id[4:6])
                assert os.path.exists(file_name) is True
                f = open(file_name, 'r')
                for line in f:
                    if len(line.strip()) > 0:
                        has_nums += 1
                f.close()
                if has_nums >= len(self.individuals):
                    all_finished = True

            # Load synflow fitness from files
            file_name = './populations/after_%s.txt'%(self.individuals[0].id[4:6])
            assert os.path.exists(file_name) is True
            f = open(file_name, 'r')
            fitness_map = {}
            for line in f:
                if len(line.strip()) > 0:
                    # Synflow format: indiXXXX={synflow:73.074, raw:..., norm:..., params:...}
                    parts = line.strip().split('=')
                    indi_id = parts[0]

                    if '{' in parts[1]:
                        # Parse synflow metrics dict
                        metrics_str = parts[1]
                        metrics_str = metrics_str.replace('{', '').replace('}', '')
                        metrics_parts = metrics_str.split(',')
                        # Extract synflow score
                        for metric in metrics_parts:
                            key_value = metric.strip().split(':')
                            if len(key_value) == 2:
                                key = key_value[0].strip()
                                value = float(key_value[1].strip())
                                if key == 'synflow':
                                    fitness_map[indi_id] = value
                                    break
            f.close()

            # Assign fitness to individuals
            for indi in self.individuals:
                if indi.acc == -1:
                    if indi.id not in fitness_map:
                        self.log.warn('The individuals have been evaluated, but the records are not correct, '
                                    'the fitness of %s does not exist in %s, wait 120 seconds'%
                                    (indi.id, file_name))
                        time.sleep(5)
                    indi.acc = fitness_map[indi.id]

            # Save to cache
            Utils.save_fitness_to_cache(self.individuals)

            # Write to history
            f = open('./populations/history.txt', 'a+')
            _str = []
            for ind in self.individuals:
                if ind.acc > 0.0:
                    _str.append(str(ind))
                    _str.append('-' * 100)
            f.write('\n'.join(_str))
            f.write('\n')
            f.close()
        else:
            self.log.info('None offspring has been evaluated with synflow')
