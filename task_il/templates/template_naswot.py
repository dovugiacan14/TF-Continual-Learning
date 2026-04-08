"""
NASWOT Template for Neural Architecture Evaluation
Uses NASWOT (Neural Architecture Search Without Training) to estimate network quality.

Reference: 'Neural Architecture Search without Training' (Mellor et al., ICML 2021)
Algorithm: Score = log(det(K)) where K is the kernel matrix built from ReLU activation patterns.

Note: This version runs sequentially without multiprocessing (CUDA compatible)
"""
from __future__ import print_function
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.arch_craft import Net
from model_code import init_code
import copy
from config import SEED
import multiprocessing  # Required for RunModel interface compatibility
import random  # For seed setting

Inc_cls = 5    # Number of classes incremented at each step

# Auto-detect and change to correct working directory (task_il/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# If script is in task_il/scripts/, go up one level to task_il/
if script_dir.endswith('scripts'):
    script_dir = os.path.dirname(script_dir)
if os.getcwd() != script_dir:
    os.chdir(script_dir)

#generated_code
class NaswotEvaluator(object):
    def __init__(self):
        self.code = code
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.inc = Inc_cls

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def calculate_naswot_score(self, net, batch_size=64, repeat=1):
        """
        Calculate NASWOT score following the original paper (Mellor et al., ICML 2021).

        Algorithm:
        1. For each repeat:
           a. Generate random input batch
           b. Custom forward pass through network
           c. At each ReLU activation, build kernel matrix from binary patterns:
              K += x @ x^T + (1-x) @ (1-x)^T
              where x = (pre_activation > 0).float()
           d. Score = log(det(K))
        2. Return mean score across repeats

        Key insight: The log-determinant of the kernel matrix measures the diversity
        of activation patterns across the batch. Higher diversity = better architecture.

        Note: We use a custom forward pass (instead of hooks) because arch_craft.py
        uses F.relu() (functional) rather than nn.ReLU modules. This lets us capture
        the exact pre-ReLU activations at each layer.
        """
        net.eval()

        scores = []
        with torch.no_grad():
            for _ in range(repeat):
                # Initialize kernel matrix (batch_size x batch_size)
                K = np.zeros((batch_size, batch_size))

                # Random Gaussian input (no real data needed for zero-shot evaluation)
                x = torch.randn(batch_size, 3, 32, 32).cuda()

                # === Custom forward pass to capture pre-ReLU activations ===

                # First conv: BasicBlock = Conv -> BN -> ReLU
                pre_act = net.first_conv.bn1(net.first_conv.conv1(x))
                act_flat = pre_act.view(pre_act.size(0), -1)
                binary = (act_flat > 0).float()
                K += (binary @ binary.t()).cpu().numpy()
                K += ((1.0 - binary) @ (1.0 - binary).t()).cpu().numpy()
                h = F.relu(pre_act)

                # Residual layers: BasicBlockRes = Conv -> BN -> (+shortcut) -> ReLU
                p_code = copy.deepcopy(net.pool_code)
                for i, layer in enumerate(net.layers):
                    while i in p_code:
                        p_code.remove(i)
                        h = F.max_pool2d(h, 2)

                    # Compute pre-ReLU activation (BN output + residual shortcut)
                    pre_act = layer.bn1(layer.conv1(h))
                    pre_act = pre_act + layer.shortcut(h)

                    # Binary activation pattern
                    act_flat = pre_act.view(pre_act.size(0), -1)
                    binary = (act_flat > 0).float()
                    K += (binary @ binary.t()).cpu().numpy()
                    K += ((1.0 - binary) @ (1.0 - binary).t()).cpu().numpy()

                    h = F.relu(pre_act)

                # Compute log-determinant of kernel matrix
                sign, logdet = np.linalg.slogdet(K)
                scores.append(logdet)

        avg_score = float(np.mean(scores))
        std_score = float(np.std(scores)) if len(scores) > 1 else 0.0
        return avg_score, std_score

    def process(self, s):
        # Set seed BEFORE any operations for reproducibility
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        depth = self.code[0]
        width = self.code[1]
        pool_code = copy.deepcopy(self.code[2])
        double_code = copy.deepcopy(self.code[3])

        # Create task information (needed for Net initialization)
        # CIFAR-100 with 5 classes per task = 20 tasks
        taskcla = [(i, self.inc) for i in range(20)]

        # Initialize network
        net = Net(taskcla, depth, width, pool_code, double_code)
        net = net.cuda()

        # Calculate number of parameters
        total_params = sum([param.nelement() for param in net.parameters()])
        self.log_record('Number of parameters: %.4fM' % (total_params / 1e6))

        # Calculate NASWOT score
        self.log_record('Calculating NASWOT score (batch_size=64, repeat=1)...')
        naswot_avg, naswot_std = self.calculate_naswot_score(net, batch_size=64, repeat=1)

        self.log_record('NASWOT score (avg): %.6f' % naswot_avg)
        self.log_record('NASWOT score (std): %.6f' % naswot_std)

        # Use NASWOT score directly as fitness
        # Higher log-det = more diverse activation patterns = better architecture
        fitness_score = naswot_avg

        self.log_record('Fitness score (NASWOT): %.3f' % fitness_score)

        # Store metrics
        self.metrics = {
            'naswot_avg': round(naswot_avg, 6),
            'naswot_std': round(naswot_std, 6),
            'naswot_fitness': round(fitness_score, 3),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = NaswotEvaluator()
        try:
            m.log_record('Used GPU#%s, sequential mode, pid:%d'%
                        (gpu_id, os.getpid()), first_time=True)
            fitness_score = m.process(s=SEED)

        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))

        finally:
            # Get metrics from the model
            metrics = getattr(m, 'metrics',
                            {'naswot_fitness': fitness_score,
                             'naswot_avg': 0.0,
                             'naswot_std': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-NASWOT:%.3f, Avg:%.6f, Std:%.6f, Params:%.4fM' %
                        (metrics['naswot_fitness'],
                         metrics['naswot_avg'],
                         metrics['naswot_std'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            # Write naswot metrics: indiXXXX={naswot:73.074, avg:..., std:..., params:...}
            f.write('%s={naswot:%.3f, avg:%.6f, std:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['naswot_fitness'],
                    metrics['naswot_avg'],
                    metrics['naswot_std'],
                    metrics['num_params']))
            f.flush()
            f.close()
