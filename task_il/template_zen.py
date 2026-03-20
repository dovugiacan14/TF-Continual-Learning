"""
Zen-NAS Template for Neural Architecture Evaluation
Uses Zen-Score (Gaussian Complexity) to estimate network quality without training.

Reference: 'Zen-NAS: A Zero-Shot NAS for High-Performance Image Recognition' (ICCV 2021)
Algorithm: Zen-Score = log(output_sensitivity) + BN_scaling_correction

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
import multiprocessing  # Required for RunModel interface compatibility

Inc_cls = 5    # Number of classes incremented at each step

# Auto-detect and change to correct working directory (task_il/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# If script is in task_il/scripts/, go up one level to task_il/
if script_dir.endswith('scripts'):
    script_dir = os.path.dirname(script_dir)
if os.getcwd() != script_dir:
    os.chdir(script_dir)

#generated_code
class ZenEvaluator(object):
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

    def _gaussian_init(self, net):
        """Initialize network weights with Gaussian distribution N(0,1).
        Following the original Zen-NAS paper."""
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _forward_pre_head(self, net, x):
        """Forward pass through feature extractor only, before task heads.
        Analogous to forward_pre_GAP() in the original Zen-NAS."""
        h = net.first_conv(x)
        p_code = copy.deepcopy(net.pool_code)
        for i, layer in enumerate(net.layers):
            while i in p_code:
                p_code.remove(i)
                h = F.max_pool2d(h, 2)
            h = layer(h)
        return h

    def calculate_zen_score(self, net, batch_size=16, mixup_gamma=1e-2, repeat=32):
        """
        Calculate Zen-Score following the original Zen-NAS paper (ICCV 2021).

        Algorithm:
        1. For each repeat:
           a. Re-initialize all weights with Gaussian N(0,1)
           b. Generate random input x and perturbed input x' = x + gamma * noise
           c. Forward pass (feature extractor only, before task heads)
           d. Sensitivity = mean(|output(x) - output(x')|)
           e. BN correction = sum of log(sqrt(mean(running_var))) per BN layer
           f. Zen-Score = log(sensitivity) + BN correction
        2. Return mean score across repeats

        Key advantages over SynFlow:
        - No backward pass needed (only forward)
        - BN actively normalizes activations (prevents overflow)
        - Gaussian input avoids all-positive compounding
        - Naturally produces log-scale scores (no inf)
        """
        # Keep net in TRAIN mode so BN computes batch stats and updates running_var
        net.train()

        scores = []
        with torch.no_grad():
            for _ in range(repeat):
                # Step 1: Fresh Gaussian init each repeat
                self._gaussian_init(net)

                # Step 2: Random Gaussian input + perturbation
                x = torch.randn(batch_size, 3, 32, 32).cuda()
                x_perturbed = x + mixup_gamma * torch.randn(batch_size, 3, 32, 32).cuda()

                # Step 3: Forward pass (features before task heads)
                output = self._forward_pre_head(net, x)
                output_perturbed = self._forward_pre_head(net, x_perturbed)

                # Step 4: Output sensitivity
                sensitivity = torch.sum(torch.abs(output - output_perturbed), dim=[1, 2, 3])
                sensitivity = torch.mean(sensitivity)

                # Step 5: BN scaling correction
                # BN normalizes activations, losing magnitude info.
                # Compensate by accumulating log of BN running variance.
                log_bn_scaling = 0.0
                for m in net.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        bn_scaling = torch.sqrt(torch.mean(m.running_var))
                        log_bn_scaling += torch.log(bn_scaling)

                # Step 6: Zen-Score = log(sensitivity) + BN correction
                zen_score = torch.log(sensitivity) + log_bn_scaling
                scores.append(float(zen_score))

        avg_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        return avg_score, std_score

    def process(self, s):
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

        # Calculate Zen-Score (no separate init needed - done inside each repeat)
        self.log_record('Calculating Zen-Score (repeat=32)...')
        zen_avg, zen_std = self.calculate_zen_score(net, batch_size=16, mixup_gamma=1e-2, repeat=32)

        self.log_record('Zen-Score (avg): %.6f' % zen_avg)
        self.log_record('Zen-Score (std): %.6f' % zen_std)

        # Use Zen-Score directly as fitness
        # Higher zen-score = more expressive architecture = potentially better
        fitness_score = zen_avg

        self.log_record('Fitness score (Zen): %.3f' % fitness_score)

        # Store metrics
        self.metrics = {
            'zen_avg': round(zen_avg, 6),
            'zen_std': round(zen_std, 6),
            'zen_fitness': round(fitness_score, 3),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = ZenEvaluator()
        try:
            m.log_record('Used GPU#%s, sequential mode, pid:%d'%
                        (gpu_id, os.getpid()), first_time=True)
            fitness_score = m.process(s=0)

        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))

        finally:
            # Get metrics from the model
            metrics = getattr(m, 'metrics',
                            {'zen_fitness': fitness_score,
                             'zen_avg': 0.0,
                             'zen_std': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-Zen:%.3f, Avg:%.6f, Std:%.6f, Params:%.4fM' %
                        (metrics['zen_fitness'],
                         metrics['zen_avg'],
                         metrics['zen_std'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            # Write zen metrics in format: indiXXXX={zen:73.074, avg:..., std:..., params:...}
            f.write('%s={zen:%.3f, avg:%.6f, std:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['zen_fitness'],
                    metrics['zen_avg'],
                    metrics['zen_std'],
                    metrics['num_params']))
            f.flush()
            f.close()
