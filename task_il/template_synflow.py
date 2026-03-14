"""
Synflow-based Template for Neural Architecture Evaluation
Instead of training, this template uses Synaptic Flow to estimate network quality
Reference: Lee et al. "Pruning neural networks without any data by conserving synaptic flow" (2019)

Note: This version runs sequentially without multiprocessing (CUDA compatible)
"""
from __future__ import print_function
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
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
class SynflowEvaluator(object):
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

    def calculate_synflow(self, net):
        net.eval()

        # Create dummy inputs and targets (all ones)
        # CIFAR-100 image size: 32x32x3
        dummy_input = torch.ones(1, 3, 32, 32).cuda()

        # Get the number of tasks from the network
        # For Task-IL, the network has multiple output heads
        num_tasks = len(net.taskcla)

        total_synflow = 0.0

        # Calculate synflow for each task head
        for t in range(num_tasks):
            # Get number of classes for this task
            n_classes = net.taskcla[t][1]

            # Create dummy target (all ones, binary classification style)
            dummy_output = torch.ones(1, n_classes).cuda()

            # Enable gradient computation for input
            dummy_input.requires_grad = True

            # Forward pass - net.forward() returns list of outputs for all tasks
            outputs = net(dummy_input)
            output = outputs[t]  # Get output for task t

            # Binary cross-entropy style loss (as in original Synflow paper)
            # This forces gradients to flow through all paths
            loss = -torch.mean(torch.log(torch.sigmoid(output) + 1e-8) *
                               dummy_output +
                               torch.log(1 - torch.sigmoid(output) + 1e-8) *
                               (1 - dummy_output))

            # Backward pass
            loss.backward()

            # Calculate synflow score: |w * grad(w)|
            synflow_score = 0.0
            for param in net.parameters():
                if param.grad is not None:
                    synflow_score += torch.sum(torch.abs(param.data * param.grad)).item()

            total_synflow += synflow_score

            # Reset gradients
            net.zero_grad()
            if dummy_input.grad is not None:
                dummy_input.grad.zero_()

        return total_synflow

    def calculate_synflow_per_layer(self, net):
        net.eval()

        dummy_input = torch.ones(1, 3, 32, 32).cuda()
        num_tasks = len(net.taskcla)

        layer_scores = {}

        for t in range(num_tasks):
            n_classes = net.taskcla[t][1]
            dummy_output = torch.ones(1, n_classes).cuda()

            dummy_input.requires_grad = True
            # Forward pass - net.forward() returns list of outputs for all tasks
            outputs = net(dummy_input)
            output = outputs[t]  # Get output for task t

            loss = -torch.mean(torch.log(torch.sigmoid(output) + 1e-8) *
                               dummy_output +
                               torch.log(1 - torch.sigmoid(output) + 1e-8) *
                               (1 - dummy_output))

            loss.backward()

            # Collect scores per parameter tensor
            for name, param in net.named_parameters():
                if param.grad is not None:
                    score = torch.sum(torch.abs(param.data * param.grad)).item()
                    if name not in layer_scores:
                        layer_scores[name] = 0.0
                    layer_scores[name] += score

            net.zero_grad()
            if dummy_input.grad is not None:
                dummy_input.grad.zero_()

        return layer_scores

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

        # Initialize network weights (important for Synflow)
        # Use Kaiming initialization for better gradient flow
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        net.apply(init_weights)
        self.log_record('Network initialized with Kaiming normal')

        # Calculate Synflow score
        self.log_record('Calculating Synflow score...')
        synflow_score = self.calculate_synflow(net)

        # Normalize by number of parameters (for fair comparison across architectures)
        normalized_synflow = synflow_score / total_params if total_params > 0 else 0.0

        self.log_record('Synflow score (raw): %.6f' % synflow_score)
        self.log_record('Synflow score (normalized per param): %.6f' % normalized_synflow)

        # Optional: Calculate per-layer scores for analysis
        layer_scores = self.calculate_synflow_per_layer(net)
        self.log_record('Layer-wise Synflow scores:')
        for layer_name, score in sorted(layer_scores.items(), key=lambda x: x[1], reverse=True):
            self.log_record('  %s: %.6f' % (layer_name, score))

        # Use normalized synflow score as fitness metric
        # Higher synflow = better gradient flow = potentially better architecture
        fitness_score = normalized_synflow * 1000  # Scale up for easier handling

        self.log_record('Fitness score (Synflow): %.3f' % fitness_score)

        # Store metrics
        self.metrics = {
            'synflow_raw': round(synflow_score, 6),
            'synflow_normalized': round(normalized_synflow, 6),
            'synflow_fitness': round(fitness_score, 3),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = SynflowEvaluator()
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
                            {'synflow_fitness': fitness_score,
                             'synflow_raw': 0.0,
                             'synflow_normalized': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-Synflow:%.3f, Raw:%.6f, Norm:%.6f, Params:%.4fM' %
                        (metrics['synflow_fitness'],
                         metrics['synflow_raw'],
                         metrics['synflow_normalized'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            # Write synflow metrics in format: indiXXXX={synflow:73.074, raw:..., norm:..., params:...}
            f.write('%s={synflow:%.3f, raw:%.6f, norm:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['synflow_fitness'],
                    metrics['synflow_raw'],
                    metrics['synflow_normalized'],
                    metrics['num_params']))
            f.flush()
            f.close()