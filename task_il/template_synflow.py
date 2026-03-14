"""
Synflow-based Template for Neural Architecture Evaluation
Instead of training, this template uses Synaptic Flow to estimate network quality

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
        """
        Calculate Synflow score following the original paper:
        'Pruning neural networks without any data by iteratively conserving synaptic flow'

        Algorithm:
        1. Linearize all weights: w = |w|
        2. Forward pass with all-ones input
        3. Loss = sum(output)
        4. Backward pass
        5. Synflow score = Σ|w * grad(w)|
        6. Restore original weights

        Note: Handles BatchNorm layers by freezing them during computation.
        """
        net.eval()

        # Handle BatchNorm: freeze BN stats to avoid issues with all-ones input
        # (all-ones input has zero variance, which breaks BN)
        bn_params = []
        for module in net.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_params.append((module, module.training))
                module.eval()  # Ensure BN uses running stats, not batch stats

        # Clear any existing gradients before we start
        net.zero_grad()

        # Step 1: Linearize weights - store signs and set all weights to positive
        # Use torch.no_grad() to avoid building unnecessary computational graph
        signs = {}
        with torch.no_grad():
            for name, param in net.named_parameters():
                signs[name] = torch.sign(param)  # Use param instead of param.data (PyTorch best practice)
                param.abs_()  # In-place absolute value

        # Step 2: Create all-ones dummy input
        # CIFAR-100 image size: 32x32x3
        dummy_input = torch.ones(1, 3, 32, 32).cuda()

        # Step 3: Forward pass
        # For Task-IL, the network returns a list of outputs for all tasks
        outputs = net(dummy_input)

        # Step 4: Calculate loss as sum of all outputs (as per original paper)
        # This ensures gradients flow through all paths
        # NOTE: Original SynFlow paper uses sum() WITHOUT abs()
        loss = sum([output.sum() for output in outputs])

        # Step 5: Backward pass
        loss.backward()

        # Step 6: Calculate synflow score: |w * grad(w)|
        # IMPORTANT: Multiply FIRST, then abs() (as per original paper)
        synflow_score = 0.0
        for param in net.parameters():
            if param.grad is not None:
                synflow_score += torch.sum((param.grad * param).abs()).item()

        # Step 7: Restore original weights
        with torch.no_grad():
            for name, param in net.named_parameters():
                param *= signs[name]  # Use direct assignment instead of param.data

        # Restore BatchNorm training state if needed
        for module, was_training in bn_params:
            if was_training:
                module.train()

        # Clean up gradients
        net.zero_grad()

        return synflow_score

    def calculate_synflow_per_layer(self, net):
        """
        Calculate per-layer synflow scores following the original paper.
        Useful for analyzing which layers contribute most to gradient flow.

        Note: Handles BatchNorm layers by freezing them during computation.
        """
        net.eval()

        # Handle BatchNorm: freeze BN stats to avoid issues with all-ones input
        bn_params = []
        for module in net.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_params.append((module, module.training))
                module.eval()

        # Clear any existing gradients
        net.zero_grad()

        # Step 1: Linearize weights
        signs = {}
        with torch.no_grad():
            for name, param in net.named_parameters():
                signs[name] = torch.sign(param)  # Use param instead of param.data (PyTorch best practice)
                param.abs_()

        # Step 2: Create all-ones dummy input
        dummy_input = torch.ones(1, 3, 32, 32).cuda()

        # Step 3: Forward pass
        outputs = net(dummy_input)

        # Step 4: Calculate loss as sum of all outputs
        # NOTE: Original SynFlow paper uses sum() WITHOUT abs()
        loss = sum([output.sum() for output in outputs])

        # Step 5: Backward pass
        loss.backward()

        # Step 6: Collect scores per parameter tensor
        # IMPORTANT: Multiply FIRST, then abs() (as per original paper)
        layer_scores = {}
        for name, param in net.named_parameters():
            if param.grad is not None:
                score = torch.sum((param.grad * param).abs()).item()
                layer_scores[name] = score

        # Step 7: Restore original weights
        with torch.no_grad():
            for name, param in net.named_parameters():
                param *= signs[name]  # Use direct assignment instead of param.data

        # Restore BatchNorm training state if needed
        for module, was_training in bn_params:
            if was_training:
                module.train()

        # Clean up gradients
        net.zero_grad()

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
        # Use Xavier uniform initialization for smaller weights and more stable synflow scores
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        net.apply(init_weights)
        self.log_record('Network initialized with Xavier uniform')

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
        # No need to scale - normalized score is already meaningful
        fitness_score = normalized_synflow

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