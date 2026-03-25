"""
Synflow-based Template for Neural Architecture Evaluation
Uses Synaptic Flow to estimate network quality without any training data.

Aligned with zero-cost-nas/foresight/pruners/measures/synflow.py (Samsung, 2021).
Core formula: score = sum( |w * grad(w)| ) with linearized (absolute-value) weights.
Key: BN bypassed (identity), float64 precision, all-ones input, no data needed.

Note: This version runs sequentially without multiprocessing (CUDA compatible)
"""
from __future__ import print_function
import os
import sys
from datetime import datetime
import types
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
        Calculate Synflow score aligned with zero-cost-nas reference implementation.

        Reference: zero-cost-nas/foresight/pruners/measures/synflow.py

        Algorithm (same as zero-cost-nas):
        1. Replace BN with identity (bn=False in zero-cost-nas)
        2. Linearize: store signs, set all params to |param|
        3. Convert network to float64 (avoid overflow in deep networks)
        4. Forward pass with all-ones input
        5. Loss = sum(output), backward pass
        6. Score = sum( |w * grad(w)| ) for Conv2d and Linear weights only
        7. Restore original weights and BN

        Multi-head adaptation for continual learning:
        - Loss = sum of all 20 task head outputs

        Returns:
            synflow_score: Scalar score (higher = better gradient flow)
            layer_scores: Dict of per-layer scores for analysis
        """
        # Step 1: Replace BN with identity (same as zero-cost-nas bn=False)
        # zero-cost-nas does: l.forward = types.MethodType(no_op, l)
        def no_op(self, x):
            return x

        bn_originals = {}
        for name, module in net.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_originals[name] = module.forward
                module.forward = types.MethodType(no_op, module)

        # Step 2: Linearize — store signs, set all state to absolute value
        # zero-cost-nas uses state_dict() which includes BN running stats
        @torch.no_grad()
        def linearize(net):
            signs = {}
            for name, param in net.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        signs = linearize(net)

        # Step 3: Convert to float64 to avoid overflow (CRITICAL)
        # Deep networks with absolute-value weights cause exponential growth
        # float32 max ~3.4e38, float64 max ~1.8e308
        net.zero_grad()
        net.double()

        # Step 4: Forward pass with all-ones input (data-free)
        input_dim = [3, 32, 32]  # CIFAR-100 image dimensions
        dummy_input = torch.ones([1] + input_dim).double().cuda()
        outputs = net(dummy_input)

        # Step 5: Loss = sum of all outputs, then backward
        # Multi-head: sum across all 20 task heads
        loss = sum([output.sum() for output in outputs])
        loss.backward()

        # Step 6: Calculate synflow score — ONLY Conv2d and Linear weights
        # Same as zero-cost-nas: get_layer_metric_array only iterates Conv2d/Linear
        synflow_score = 0.0
        layer_scores = {}
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    score = torch.sum(torch.abs(module.weight * module.weight.grad)).item()
                else:
                    score = 0.0
                synflow_score += score
                layer_scores[name] = score

        # Step 7: Restore original weights (nonlinearize)
        # zero-cost-nas: param.mul_(signs[name]) for all state_dict items
        @torch.no_grad()
        def nonlinearize(net, signs):
            for name, param in net.state_dict().items():
                if 'weight_mask' not in name:
                    param.mul_(signs[name])

        nonlinearize(net, signs)

        # Convert back to float32 for subsequent use
        net.float()

        # Restore BN forward methods
        for name, module in net.named_modules():
            if name in bn_originals:
                module.forward = bn_originals[name]

        net.zero_grad()

        return synflow_score, layer_scores

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
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        net.apply(init_weights)
        self.log_record('Network initialized with Xavier uniform')

        # Calculate Synflow score (single call, returns both total and per-layer)
        self.log_record('Calculating Synflow score (aligned with zero-cost-nas)...')
        synflow_score, layer_scores = self.calculate_synflow(net)

        self.log_record('Synflow score (raw): %.6f' % synflow_score)

        # Log per-layer scores for analysis
        self.log_record('Layer-wise Synflow scores:')
        for layer_name, score in sorted(layer_scores.items(), key=lambda x: x[1], reverse=True):
            self.log_record('  %s: %.6f' % (layer_name, score))

        # Use RAW synflow score as fitness (no normalization — same as zero-cost-nas)
        # zero-cost-nas find_measures() uses sum_arr() without dividing by params
        fitness_score = synflow_score

        self.log_record('Fitness score (Synflow raw): %.3f' % fitness_score)
        self.log_record('Num parameters: %.4fM' % (total_params / 1e6))

        # Store metrics
        self.metrics = {
            'synflow_raw': round(synflow_score, 6),
            'synflow_fitness': round(fitness_score, 3),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        import traceback
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = SynflowEvaluator()
        try:
            m.log_record('Used GPU#%s, sequential mode, pid:%d'%
                        (gpu_id, os.getpid()), first_time=True)
            fitness_score = m.process(s=0)

        except BaseException as e:
            err_msg = traceback.format_exc()
            print('Exception occurs, file:%s, pid:%d\n%s'%(file_id, os.getpid(), err_msg))
            m.log_record('Exception occur:%s\n%s'%(str(e), err_msg))

        finally:
            metrics = getattr(m, 'metrics',
                            {'synflow_fitness': fitness_score,
                             'synflow_raw': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-Synflow:%.3f, Raw:%.6f, Params:%.4fM' %
                        (metrics['synflow_fitness'],
                         metrics['synflow_raw'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            f.write('%s={synflow:%.3f, raw:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['synflow_fitness'],
                    metrics['synflow_raw'],
                    metrics['num_params']))
            f.flush()
            f.close()
