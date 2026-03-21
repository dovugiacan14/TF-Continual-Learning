"""
Fisher Information-based Template for Neural Architecture Evaluation
Uses Fisher Information to estimate network quality without training.

Adapted from pytorch-blockswap Fisher implementation.
Fisher score = sum of 0.5 * mean((activation * gradient)^2) across all layers.

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
import torchvision
import torchvision.transforms as transforms

Inc_cls = 5    # Number of classes incremented at each step

# Auto-detect and change to correct working directory (task_il/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# If script is in task_il/scripts/, go up one level to task_il/
if script_dir.endswith('scripts'):
    script_dir = os.path.dirname(script_dir)
if os.getcwd() != script_dir:
    os.chdir(script_dir)

#generated_code
class FisherEvaluator(object):
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

    def get_dataloader(self, batch_size=64):
        """
        Load a single minibatch of CIFAR-100 for Fisher computation.
        Fisher only needs one forward+backward pass with real data.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        return dataloader

    def calculate_fisher(self, net, dataloader, n_steps=1):
        """
        Calculate Fisher Information score adapted from pytorch-blockswap.

        Algorithm (from blockswap _fisher method):
        1. Forward pass with real data minibatch
        2. Compute cross-entropy loss
        3. Backward pass
        4. For each layer with parameters:
           Fisher_k = 0.5 * mean((param * grad)^2)
        5. Sum across all layers = architecture Fisher score

        Args:
            net: Neural network model
            dataloader: DataLoader with real training data
            n_steps: Number of minibatches to accumulate (default=1 for one-shot)

        Returns:
            fisher_score: Scalar Fisher information score
        """
        net.train()
        criterion = nn.CrossEntropyLoss()

        # Accumulate Fisher over n_steps minibatches
        running_fisher = 0.0

        data_iter = iter(dataloader)
        for step in range(n_steps):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            inputs, targets = inputs.cuda(), targets.cuda()

            # Clear gradients
            net.zero_grad()

            # Forward pass - Net returns list of outputs (one per task)
            outputs = net(inputs)

            # Use first task head for Fisher (task 0, classes 0-4)
            # Remap targets to task-local labels
            task_targets = targets % self.inc
            loss = criterion(outputs[0], task_targets)

            # Backward pass
            loss.backward()

            # Compute Fisher information per parameter
            # Following blockswap: F_k = 0.5 * mean((param * grad)^2)
            step_fisher = 0.0
            for param in net.parameters():
                if param.grad is not None:
                    # Element-wise product of parameter and gradient, squared, mean, scaled by 0.5
                    fisher_param = (param * param.grad).pow(2).sum().item() * 0.5
                    step_fisher += fisher_param

            running_fisher += step_fisher

        # Average over steps
        fisher_score = running_fisher / n_steps

        # Clean up gradients
        net.zero_grad()

        return fisher_score

    def calculate_fisher_per_layer(self, net, dataloader):
        """
        Calculate per-layer Fisher scores for analysis.
        Useful for understanding which layers contribute most.
        """
        net.train()
        criterion = nn.CrossEntropyLoss()

        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.cuda(), targets.cuda()

        net.zero_grad()
        outputs = net(inputs)
        task_targets = targets % self.inc
        loss = criterion(outputs[0], task_targets)
        loss.backward()

        layer_scores = {}
        for name, param in net.named_parameters():
            if param.grad is not None:
                score = (param * param.grad).pow(2).sum().item() * 0.5
                layer_scores[name] = score

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

        # Initialize network weights
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        net.apply(init_weights)
        self.log_record('Network initialized with Xavier uniform')

        # Load real data for Fisher computation
        self.log_record('Loading CIFAR-100 data for Fisher computation...')
        dataloader = self.get_dataloader(batch_size=64)

        # Calculate Fisher score
        self.log_record('Calculating Fisher score...')
        fisher_score = self.calculate_fisher(net, dataloader, n_steps=1)

        # Normalize by number of parameters (for fair comparison across architectures)
        normalized_fisher = fisher_score / total_params if total_params > 0 else 0.0

        self.log_record('Fisher score (raw): %.6f' % fisher_score)
        self.log_record('Fisher score (normalized per param): %.6f' % normalized_fisher)

        # Optional: Calculate per-layer scores for analysis
        dataloader2 = self.get_dataloader(batch_size=64)
        layer_scores = self.calculate_fisher_per_layer(net, dataloader2)
        self.log_record('Layer-wise Fisher scores:')
        for layer_name, score in sorted(layer_scores.items(), key=lambda x: x[1], reverse=True):
            self.log_record('  %s: %.6f' % (layer_name, score))

        # Use normalized Fisher score as fitness metric
        fitness_score = normalized_fisher

        self.log_record('Fitness score (Fisher): %.3f' % fitness_score)

        # Store metrics
        self.metrics = {
            'fisher_raw': round(fisher_score, 6),
            'fisher_normalized': round(normalized_fisher, 6),
            'fisher_fitness': round(fitness_score, 3),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = FisherEvaluator()
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
                            {'fisher_fitness': fitness_score,
                             'fisher_raw': 0.0,
                             'fisher_normalized': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-Fisher:%.3f, Raw:%.6f, Norm:%.6f, Params:%.4fM' %
                        (metrics['fisher_fitness'],
                         metrics['fisher_raw'],
                         metrics['fisher_normalized'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            # Write fisher metrics in format: indiXXXX={fisher:73.074, raw:..., norm:..., params:...}
            f.write('%s={fisher:%.3f, raw:%.6f, norm:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['fisher_fitness'],
                    metrics['fisher_raw'],
                    metrics['fisher_normalized'],
                    metrics['num_params']))
            f.flush()
            f.close()
