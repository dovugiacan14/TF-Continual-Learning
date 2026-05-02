"""
2026-04-14  21:17:31
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
import types
import multiprocessing  # Required for RunModel interface compatibility
import torchvision
import torchvision.transforms as transforms
import random  # For seed setting

Inc_cls = 5    # Number of classes incremented at each step

# Auto-detect and change to correct working directory (task_il/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# If script is in task_il/scripts/, go up one level to task_il/
if script_dir.endswith('scripts'):
    script_dir = os.path.dirname(script_dir)
if os.getcwd() != script_dir:
    os.chdir(script_dir)


code = [10, 40, [0, 2, 8, 10, 10], [3, 8, 10, 10, 10]]
class SnipEvaluator(object):
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
        Load a single minibatch of CIFAR-100 for SNIP computation.
        SNIP needs one forward+backward pass with real data.
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

    def calculate_snip(self, net, dataloader):
        """
        Calculate SNIP score (connection sensitivity).

        Algorithm (from zero-cost-nas Samsung implementation):
        1. For each Conv2d/Linear layer, create a weight_mask = ones_like(weight)
        2. Freeze original weights, make weight_mask a Parameter
        3. Replace forward: output = conv(x, weight * weight_mask, ...)
        4. Forward pass with real data, backward pass
        5. SNIP score = sum of |weight_mask.grad| across all layers

        The gradient of weight_mask measures how sensitive the loss is
        to each connection. Higher total sensitivity = better architecture.

        Args:
            net: Neural network model
            dataloader: DataLoader with real training data

        Returns:
            snip_score: Scalar SNIP score
        """
        net.train()
        criterion = nn.CrossEntropyLoss()

        # Step 1-3: Install weight_mask and replace forward methods
        def snip_forward_conv2d(self, x):
            return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

        def snip_forward_linear(self, x):
            return F.linear(x, self.weight * self.weight_mask, self.bias)

        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False

            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

        # Step 4: Forward + backward with real data
        net.zero_grad()

        inputs, targets = next(iter(dataloader))
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)

        # Sum loss across all task heads (no averaging, consistent with Fisher fix)
        task_targets = targets % self.inc
        loss = sum([criterion(out, task_targets) for out in outputs])
        loss.backward()

        # Step 5: Collect |weight_mask.grad| as SNIP score
        snip_score = 0.0
        layer_scores = {}
        with torch.no_grad():
            for name, layer in net.named_modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    if layer.weight_mask.grad is not None:
                        layer_snip = torch.abs(layer.weight_mask.grad).sum().item()
                    else:
                        layer_snip = 0.0
                    snip_score += layer_snip
                    layer_scores[name] = layer_snip

        net.zero_grad()
        return snip_score, layer_scores

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

        # Initialize network weights with Xavier uniform
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        net.apply(init_weights)
        self.log_record('Network initialized with Xavier uniform')

        # Calculate SNIP score (average over 3 runs for stability)
        self.log_record('Calculating SNIP score...')
        snip_scores = []
        all_layer_scores = None
        for run in range(3):
            # Re-init weights each run for unbiased measurement
            net.apply(init_weights)
            dl = self.get_dataloader(batch_size=64)
            score, layer_scores = self.calculate_snip(net, dl)
            snip_scores.append(score)
            if all_layer_scores is None:
                all_layer_scores = layer_scores
            self.log_record('SNIP run %d: %.6f' % (run, score))

        snip_score = np.mean(snip_scores)
        snip_std = np.std(snip_scores)

        # Use log(1 + score) for fitness
        fitness_score = float(np.log1p(snip_score))

        self.log_record('SNIP score (raw mean): %.6f' % snip_score)
        self.log_record('SNIP score (raw std): %.6f' % snip_std)
        self.log_record('Fitness score (log1p): %.6f' % fitness_score)

        # Per-layer scores for analysis
        self.log_record('Layer-wise SNIP scores (top 10):')
        for i, (layer_name, score) in enumerate(sorted(all_layer_scores.items(), key=lambda x: x[1], reverse=True)):
            if i >= 10:
                break
            self.log_record('  %s: %.6f' % (layer_name, score))

        self.log_record('Fitness score (SNIP): %.6f' % fitness_score)

        # Store metrics
        self.metrics = {
            'snip_raw': round(snip_score, 6),
            'snip_std': round(snip_std, 6),
            'snip_fitness': round(fitness_score, 6),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = SnipEvaluator()
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
                            {'snip_fitness': fitness_score,
                             'snip_raw': 0.0,
                             'snip_std': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-SNIP:%.6f, Raw:%.6f, Std:%.6f, Params:%.4fM' %
                        (metrics['snip_fitness'],
                         metrics['snip_raw'],
                         metrics['snip_std'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            f.write('%s={snip:%.6f, raw:%.6f, std:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['snip_fitness'],
                    metrics['snip_raw'],
                    metrics['snip_std'],
                    metrics['num_params']))
            f.flush()
            f.close()