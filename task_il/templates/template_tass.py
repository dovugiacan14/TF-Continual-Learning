"""
TASS: Task-Aware Sensitivity Score for Continual Learning NAS
A training-free proxy that jointly measures plasticity AND stability.

Formula: TASS = Zen-Score + lambda * (RankScore - InterferenceScore)

Components:
  1. Zen-Score (ICCV 2021): log(output_sensitivity) + BN_correction
     -> Measures network expressivity (plasticity proxy)

  2. RankScore: log(effective_rank) of feature representations
     -> Measures multi-task capacity via feature diversity
     -> Higher rank = more independent dimensions = more "slots" for different tasks
     Grounded in: Loss of Plasticity (Nature 2024), Spectral Collapse (NeurIPS 2025)

  3. InterferenceScore: cosine similarity of per-channel sensitivities across synthetic tasks
     -> Measures forgetting potential via shared computation paths
     -> High similarity = same channels process different distributions = gradient conflict
     Grounded in: GPM (ICLR 2021), Putting a Face to Forgetting (2026)

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
class TASSEvaluator(object):
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
        """Initialize network weights with Gaussian distribution N(0,1)."""
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
        """Forward pass through feature extractor only, before task heads."""
        h = net.first_conv(x)
        p_code = copy.deepcopy(net.pool_code)
        for i, layer in enumerate(net.layers):
            while i in p_code:
                p_code.remove(i)
                h = F.max_pool2d(h, 2)
            h = layer(h)
        return h

    def calculate_tass_score(self, net, num_synthetic_tasks=3, batch_size=16,
                             mixup_gamma=1e-2, repeat=32, lambda_stability=1.0):
        """
        Calculate TASS (Task-Aware Sensitivity Score).

        A training-free proxy for Continual Learning architecture evaluation.
        Combines three components measuring plasticity, capacity, and stability.

        TASS = Zen-Score + lambda * (RankScore - InterferenceScore)

        Algorithm (per repeat):
        1. Re-initialize all weights ~ N(0,1)

        2. [Zen-Score] Plasticity measurement:
           a. x ~ N(0,I), x' = x + gamma * noise
           b. sensitivity = mean(|f(x) - f(x')|)
           c. BN correction = sum(log(sqrt(mean(running_var))))
           d. zen = log(sensitivity) + BN_correction

        3. [RankScore] Multi-task capacity measurement:
           a. Feature map F = f(x), reshape to [B, C*H*W]
           b. SVD -> singular values S
           c. Effective rank = exp(entropy(S_normalized))
           d. rank_score = log(effective_rank)

        4. [InterferenceScore] Forgetting potential measurement:
           a. Generate K synthetic tasks: x_k ~ N(mu_k, I) with random mu_k
           b. Per-channel sensitivity: s_k = mean(|f(x_k) - f(x_k')|, dim=[batch,spatial])
           c. Pairwise cosine similarity between s_k vectors
           d. interference = mean of pairwise similarities

        5. TASS = zen + lambda * (rank_score - interference)

        Args:
            net: Neural network to evaluate
            num_synthetic_tasks: Number of synthetic task distributions (K)
            batch_size: Batch size for random inputs
            mixup_gamma: Perturbation magnitude (alpha)
            repeat: Number of repeats for averaging
            lambda_stability: Weight for stability term (plasticity-stability tradeoff)

        Returns:
            Dictionary with all component scores
        """
        net.train()

        tass_scores = []
        zen_scores = []
        rank_scores = []
        itf_scores = []

        with torch.no_grad():
            for _ in range(repeat):
                # Fresh Gaussian init each repeat
                self._gaussian_init(net)

                # ============================================
                # COMPONENT 1: Zen-Score (Plasticity Proxy)
                # ============================================
                x = torch.randn(batch_size, 3, 32, 32).cuda()
                x_perturbed = x + mixup_gamma * torch.randn(batch_size, 3, 32, 32).cuda()

                output = self._forward_pre_head(net, x)
                output_perturbed = self._forward_pre_head(net, x_perturbed)

                # Output sensitivity
                sensitivity = torch.sum(torch.abs(output - output_perturbed), dim=[1, 2, 3])
                sensitivity = torch.mean(sensitivity)

                # BN scaling correction
                log_bn_scaling = 0.0
                for m in net.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        bn_scaling = torch.sqrt(torch.mean(m.running_var))
                        log_bn_scaling += torch.log(bn_scaling)

                zen = torch.log(sensitivity) + log_bn_scaling
                zen_val = float(zen)

                # ============================================
                # COMPONENT 2: RankScore (Multi-task Capacity)
                # ============================================
                # Reshape feature map to [B, d] where d = C*H*W
                features = output.view(batch_size, -1)
                # Center features (remove mean)
                features = features - features.mean(dim=0, keepdim=True)

                try:
                    # Compute singular values
                    S = torch.linalg.svdvals(features)
                    # Filter near-zero singular values
                    S = S[S > 1e-6]
                    if len(S) > 1:
                        # Effective rank = exp(entropy of normalized singular values)
                        S_norm = S / S.sum()
                        entropy = -(S_norm * torch.log(S_norm)).sum()
                        rank_val = float(entropy)
                    else:
                        rank_val = 0.0
                except Exception:
                    rank_val = 0.0

                # ============================================
                # COMPONENT 3: InterferenceScore (Forgetting Proxy)
                # ============================================
                # Generate K synthetic "tasks" with different distributions
                # Each task has a different random mean, simulating different data domains
                task_sensitivities = []
                for k in range(num_synthetic_tasks):
                    # Random per-channel mean shift (simulating different image distributions)
                    mu_k = torch.randn(1, 3, 1, 1).cuda() * 2.0
                    x_k = mu_k + torch.randn(batch_size, 3, 32, 32).cuda()
                    x_k_perturbed = x_k + mixup_gamma * torch.randn(batch_size, 3, 32, 32).cuda()

                    out_k = self._forward_pre_head(net, x_k)
                    out_k_perturbed = self._forward_pre_head(net, x_k_perturbed)

                    # Per-channel sensitivity: average over batch and spatial dims -> [C]
                    sens_k = torch.mean(torch.abs(out_k - out_k_perturbed), dim=[0, 2, 3])
                    task_sensitivities.append(sens_k)

                # Pairwise cosine similarity between channel sensitivity vectors
                interference = 0.0
                n_pairs = 0
                for i in range(num_synthetic_tasks):
                    for j in range(i + 1, num_synthetic_tasks):
                        cos_sim = F.cosine_similarity(
                            task_sensitivities[i].unsqueeze(0),
                            task_sensitivities[j].unsqueeze(0)
                        )
                        interference += float(cos_sim)
                        n_pairs += 1
                interference /= n_pairs  # Average pairwise similarity in [0, 1]

                # ============================================
                # COMBINE: TASS = Plasticity + lambda * Stability
                # ============================================
                stability = rank_val - interference
                tass = zen_val + lambda_stability * stability

                tass_scores.append(tass)
                zen_scores.append(zen_val)
                rank_scores.append(rank_val)
                itf_scores.append(interference)

        return {
            'tass_avg': float(np.mean(tass_scores)),
            'tass_std': float(np.std(tass_scores)),
            'zen_avg': float(np.mean(zen_scores)),
            'rank_avg': float(np.mean(rank_scores)),
            'itf_avg': float(np.mean(itf_scores)),
        }

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

        # Calculate TASS score
        self.log_record('Calculating TASS (K=3, repeat=32, lambda=1.0)...')
        results = self.calculate_tass_score(
            net,
            num_synthetic_tasks=3,
            batch_size=16,
            mixup_gamma=1e-2,
            repeat=32,
            lambda_stability=1.0
        )

        self.log_record('TASS (avg): %.6f' % results['tass_avg'])
        self.log_record('TASS (std): %.6f' % results['tass_std'])
        self.log_record('  Zen component:    %.6f' % results['zen_avg'])
        self.log_record('  Rank component:   %.6f' % results['rank_avg'])
        self.log_record('  Interf component: %.6f' % results['itf_avg'])

        fitness_score = results['tass_avg']
        self.log_record('Fitness score (TASS): %.3f' % fitness_score)

        # Store metrics
        self.metrics = {
            'tass_avg': round(results['tass_avg'], 6),
            'tass_std': round(results['tass_std'], 6),
            'zen_avg': round(results['zen_avg'], 6),
            'rank_avg': round(results['rank_avg'], 6),
            'itf_avg': round(results['itf_avg'], 6),
            'tass_fitness': round(fitness_score, 3),
            'num_params': round(total_params / 1e6, 4)
        }

        return fitness_score

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        fitness_score = 0.0
        m = TASSEvaluator()
        try:
            m.log_record('Used GPU#%s, sequential mode, pid:%d'%
                        (gpu_id, os.getpid()), first_time=True)
            fitness_score = m.process(s=0)

        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))

        finally:
            metrics = getattr(m, 'metrics',
                            {'tass_fitness': fitness_score,
                             'tass_avg': 0.0,
                             'tass_std': 0.0,
                             'zen_avg': 0.0,
                             'rank_avg': 0.0,
                             'itf_avg': 0.0,
                             'num_params': 0.0})

            m.log_record('Finished-TASS:%.3f, Zen:%.6f, Rank:%.6f, Itf:%.6f, Params:%.4fM' %
                        (metrics['tass_fitness'],
                         metrics['zen_avg'],
                         metrics['rank_avg'],
                         metrics['itf_avg'],
                         metrics['num_params']))

            f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
            f.write('%s={tass:%.3f, zen:%.6f, rank:%.6f, itf:%.6f, params:%.4f}\n'%
                   (file_id,
                    metrics['tass_fitness'],
                    metrics['zen_avg'],
                    metrics['rank_avg'],
                    metrics['itf_avg'],
                    metrics['num_params']))
            f.flush()
            f.close()
