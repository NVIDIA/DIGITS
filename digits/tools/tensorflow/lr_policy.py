# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# This document should comply with PEP-8 Style Guide
# Linter: pylint

"""
Class for generating Caffe-style learning rates using different policies.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class LRPolicy(object):
    """This class contains details of learning rate policies that are used in caffe.
    Calculates and returns the current learning rate. The currently implemented learning rate
    policies are as follows:
       - fixed: always return base_lr.
       - step: return base_lr * gamma ^ (floor(iter / step))
       - exp: return base_lr * gamma ^ iter
       - inv: return base_lr * (1 + gamma * iter) ^ (- power)
       - multistep: similar to step but it allows non uniform steps defined by
         stepvalue
       - poly: the effective learning rate follows a polynomial decay, to be
         zero by the max_steps. return base_lr (1 - iter/max_steps) ^ (power)
       - sigmoid: the effective learning rate follows a sigmod decay
         return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
    """

    def __init__(self, policy, base_rate, gamma, power, max_steps, step_values):
        """Initialize a learning rate policy
        Args:
            policy: Learning rate policy
            base_rate: Base learning rate
            gamma: parameter to compute learning rate
            power: parameter to compute learning rate
            max_steps: parameter to compute learning rate
            step_values: parameter(s) to compute learning rate. should be a string, multiple values divided as csv
        Returns:
            -
        """
        self.policy = policy
        self.base_rate = base_rate
        self.gamma = gamma
        self.power = power
        self.max_steps = max_steps
        self.step_values = step_values
        if self.step_values:
            self.stepvalues_list = map(float, step_values.split(','))
        else:
            self.stepvalues_list = []

        if (self.max_steps < len(self.stepvalues_list)):
            self.policy = 'step'
            self.stepvalues_list[0] = 1
            logging.info("Maximum iterations (i.e., %s) is less than provided step values count "
                         "(i.e, %s), so learning rate policy is reset to (%s) policy with the "
                         "step value (%s).",
                         self.max_steps, len(self.stepvalues_list),
                         self.policy,
                         self.stepvalues_list[0])
        else:            # Converting stepsize percentages into values
            for i in range(len(self.stepvalues_list)):
                self.stepvalues_list[i] = round(self.max_steps * self.stepvalues_list[i] / 100)
                # Avoids 'nan' values during learning rate calculation
                if self.stepvalues_list[i] == 0:
                    self.stepvalues_list[i] = 1

        if (self.policy == 'step') or (self.policy == 'sigmoid'):
            # If the policy is not multistep, then even though multiple step values
            # are provided as input, we will consider only the first value.
            self.step_size = self.stepvalues_list[0]
        elif (self.policy == 'multistep'):
            self.current_step = 0  # This counter is important to take arbitary steps
            self.stepvalue_size = len(self.stepvalues_list)

    def get_learning_rate(self, step):
        """Initialize a learning rate policy
        Args:
            step: the current step for which the learning rate should be computed
        Returns:
            rate: the learning rate for the requested step
        """
        rate = 0
        progress = 100 * (step / self.max_steps)  # expressed in percent units

        if self.policy == "fixed":
            rate = self.base_rate
        elif self.policy == "step":
            current_step = math.floor(step/self.step_size)
            rate = self.base_rate * math.pow(self.gamma, current_step)
        elif self.policy == "exp":
            rate = self.base_rate * math.pow(self.gamma, progress)
        elif self.policy == "inv":
            rate = self.base_rate * math.pow(1 + self.gamma * progress, - self.power)
        elif self.policy == "multistep":
            if ((self.current_step < self.stepvalue_size) and (step > self.stepvalues_list[self.current_step])):
                self.current_step = self.current_step + 1
            rate = self.base_rate * math.pow(self.gamma, self.current_step)
        elif self.policy == "poly":
            rate = self.base_rate * math.pow(1.0 - (step / self.max_steps), self.power)
        elif self.policy == "sigmoid":
            rate = self.base_rate * \
                (1.0 / (1.0 + math.exp(self.gamma * (progress - 100 * self.step_size / self.max_steps))))
        else:
            logging.error("Unknown learning rate policy: %s", self.policy)
            exit(-1)
        return rate
