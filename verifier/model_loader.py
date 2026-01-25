import configparser
import os
from typing import List

import numpy as np
from joblib import load
from maraboupy import Marabou

from verifier.constraints import Constraint, ConstraintType

config = configparser.ConfigParser()
config.read('resources/config/config.ini')
config.sections()

MODEL_PATH = config['MODEL']['model.path'] + config['MODEL']['model.name']
SCALER_PATH = config['MODEL']['model.path'] + config['MODEL']['scaler.name']


class ModelUnderVerification:

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise Exception("Model file not found.")

        if not os.path.exists(SCALER_PATH):
            raise Exception("Scaler file not found.")

        self.scaler = load(SCALER_PATH)
        self.model = Marabou.read_tf(filename=MODEL_PATH)
        self.feature_names = config['MODEL']['feature.names'].split(',')

    def clear(self):
        self.model = Marabou.read_tf(filename=MODEL_PATH)

    def get_model(self):
        if not self.model:
            raise Exception("Model not loaded.")
        return self.model

    def lookup_constraint(self, var_i: int, constraints: List[Constraint]):
        for constraint in constraints:
            if constraint.var_index == var_i:
                return constraint
        return None

    def apply(self, constraints: List[Constraint]):
        inputs = self.model.inputVars[0][0]
        placeholder = self.scaler.mean_
        min_unscaled_vec = np.copy(placeholder)
        max_unscaled_vec = np.copy(placeholder)

        for constraint in constraints:
            if constraint.c_type == ConstraintType.INPUT_BOUND:
                min_unscaled_vec[constraint.var_index] = constraint.lower_bound
                max_unscaled_vec[constraint.var_index] = constraint.upper_bound

        min_scaled = self.scaler.transform([min_unscaled_vec])[0]
        max_scaled = self.scaler.transform([max_unscaled_vec])[0]

        for i in range(len(inputs)):
            if self.lookup_constraint(i, constraints) is None:
                self.model.setLowerBound(inputs[i].item(), 0)
                self.model.setUpperBound(inputs[i].item(), 1000.0)
            else:
                self.model.setLowerBound(inputs[i].item(), min_scaled[i])
                self.model.setUpperBound(inputs[i].item(), max_scaled[i])

        outputs = self.model.outputVars[0]

        for constraint in constraints:
            if constraint.c_type == ConstraintType.OUTPUT_BOUND:
                var = outputs[constraint.var_index].item()
                self.model.setLowerBound(var, constraint.lower_bound)
                self.model.setUpperBound(var, constraint.upper_bound)

    def solve(self):
        exitCode, vals, stats = self.model.solve('pima_marabou.log', verbose=False)
        if exitCode == 'sat':
            inputs = self.model.inputVars[0][0]
            out_var = self.model.outputVars[0][0].item()
            example_inputs_scaled = [vals[var.item()] for var in inputs]
            unscaled_inputs = self.scaler.inverse_transform([example_inputs_scaled])[0]
            return exitCode, zip(self.feature_names, unscaled_inputs), vals[out_var]
        else:
            return exitCode, None, None
