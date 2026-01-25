from enum import Enum


class ConstraintType(Enum):
    INPUT_BOUND = 1
    OUTPUT_BOUND = 2


class Constraint:
    def __init__(self, c_type: ConstraintType, var_index: int, l_b: float = -1000.0, u_b: float = 1000.0):
        self.c_type = c_type
        self.var_index = var_index
        self.lower_bound = l_b
        self.upper_bound = u_b
