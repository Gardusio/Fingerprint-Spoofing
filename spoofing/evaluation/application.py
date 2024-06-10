import math
import numpy as np


class Application:
    def __init__(self, t_prior, c_fp, c_fn, name) -> None:
        self.t_prior = t_prior
        self.n_prior = 1 - t_prior
        self.c_fn = c_fn
        self.c_fp = c_fp
        self.name = name

    def get_effective_prior(self):
        n_prior = 1 - self.t_prior
        tpcfn = self.t_prior * self.c_fn
        return tpcfn / (tpcfn + (n_prior * self.c_fp))

    def get_name(
        self,
    ):
        return self.name

    def get_treshold(self):
        t = -math.log(self.t_prior * self.c_fn / self.n_prior * self.c_fp)
        return t

    def get_norm(self):
        return min(self.t_prior * self.c_fn, self.n_prior * self.c_fp)

    def info(self):
        return f"{self.name}, True prior: {self.t_prior}, Fn Cost: {self.c_fn}, Fp Cost: {self.c_fp}"

    def get_dummy(self):
        return np.minimum(self.t_prior * self.c_fn, self.n_prior * self.c_fp)
