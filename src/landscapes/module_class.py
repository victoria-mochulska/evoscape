import random

import numpy as np


class Module:
    def __init__(self, x=0., y=0., a=1., s=1., tau=None):  # removed sigmin and Amax, V parameters
        self.x = x
        self.y = y

        self.a = np.asarray(a)
        self.s = np.asarray(s)

        self.tau = tau

        self.mutable_parameters_list = [par for par in vars(self).keys() if vars(self)[par] is not None]

        self.par_limits = {}
        self.par_choice_values = {}

    # _________________________________________________________________________________________________________
    @classmethod
    def generate(cls, par_limits, par_choice_values, n_regimes=2, immutable_pars_list=()):
        # fp_type = random.choice(fp_types)
        # V0 = 1.

        a = np.ones(n_regimes)
        s = np.ones(n_regimes)

        if 'tau' in par_limits:
            tau = par_limits['tau'][0]
        elif 'tau' in par_choice_values:
            tau = par_choice_values['tau'][0]
        else:
            tau = None

        # if n_regimes == 1:
        #     a = 1.
        #     s = 1.
        #     # V1 = None
        #     # V2 = None
        #     # t0 = 100. #TODO: check n_regimes = 1
        #     # t1 = 100.
        #     # t2 = None
        # if n_regimes == 2:
        #     a = np.ones(n_regimes)
        #     s = np.ones(n_regimes)
        #     # t0 = 0.
        #     # t1 = None
        #     # t2 = None
        #     # tau = 0.5
        # if n_regimes == 3:
        #     a = np.ones(n_regimes)
        #     s = np.ones(n_regimes)
        #     # V1 = 1.
        #     # V2 = 1.
        #     # t0 = 0.
        #     # t1 = 1.
        #     # t2 = None
        if n_regimes >= 4:
            print('Only up to 3 regimes are currently supported')
        module = cls(a=a, s=s, tau=tau)
        for par in module.mutable_parameters_list:
            value = getattr(module, par)
            if par in par_limits:
                if isinstance(value, np.ndarray):
                    setattr(module, par, np.random.uniform(*par_limits[par], len(value)))
                else:
                    setattr(module, par, np.random.uniform(*par_limits[par]))
            elif par in par_choice_values:
                if isinstance(value, np.ndarray):
                    setattr(module, par, np.random.choice(par_choice_values[par], size=len(value)))
                else:
                    setattr(module, par, np.random.choice(par_choice_values[par]))
            else:
                raise ValueError("Limits or choice values not provided for parameter " + par)
        for par in immutable_pars_list:
            module.remove_mutable_parameter(par)
        return module

    def __str__(self):
        par_str = self.__class__.__name__ + ' at ' + str((self.x, self.y))
        for par_name in self.mutable_parameters_list:
            if par_name == 'x' or par_name == 'y':
                continue
            par_str += ', ' + par_name + ' = ' + str(np.round(getattr(self, par_name), 4))
        return par_str

    # def _setattr(self, key, value, index=None):
    #     attr = getattr(self, key)
    #     if isinstance(attr, np.ndarray):
    #         attr[index] = value
    #         # setattr(self, key, attr)
    #     else:
    #         setattr(self, key, value)

    def mutate(self, par_limits, par_choice_values):
        rand_par = random.choice(self.mutable_parameters_list)
        # val = getattr(self, rand_par)
        # if isinstance(val, np.ndarray):
        #     index = np.random.randint(val.size)
        # else:
        #     index = None

        if rand_par in self.par_limits:
            new_val = np.random.uniform(*self.par_limits[rand_par])
        elif rand_par in self.par_choice_values:
            new_val = np.random.choice(self.par_choice_values[rand_par])
            # self._setattr(rand_par, np.random.choice(self.par_choice_values[rand_par]), index)
        elif rand_par in par_limits:
            new_val = np.random.uniform(*par_limits[rand_par])
            # self._setattr(rand_par, np.random.uniform(*par_limits[rand_par]), index)
        elif rand_par in par_choice_values:
            new_val = np.random.choice(par_choice_values[rand_par])
            # setattr(self, rand_par, np.random.choice(par_choice_values[rand_par]))
        else:
            raise ValueError("Limits or choice values not provided for parameter " + rand_par)

        attr = getattr(self, rand_par)
        if isinstance(attr, np.ndarray):
            index = np.random.randint(attr.size)
            attr[index] = new_val
            # setattr(self, key, attr)
        else:
            setattr(self, rand_par, new_val)

        # self._setattr(rand_par, new_val, index)

    def get_current_pars(self, t, regime, t0=100., t1=None, t2=None):
        if self.a.size == 1 and self.s.size == 1:
            V = self.a * self.s ** 2
            return V, self.s, self.a

        s_t, a_t = regime(t, self.a, self.s, t0=t0, t1=t1, t2=t2, tau=self.tau)
        V = a_t * s_t ** 2
        # V = regime(t, V0=self.V0, V1=self.V1, V2=self.V2, V3=self.V3,
        #            t0=t0, t1=t1, t2=t2, tau=self.tau)
        # if V >= self.sigmin ** 2 * self.Amax:
        #     A = self.Amax
        #     sig = np.sqrt(V / self.Amax)
        # else:
        #     A = V / self.sigmin ** 2
        #     sig = self.sigmin
        return V, s_t, a_t

    def add_mutable_parameter(self, par):
        if par not in self.mutable_parameters_list:
            self.mutable_parameters_list.append(par)
        else:
            print(par + ' has already been included.')

    def remove_mutable_parameter(self, par):
        if par in self.mutable_parameters_list:
            self.mutable_parameters_list.remove(par)
        else:
            print(par + ' is not included.')


# ________________________ Fixed point types ____________________________________


class Node(Module):
    def __init__(self, x=0, y=0, a=1., s=1., tau=None):
        super().__init__(x, y, a, s, tau=tau)
        self.J = np.array(((-1, 0.), (0., -1)))


class UnstableNode(Module):
    def __init__(self, x=0., y=0., a=1., s=1., tau=None):
        super().__init__(x, y, a, s, tau=tau)
        self.J = np.array(((+1, 0.), (0., +1)))


class Center(Module):
    def __init__(self, x=0., y=0., a=1., s=1., tau=None):
        super().__init__(x, y, a, s, tau=tau)
        self.J = np.array(((0., -1.), (+1., 0.)))


class NegCenter(Module):
    def __init__(self, x=0., y=0., a=1., s=1, tau=None):
        super().__init__(x, y, a, s, tau=tau)
        self.J = np.array(((0., +1.), (-1., 0.)))
