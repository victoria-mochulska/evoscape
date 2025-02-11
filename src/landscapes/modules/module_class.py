import random
import numpy as np


class Module:
    """ Parent class for a generic module (local dynamics kernel) """
    def __init__(self, x=0., y=0., a=1., s=1., tau=None, immutable_pars_list=()):
        """
        :param x: float
        :param y: float
        :param a: float, list, or array - parameters specifying amplitude
        :param s: float, list, or array - parameters specifying width/size
        :param tau: float - timescale parameter
        :param immutable_pars_list: list of par names (strings)
        All non-None parameters are mutable by default, unless specified in immutable_pars_list.
        Immutable parameter values are not mutated/updated by optimization but can still be changed manually.
        """
        self.x = x
        self.y = y
        self.a = np.asarray(a)
        self.s = np.asarray(s)
        self.tau = tau

        self.mutable_parameters_list = [par for par in vars(self).keys() if vars(self)[par] is not None]
        for par in immutable_pars_list:
            self.remove_mutable_parameter(par)
        self.immutable_idx = []  # for partly mutable vector parameters: indices of elements to keep fixed

        # parameter priors (limits or choice values) can be specified for each module separately:
        self.par_limits = {}
        self.par_choice_values = {}

    def __str__(self):
        # TODO: add precision (decimals)
        module_str = self.__class__.__name__ + ': '
        pars_str = []
        # for par in self.mutable_parameters_list:
        for par in ('x', 'y', 'a', 's', 'tau'):
            value = getattr(self, par)
            if value is not None:
                if isinstance(value, np.ndarray):
                    par_str = np.array2string(value, separator=',', floatmode='maxprec_equal')
                else:
                    par_str = str(value)
                pars_str.append(par + '=' + par_str)
        module_str += '; '.join(pars_str)
        return module_str

    # _________________________________________________________________________________________________________
    @classmethod
    def generate(cls, par_limits, par_choice_values, n_regimes, immutable_pars_list=()):
        """
        Generate a random module from given priors (limits or choice values)
        Either par_limits or par_choice_values should be specified for x, y, a and s; tau is optional
        :param par_limits: dict of lower and upper bounds for a uniform prior, e.g: {'x' : (-1, 1), 'y' : (-1, 1)}
        :param par_choice_values: dict of discrete possible values, e.g: {'a' : (0, 0.5, 1.)}
        :param n_regimes: int (1, 2, or 3) - length of a and s to sample = number of signalling regimes
        :param immutable_pars_list: parameters to make immutable after random generation
        """
        a = np.ones(n_regimes)
        s = np.ones(n_regimes)

        if 'tau' in par_limits:
            tau = par_limits['tau'][0]
        elif 'tau' in par_choice_values:
            tau = par_choice_values['tau'][0]
        else:
            tau = None

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

    def get_current_pars(self, t, regime, t0=None, t1=None, t2=None, t3=None, t4=None):
        """
        Calculate the amplitude and size of the module at time t, based on the a and s parameters and a chosen regime.
        :param t: float
        :param regime: function (list provided in morphogen_regimes)
        :param t0: float, optional
        :param t1: float, optional
        :param t2: float, optional
        :return: V - volume, s_t - size at time t, a_t - amplitude at time t
        """
        if self.a.size == 1 and self.s.size == 1:
            V = float(self.a) * float(self.s) ** 2
            return V, float(self.s), float(self.a)

        s_t, a_t = regime(t, self.a, self.s, t0=t0, t1=t1, t2=t2, t3=t3, t4=t4, tau=self.tau)
        V = a_t * s_t ** 2
        return V, s_t, a_t

    def mutate(self, par_limits, par_choice_values):
        """
        Randomly sample one new parameter value from provided priors.
        For array-like parameters (a, s), only one element is mutated at a time.
        If the module contains its own par_limits or par_choice_values for some parameters, these will be prioritized.
        :param par_limits: dict of lower and upper bounds for a uniform prior, e.g: {'x' : (-1, 1), 'y' : (-1, 1)}
        :param par_choice_values: dict of discrete possible values, e.g: {'a' : (0, 0.5, 1.)}
        """
        rand_par = random.choice(self.mutable_parameters_list)

        if rand_par in self.par_limits:
            new_val = np.random.uniform(*self.par_limits[rand_par])
        elif rand_par in self.par_choice_values:
            new_val = np.random.choice(self.par_choice_values[rand_par])

        elif rand_par in par_limits:
            new_val = np.random.uniform(*par_limits[rand_par])
        elif rand_par in par_choice_values:
            new_val = np.random.choice(par_choice_values[rand_par])
        else:
            raise ValueError("Limits or choice values not provided for parameter " + rand_par)

        attr = getattr(self, rand_par)
        if isinstance(attr, np.ndarray):
            index = np.random.randint(attr.size)
            #  resample if the element is immutable:
            while index in self.immutable_idx:
                index = np.random.randint(attr.size)
            #
            attr[index] = new_val
        else:
            setattr(self, rand_par, new_val)

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

    def set_immutable_idx(self, idx):
        idx = list(idx)
        idx.sort()
        if idx == list(range(self.a.size)):
            print('All vector elements are immutable - setting immutable parameters')
            self.remove_mutable_parameter('a')
            self.remove_mutable_parameter('s')
        else:
            self.immutable_idx = idx



# _______________________________________________________________________________

# _______________________________________________________________________________
# ________________________ Fixed point types ____________________________________


class Node(Module):
    """ Attracting module, gradient dynamics """
    def __init__(self, x=0, y=0, a=1., s=1., tau=None):
        super().__init__(x=x, y=y, a=a, s=s, tau=tau)
        self.J = np.array(((-1, 0.), (0., -1)))


class UnstableNode(Module):
    """ Repelling module, gradient dynamics """
    def __init__(self, x=0., y=0., a=1., s=1., tau=None):
        super().__init__(x=x, y=y, a=a, s=s, tau=tau)
        self.J = np.array(((+1, 0.), (0., +1)))


class Center(Module):
    """ Counterclockwise rotation module """
    def __init__(self, x=0., y=0., a=1., s=1., tau=None):
        super().__init__(x=x, y=y, a=a, s=s, tau=tau)
        self.J = np.array(((0., -1.), (+1., 0.)))


class NegCenter(Module):
    """ Clockwise rotation module """
    def __init__(self, x=0., y=0., a=1., s=1, tau=None):
        super().__init__(x=x, y=y, a=a, s=s, tau=tau)
        self.J = np.array(((0., +1.), (-1., 0.)))
