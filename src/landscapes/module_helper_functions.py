import numpy as np
import landscapes.modules as modules


def module_from_string(module_str):
    pars_dict = {}
    module_name, _, module_pars = module_str.partition(': ')
    # module_cls = globals()[module_name]
    module_cls = getattr(modules, module_name)

    pars_str = [par_str for par_str in module_pars.split(', ') if par_str]     #
    # print(pars_str)
    n_str = len(pars_str)
    for i in range(n_str):
        #   this is to fix extra whitespaces in arrays
        if i < n_str-1 and '=' in pars_str[i] and '=' not in pars_str[i+1]:
            par_str = pars_str[i] + ',' + pars_str[i+1]
            if i < n_str-2 and '=' not in pars_str[i+2]:
                par_str = par_str + ',' + pars_str[i + 2]
        elif '=' not in pars_str[i]:
            continue
        else:
            par_str = pars_str[i]

        par_name, _, par_value = par_str.partition('=')
        if '[' in par_value:
            par_value = np.fromstring(par_value.strip('[]'), sep=',')
        else:
            par_value = float(par_value)
        pars_dict[par_name] = par_value

    return module_cls(**pars_dict)


def modules_to_txt(module_list, filename):
    with open(filename, 'w') as f:
        for module in module_list:
            f.write(module.__str__() + '\n')


def modules_from_txt(filename):
    module_list = []
    with open(filename, 'r') as f:
        for line in f.read().splitlines():
            module_list.append(module_from_string(line))
    return module_list
