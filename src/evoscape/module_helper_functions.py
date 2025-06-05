import numpy as np
import evoscape.modules as modules


def module_from_string(module_str):
    pars_dict = {}
    module_name, _, module_pars = module_str.partition(': ')
    # module_cls = globals()[module_name]
    module_cls = getattr(modules, module_name)

    pars_str = [par_str.strip() for par_str in module_pars.split(';') if par_str]     #
    # print(pars_str)
    n_str = len(pars_str)
    for par_str in pars_str:
        par_name, _, par_value = par_str.partition('=')
        if '[' in par_value:
            par_value = np.fromstring(par_value.strip('[]'), sep=',')
        else:
            par_value = float(par_value)
        pars_dict[par_name] = par_value

    return module_cls(**pars_dict)

def module_from_string_old(module_str):
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


def landscape_from_timecode(landscape_type, timecode, data_dir, gen, landscape_pars):
    filename = data_dir + timecode + '/' + timecode + '_module_list_' + str(gen) + '.txt'
    with open(filename, 'r') as f:
        module_list = modules_from_txt(filename)
    landscape = landscape_type(module_list, A0=landscape_pars['A0'], regime=landscape_pars['regime'], n_regimes=5)
    return landscape


def transform_coords(module_list, old_coords, origin=0, direction=2, left=None, right=None, bottom=None, scale=False):
    '''
    Transform coordinates from old_coords to the new coordinate system defined by the modules in module_list.
    origin: index of the module that defines the origin
    direction: index of the module(s) that defines the direction of x-axis. If list of indices, the average direction is taken
    left: module index, mirror flip to have x<0
    right: module index, mirror flip to have x>0
    bottom: module index, mirror flip to have y<0
    scale: bool, scale the coordinates to a range of [-1, 1] if True
    '''
    module_coords = np.zeros((len(module_list), 2))
    for i, module in enumerate(module_list):
        module_coords[i, :] = module.x, module.y
    coords = old_coords - module_coords[origin]  # move origin
    module_coords = module_coords - module_coords[origin]

    # x, y = np.mean(module_coords[direction, 0]), np.mean(module_coords[direction, 1])
    norm_coords = (module_coords.T / np.linalg.norm(module_coords, axis=1).T).T
    x, y = np.sum(norm_coords[direction, 0]), np.sum(norm_coords[direction, 1])

    d = np.linalg.norm((x, y))
    R = np.array([[x, y], [-y, x]]) / d
    coords = (R @ coords.T).T  # rotate to align with the direction
    module_coords = (R @ module_coords.T).T

    if bottom is not None:
        if module_coords[bottom, 1] > 0:  # flip the y-axis if needed
            coords[:, 1] *= -1.
    if left is not None:
        if module_coords[left, 0] > 0:  # flip the x-axis if needed
            coords[:, 0] *= -1.
    if right is not None:
        if module_coords[right, 0] < 0:  # flip the x-axis if needed
            coords[:, 0] *= -1.

    if scale:
        coords /= np.max(np.abs(coords))

    return coords


def rotate_landscape(landscape, origin=0, direction=2, left=None, right=None, bottom=None):
    '''
    Transform the coordinate system of the landscape. In-place modification of landscape.module_list.
    origin: index of the module that defines the origin
    direction: index of the module(s) that defines the direction of x-axis. If list of indices, the average direction is taken
    left: module index, mirror flip to have x<0
    right: module index, mirror flip to have x>0
    bottom: module index, mirror flip to have y<0
    scale: bool, scale the coordinates to a range of [-1, 1] if True
    '''
    module_coords = np.zeros((len(landscape.module_list), 2))
    for i, module in enumerate(landscape.module_list):
        module_coords[i, :] = module.x, module.y
    x0 = np.zeros((1, 2))
    x0 -= module_coords[origin]
    module_coords = module_coords - module_coords[origin]
    norm_coords = (module_coords.T / np.linalg.norm(module_coords, axis=1).T).T
    x, y = np.sum(norm_coords[direction, 0]), np.sum(norm_coords[direction, 1])

    d = np.linalg.norm((x, y))
    R = np.array([[x, y], [-y, x]]) / d
    x0 = (R @ x0.T).T
    module_coords = (R @ module_coords.T).T

    if bottom is not None:
        if module_coords[bottom, 1] > 0:  # flip the y-axis if needed
            module_coords[:, 1] *= -1.
            x0[0, 1] *= -1.
    if left is not None:
        if module_coords[left, 0] > 0:  # flip the x-axis if needed
            module_coords[:, 0] *= -1.
            x0[0, 0] *= -1.
    if right is not None:
        if module_coords[right, 0] < 0:  # flip the x-axis if needed
            module_coords[:, 0] *= -1.
            x0[0, 0] *= -1.
    for i, module in enumerate(landscape.module_list):
        module.x, module.y = module_coords[i, :]
    landscape.x0 = x0[0]

    return landscape