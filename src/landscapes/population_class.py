import multiprocessing as mp
import os
import pickle
import json
import time
from copy import deepcopy

import numpy as np

from .module_helper_functions import modules_to_txt


class Population:
    def __init__(self, N, problem_type, landscape_pars, prob_pars, fitness_pars,
                 par_limits, par_choice_values, start_module_list=(), start_fitness=-np.inf):
        self.N = N  # N >= 1 !
        self.problem_type = problem_type
        self.landscape_pars = landscape_pars
        self.prob_pars = prob_pars
        self.par_limits = par_limits
        self.par_choice_values = par_choice_values

        self.landscape_list = []
        for i in range(N):
            self.landscape_list.append(self.problem_type(start_module_list, landscape_pars['A0'],
                                                         landscape_pars['init_cond'],
                                                         landscape_pars['regime'], landscape_pars['n_regimes'],
                                                         landscape_pars['morphogen_times'],
                                                         landscape_pars['used_fp_types'],
                                                         landscape_pars['immutable_pars_list']))

        if start_module_list and fitness_pars:
            fitness = self.landscape_list[0].get_fitness(fitness_pars)
        else:
            fitness = start_fitness
        for landscape in self.landscape_list:
            landscape.fitness = fitness

    # ____________________________________________________________________________________________________________________
    def evolve(self, ngenerations, fitness_pars, saved_files_dir, save_each=10):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        print('Timecode:', timestr)

        os.makedirs(saved_files_dir + self.problem_type.__name__ + '/' + timestr)
        save_dir = saved_files_dir + self.problem_type.__name__ + '/' + timestr + '/'
        save_gens_file = open(save_dir + timestr + "_generations.txt", "a")
        pickle_name = save_dir + timestr + "_module_list_"

        save_gens_file.write('# Evolution of ' + str(self.N) + ' landscapes for ' + self.problem_type.__name__ + '\n'
                             + '# Starting: ' + str(
            max([landscape.fitness for landscape in self.landscape_list])) + '\n')
        with open(save_dir + timestr + "_parameters.pickle", "wb") as f:
            pickle.dump([self.landscape_pars, self.prob_pars,
                         fitness_pars, self.par_limits, self.par_choice_values], f)
        # with open(pickle_name + 'initial' + '.pickle', "wb") as f:
        #     pickle.dump(self.landscape_list[0].module_list, f)
        with open(save_dir + timestr + "_initial_full.pickle", "wb") as f:
            pickle.dump(self, f)

        for igen in range(ngenerations):
            for odd_landscape in self.landscape_list[::2]:
                odd_landscape.mutate(self.par_limits, self.par_choice_values, self.prob_pars, fitness_pars)
            self.landscape_list.sort(key=lambda landscape: landscape.fitness + 0.002 * np.random.randn(), reverse=True)
            del (self.landscape_list[self.N // 2:])
            self.landscape_list = [deepcopy(landscape) for landscape in self.landscape_list for _ in range(2)]

            save_gens_file.write(str(igen) + ' ' + str(np.round(self.landscape_list[0].fitness, 4)) + ' '
                                 + str(len(self.landscape_list[0].module_list)) + '\n')
            if igen % save_each == 0:
                with open(pickle_name + str(igen) + '.pickle', "wb") as f:
                    pickle.dump(self.landscape_list[0].module_list, f)
        save_gens_file.close()
        with open(save_dir + timestr + "_result_full.pickle", "wb") as f:
            pickle.dump(self, f)
        print('Best fitness:', max([landscape.fitness for landscape in self.landscape_list]))

    # ______________________________________________________________________________________________________________________
    def evolve_parallel(self, ngenerations, fitness_pars, saved_files_dir, save_each=10):
        """ Evolutionary optimization using all CPUs """
        timestr = time.strftime("%Y%m%d-%H%M%S")
        print('Timecode:', timestr)
        fitness_traj = np.zeros(ngenerations)
        os.makedirs(saved_files_dir + self.problem_type.__name__ + '/' + timestr)
        save_dir = saved_files_dir + self.problem_type.__name__ + '/' + timestr + '/'
        # save_gens_file = open(save_dir + timestr + "_generations.txt", "a")
        modules_filename = save_dir + timestr + "_module_list_"
        # save_gens_file.write(
        #     '# Parallel evolution of ' + str(self.N) + ' landscapes for ' + self.problem_type.__name__ + '\n'
        #     + '#Starting: ' + str(max([landscape.fitness for landscape in self.landscape_list])) + '\n')
        with open(save_dir + timestr + "_parameters.pickle", "wb") as f:
            pickle.dump([self.landscape_pars, self.prob_pars, fitness_pars,
                         self.par_limits, self.par_choice_values], f)
        with open(save_dir + timestr + "_initial_full.pickle", "wb") as f:
            pickle.dump(self, f)

        pool = mp.Pool(mp.cpu_count())

        for igen in range(ngenerations):
            results = []
            for odd_landscape in self.landscape_list[::2]:
                results.append(pool.apply_async(odd_landscape.mutate_and_return,
                                                args=(self.par_limits, self.par_choice_values,
                                                      self.prob_pars, fitness_pars)))

            for ind in range(self.N // 2):
                result = results[ind].get()
                self.landscape_list[2 * ind] = result

            self.landscape_list.sort(key=lambda landscape: landscape.fitness + 0.002 * np.random.randn(), reverse=True)
            del (self.landscape_list[self.N // 2:])

            fitness_traj[igen] = self.landscape_list[0].fitness
            self.landscape_list = [deepcopy(landscape) for landscape in self.landscape_list for _ in range(2)]

            if igen % save_each == 0:
                modules_to_txt(self.landscape_list[0].module_list, modules_filename + str(igen) + '.txt')
                # with open(modules_filename + str(igen) + '.pickle', "wb") as f:
                #     pickle.dump(self.landscape_list[0].module_list, f)

        pool.close()
        pool.join()

        with open(save_dir + timestr + "_result_full.pickle", "wb") as f:
            pickle.dump(self, f)
        print('Best fitness:', max([landscape.fitness for landscape in self.landscape_list]))
        return fitness_traj

# ____________________________________________________________________________________________________________________

# def evolve_noisy_parallel(self, ngenerations, saved_files_dir, save_each=10, noise=0.05, n_sym=5, ndt=10):   ####  make this a case in evolve_parallel
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     print('Timecode:', timestr)
#     os.makedirs(saved_files_dir + 'stochastic/' + timestr)
#     save_dir = saved_files_dir + 'stochastic/' + timestr + '/'
#     save_gens_file = open(save_dir + timestr+"_generations.txt", "a")
#     pickle_name = save_dir + timestr + "_module_list_"
#     save_gens_file.write('# Parallel evolution of ' + str(self.N)+' landscapes with noise = '+str(noise)+'\n'
#                          +'#Starting: ' + str(max([landscape.fitness for landscape in self.landscape_list])) + '\n')
#     with open(save_dir+timestr+"_parameters.pickle", "wb") as f:
#         pickle.dump([self.landscape_pars,self.prob_pars,self.time_pars,self.fitness_pars,self.par_limits], f)
#     with open(save_dir +timestr + "_initial_full.pickle", "wb") as f:
#         pickle.dump(self, f)
#     pool = mp.Pool(mp.cpu_count())
#     for igen in range(ngenerations):
#         results = []
#         for odd_landscape in self.landscape_list[::2]:
#             results.append(pool.apply_async(odd_landscape.mutate_and_return_noisy,
#                                             args=(self.par_limits, self.prob_pars,
#                                                   self.fitness_pars,self.times,
#                                                   noise, n_sym, ndt)))
#         for ind in range(self.N//2):
#             result = results[ind].get()
#             self.landscape_list[2*ind] = result
#         self.landscape_list.sort(key=lambda landscape: landscape.fitness+0.002*np.random.randn(), reverse=True)
#         del (self.landscape_list[self.N // 2:])
#         self.landscape_list = [deepcopy(landscape) for landscape in self.landscape_list for _ in range(2)]
#         save_gens_file.write(str(igen) + ' ' + str(np.round(self.landscape_list[0].fitness, 4)) + ' '
#                              + str(len(self.landscape_list[0].module_list)) + '\n')
#         if igen % save_each == 0:
#             with open(pickle_name + str(igen) + '.pickle', "wb") as f:
#                 pickle.dump(self.landscape_list[0].module_list, f)
#     save_gens_file.close()
#     pool.close()
#     pool.join()
#     with open(save_dir + timestr + "_result_full.pickle", "wb") as f:
#         pickle.dump(self, f)
#     print('Best fitness:', max([landscape.fitness for landscape in self.landscape_list]))


# ______________________________________________________________________________________________________________________
# def evolve_threading(self, ngenerations):
#     for igen in range(ngenerations):
#         threads = []
#         print('before mutation:', [landscape.fitness for landscape in self.landscape_list])
#
#         for odd_landscape in self.landscape_list[::2]:
#             threads.append(threading.Thread(None, odd_landscape.mutate, None,
#                                             args=(self.par_limits, self.prob_pars,
#                                                   self.fitness_pars, self.times)))
#         for index in threads: index.start()
#         for index in threads: index.join()
#
#         print('after mutation:', [landscape.fitness for landscape in self.landscape_list])
#         self.landscape_list.sort(key=lambda landscape: landscape.fitness + 0.002 * np.random.randn(), reverse=True)
#         del (self.landscape_list[self.N // 2:])
#         self.landscape_list = [deepcopy(landscape) for landscape in self.landscape_list for _ in range(2)]
