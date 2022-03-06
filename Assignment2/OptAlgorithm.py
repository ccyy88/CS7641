import mlrose_hiive as mh
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from mlrose_hiive.algorithms.decay import GeomDecay

# # Create Optimization Problems(Fitness Functions)
class OptAlgos:
    def __init__(self):
        self.seed = 5
        self.state = 2
        self.num = 50
        self.num2 = 80

    def ga(self, problem_fit, max_attempts=10, max_iters=1000, curve = False):
        start= time.time()
        best_state, best_fitness, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=max_attempts, max_iters=max_iters, random_state=self.state, curve=curve)
        t = time.time() - start
        return best_fitness, t, fitness_curve


    def sa(self, problem_fit, max_attempts=10, max_iters=1000, curve = False):
        start = time.time()
        best_state, best_fitness, fitness_curve = mh.simulated_annealing(problem_fit, max_attempts=max_attempts,
                                                                 max_iters=max_iters, random_state=self.state, curve=curve)
        t = time.time() - start
        return best_fitness, t, fitness_curve

    def rhc(self, problem_fit, max_attempts=10, max_iters=1000, curve = False):
        start = time.time()
        best_state, best_fitness, fitness_curve = mh.random_hill_climb(problem_fit, max_attempts=max_attempts,
                                                                 max_iters=max_iters, random_state=self.state, curve=curve, restarts=2)
        t = time.time() - start
        return best_fitness, t, fitness_curve


    def mmc(self, problem_fit, max_attempts=10, max_iters=1000, curve = False):
        start = time.time()
        best_state, best_fitness, fitness_curve = mh.mimic(problem_fit, max_attempts=max_attempts,
                                                                 max_iters=max_iters, random_state=self.state, curve=curve,keep_pct=0.3)
        t = time.time() - start
        return best_fitness, t, fitness_curve

    def pltsavefig(self, df, title, ylabel):
        plt.figure()
        df.plot()
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid()
        plt.tight_layout()
        plt.savefig('{}.png'.format(title))



    def fitness_iter_plot(self, problem_fit,problem_name):
        fitness_lst = []
        time_lst = []
        for iter in [10,20,30,50,80]+list(range(100,1001,50)):
            fitness_ga, time_ga, fit_curve_ga = self.ga(problem_fit,max_attempts=10, max_iters=iter)
            fitness_sa, time_sa, fit_curve_sa = self.sa(problem_fit,max_attempts=10, max_iters=iter)
            fitness_rhc, time_rhc, fit_curve_rhc = self.rhc(problem_fit,max_attempts=10, max_iters=iter)
            fitness_mmc, time_mmc, fit_curve_mmc = self.mmc(problem_fit,max_attempts=10, max_iters=iter)
            fitness_lst.append([iter, fitness_ga, fitness_sa, fitness_rhc, fitness_mmc])
            time_lst.append([iter, time_ga, time_sa, time_rhc, time_mmc])
        fitness_df = pd.DataFrame(fitness_lst, columns=['Iterations', 'GA', 'SA', 'RHC', "MIMIC"])
        fitness_df.set_index('Iterations', inplace=True)
        if problem_name == 'TSP' or problem_name == 'Queens':
            fitness_df = 1.0/fitness_df
        self.pltsavefig(fitness_df, title='{}: Fitness vs Iterations'.format(problem_name), ylabel='Fitness')

        time_df = pd.DataFrame(time_lst, columns=['Iterations', 'GA', 'SA', 'RHC', "MIMIC"])
        time_df.set_index('Iterations', inplace=True)
        self.pltsavefig(time_df, title='{}: Wall Clock Time vs Iterations'.format(problem_name), ylabel='Time')
        self.pltsavefig(time_df[['GA', 'SA', 'RHC']], title='{}: Wall Clock Time vs Iterations_2'.format(problem_name), ylabel='Time (s)')

    def fitness_size_plot(self, problem_name):
        fitness_lst = []
        time_lst = []
        for size in range(10, 201, 10):
            if problem_name == 'FlipFlop':
                problem_fit = mh.FlipFlopGenerator.generate(seed=self.seed, size=size)
            elif problem_name == 'TSP':
                size = int(size / 2)
                problem_fit = mh.TSPGenerator.generate(seed=self.seed, number_of_cities=size)
            elif problem_name == 'Queens':
                problem_fit = mh.QueensGenerator.generate(seed=self.seed, size=size)
            elif problem_name == 'ContinuousPeaks':
                problem_fit = mh.ContinuousPeaksGenerator.generate(seed=self.seed, size=size)
            else:
                print('No such problem defined')
                return
            fitness_ga, time_ga, fit_curve_ga = self.ga(problem_fit,max_attempts=10, max_iters=500)
            fitness_sa, time_sa, fit_curve_sa = self.sa(problem_fit,max_attempts=10, max_iters=500)
            fitness_rhc, time_rhc, fit_curve_rhc = self.rhc(problem_fit,max_attempts=10, max_iters=500)
            fitness_mmc, time_mmc, fit_curve_mmc = self.mmc(problem_fit,max_attempts=10, max_iters=500)
            fitness_lst.append([size, fitness_ga, fitness_sa, fitness_rhc, fitness_mmc])
            time_lst.append([size, time_ga, time_sa, time_rhc, time_mmc])
        fitness_df = pd.DataFrame(fitness_lst, columns=['Size', 'GA', 'SA', 'RHC', "MIMIC"])
        fitness_df.set_index('Size', inplace=True)
        if problem_name == 'TSP' or problem_name == 'Queens':
            fitness_df = 1.0/fitness_df
        self.pltsavefig(fitness_df, title='{}: Fitness vs Problem Size'.format(problem_name), ylabel='Fitness')

        time_df = pd.DataFrame(time_lst, columns=['Size', 'GA', 'SA', 'RHC', "MIMIC"])
        time_df.set_index('Size', inplace=True)
        self.pltsavefig(time_df, title='{}:Wall Clock Time vs Problem Size'.format(problem_name), ylabel='Time')
        self.pltsavefig(time_df[['GA', 'SA', 'RHC']], title='{}:Wall Clock Time vs Problem Size 2'.format(problem_name), ylabel='Time (s)')

    def compare_algos(self):
        self.fitness_iter_plot(problem_fit=mh.FlipFlopGenerator.generate(seed=self.seed, size=self.num), problem_name='FlipFlop')
        self.fitness_size_plot(problem_name='FlipFlop')

        self.fitness_iter_plot(problem_fit=mh.ContinuousPeaksGenerator.generate(seed=self.seed, size=self.num2),
                          problem_name='ContinuousPeaks')
        self.fitness_size_plot(problem_name='ContinuousPeaks')

        self.fitness_iter_plot(problem_fit=mh.TSPGenerator.generate(seed=self.seed, number_of_cities=self.num), problem_name='TSP')
        self.fitness_size_plot(problem_name='TSP')

        # self.fitness_iter_plot(problem_fit=mh.QueensGenerator.generate(seed=self.seed, size=self.num), problem_name='Queens')
        # self.fitness_size_plot(problem_name='Queens')

    def fitness_curve(self):
        problem_fit = mh.FlipFlopGenerator.generate(seed=self.seed, size=self.num)
        # problem_name = 'FlipFlop'
        fitness_ga, time_ga, fit_curve_ga = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=1000, state_fitness_callback=None, curve = True)
        fitness_sa, time_sa, fit_curve_sa = mh.simulated_annealing(problem_fit, max_attempts=10, max_iters=1000, state_fitness_callback=None, curve = True)
        fitness_rhc, time_rhc, fit_curve_rhc = mh.random_hill_climb(problem_fit, max_attempts=10, max_iters=1000, state_fitness_callback=None, curve = True)
        fitness_mmc, time_mmc, fit_curve_mmc = mh.mimic(problem_fit, max_attempts=10, max_iters=1000, state_fitness_callback=None, curve = True)
        fitness_ga_df = pd.DataFrame(fit_curve_ga, columns=['GA Fitness', 'Evaluations'])
        fitness_sa_df = pd.DataFrame(fit_curve_sa, columns=['SA Fitness', 'Evaluations'])
        fitness_rhc_df = pd.DataFrame(fit_curve_rhc, columns=['RHC Fitness', 'Evaluations'])
        fitness_mmc_df = pd.DataFrame(fit_curve_mmc, columns=['MIMIC Fitness', 'Evaluations'])
        fitness_ga_df.set_index('Evaluations', inplace=True)
        fitness_sa_df.set_index('Evaluations', inplace=True)
        fitness_rhc_df.set_index('Evaluations', inplace=True)
        fitness_mmc_df.set_index('Evaluations', inplace=True)
        # fitness_ga_df.merge()
        # print(fitness_ga_df.head())
        plt.subplot()
        ax = fitness_ga_df.plot()
        fitness_sa_df.plot(ax = ax)
        fitness_rhc_df.plot(ax = ax)
        fitness_mmc_df.plot(ax = ax)

        plt.savefig('evaluation.png')
    def rhc_hppm(self):
        # problem_fit = mh.ContinuousPeaksGenerator.generate(seed=self.seed, size=self.num2)
        problem_fit=mh.FlipFlopGenerator.generate(seed=self.seed, size=self.num)

        fitness_mt_lst = []
        for iter in range(20, 1000, 200):
            best_state1, best_fitness1, fitness_curve = mh.random_hill_climb(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, restarts = 0)
            best_state2, best_fitness2, fitness_curve = mh.random_hill_climb(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state,  restarts = 1)
            best_state3, best_fitness3, fitness_curve = mh.random_hill_climb(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state,  restarts = 3)
            best_state4, best_fitness4, fitness_curve = mh.random_hill_climb(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state,  restarts = 5)
            fitness_mt_lst.append([iter, best_fitness1, best_fitness2, best_fitness3, best_fitness4])
        fitness_mt_df = pd.DataFrame(fitness_mt_lst, columns=['Iterations', 'restarts 0', 'restarts 1', 'restarts 3' , 'restarts 5'])
        fitness_mt_df.set_index('Iterations', inplace=True)

        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(1,1,1)
        fitness_mt_df.plot(ax=ax1)
        ax1.set_title('RHC: restarts')
        ax1.set(xlabel = 'Iteration', ylabel='Fitness')
        ax1.grid()
        fig.savefig('RHC_Hyperparamter.png')
    def ga_hppm(self):
        problem_fit = mh.TSPGenerator.generate(seed=self.seed, number_of_cities=self.num)
        # problem_fit=mh.FlipFlopGenerator.generate(seed=self.seed, size=self.num)
        fitness_mt_lst = []
        fitness_pb_lst = []
        fitness_pz_lst = []
        fitness_ed_lst = []
        for iter in range(20, 1000, 200):
            best_state1, best_fitness1, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, mutation_prob=0.1 )
            best_state2, best_fitness2, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, mutation_prob=0.3 )
            best_state3, best_fitness3, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, mutation_prob=0.5 )
            best_state4, best_fitness4, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, mutation_prob=0.8 )
            fitness_mt_lst.append([iter, best_fitness1, best_fitness2, best_fitness3, best_fitness4])
        fitness_mt_df = pd.DataFrame(fitness_mt_lst, columns=['Iterations', 'Mutate 0.1', 'Mutate 0.3', 'Mutate 0.5', 'Mutate 0.8'])
        fitness_mt_df.set_index('Iterations', inplace=True)
        fitness_mt_df = 1.0 / fitness_mt_df
        for iter in range(20, 1000, 200):
            best_state1, best_fitness1, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_breed_percent=0.95 )
            best_state2, best_fitness2, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_breed_percent=0.85 )
            best_state3, best_fitness3, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_breed_percent=0.75 )
            best_state4, best_fitness4, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_breed_percent=0.65 )
            fitness_pb_lst.append([iter, best_fitness1, best_fitness2, best_fitness3, best_fitness4])
        fitness_pb_df = pd.DataFrame(fitness_pb_lst, columns=['Iterations', 'pop_breed_percent 0.95', 'pop_breed_percent 0.85', 'pop_breed_percent 0.75', 'pop_breed_percent 0.65'])
        fitness_pb_df.set_index('Iterations', inplace=True)
        fitness_pb_df = 1.0 / fitness_pb_df
        for iter in range(20, 1000, 200):
            best_state1, best_fitness1, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, elite_dreg_ratio=0.99 )
            best_state2, best_fitness2, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, elite_dreg_ratio=0.9 )
            best_state3, best_fitness3, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, elite_dreg_ratio=0.8 )
            best_state4, best_fitness4, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, elite_dreg_ratio=0.7 )
            fitness_ed_lst.append([iter, best_fitness1, best_fitness2, best_fitness3, best_fitness4])
        fitness_ed_df = pd.DataFrame(fitness_ed_lst, columns=['Iterations', 'elite_dreg_ratio 0.99', 'elite_dreg_ratio 0.9', 'elite_dreg_ratio 0.8', 'elite_dreg_ratio 0.7'])
        fitness_ed_df.set_index('Iterations', inplace=True)
        fitness_ed_df = 1.0 / fitness_ed_df
        for iter in range(20, 1000, 200):
            best_state1, best_fitness1, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_size=50 )
            best_state2, best_fitness2, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_size=100 )
            best_state3, best_fitness3, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_size=200 )
            best_state4, best_fitness4, fitness_curve = mh.genetic_alg(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_size=300 )
            fitness_pz_lst.append([iter, best_fitness1, best_fitness2, best_fitness3, best_fitness4])
        fitness_pz_df = pd.DataFrame(fitness_pz_lst, columns=['Iterations', 'pop_size 50', 'pop_size 100', 'pop_size 200', 'pop_size 300'])
        fitness_pz_df.set_index('Iterations', inplace=True)
        fitness_pz_df = 1.0 / fitness_pz_df
        fig = plt.figure(figsize=(12,12))

        ax1 = fig.add_subplot(2,2,1)
        fitness_pz_df.plot(ax=ax1)
        ax1.set_title('GA: pop_size')
        ax1.set(xlabel = 'Iteration', ylabel='Fitness')
        ax1.grid()

        ax2 = fig.add_subplot(2, 2, 2)
        fitness_pb_df.plot(ax=ax2)
        ax2.set_title('GA: pop_breed_percent')
        ax2.set(xlabel='Iteration', ylabel='Fitness')
        ax2.grid()

        ax3 = fig.add_subplot(2, 2, 3)
        fitness_ed_df.plot(ax=ax3)
        ax3.set_title('GA: elite_dreg_ratio')
        ax3.set(xlabel='Iteration', ylabel='Fitness')
        ax3.grid()

        ax4 = fig.add_subplot(2, 2, 4)
        fitness_mt_df.plot(ax=ax4)
        ax4.set_title('GA: Mutation')
        ax4.set(xlabel='Iteration', ylabel='Fitness')
        ax4.grid()

        fig.savefig('GA_Hyperparamter.png')

    def sa_hppm(self):
        # problem_fit = mh.ContinuousPeaksGenerator.generate(seed=self.seed, size=self.num2)
        problem_fit=mh.FlipFlopGenerator.generate(seed=self.seed, size=self.num)

        fitness_mt_lst = []
        fitness_ma_lst = []
        for iter in range(20, 1000, 200):
            best_state1, best_fitness1, fitness_curve = mh.simulated_annealing(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, schedule=GeomDecay(min_temp=0.01))
            best_state2, best_fitness2, fitness_curve = mh.simulated_annealing(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, schedule=GeomDecay(min_temp=0.1))
            best_state3, best_fitness3, fitness_curve = mh.simulated_annealing(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, schedule=GeomDecay(min_temp=0.3))
            best_state4, best_fitness4, fitness_curve = mh.simulated_annealing(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, schedule=GeomDecay(min_temp=0.5))
            fitness_mt_lst.append([iter, best_fitness1, best_fitness2, best_fitness3, best_fitness4])
        fitness_mt_df = pd.DataFrame(fitness_mt_lst, columns=['Iterations', 'min_temp 0.01', 'min_temp 0.1', 'min_temp 0.3', 'min_temp 0.5'])
        fitness_mt_df.set_index('Iterations', inplace=True)
        for iter in range(20, 1000, 200):
            best_state1, best_fitness1, fitness_curve = mh.simulated_annealing(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, schedule=GeomDecay(decay=0.99))
            best_state2, best_fitness2, fitness_curve = mh.simulated_annealing(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, schedule=GeomDecay(decay=0.95))
            best_state3, best_fitness3, fitness_curve = mh.simulated_annealing(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, schedule=GeomDecay(decay=0.90))
            best_state4, best_fitness4, fitness_curve = mh.simulated_annealing(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, schedule=GeomDecay(decay=0.85))
            fitness_ma_lst.append([iter, best_fitness1, best_fitness2, best_fitness3, best_fitness4])
        fitness_ma_df = pd.DataFrame(fitness_ma_lst, columns=['Iterations', 'decay 0.99', 'decay 0.95', 'decay 0.90', 'decay 0.85'])
        fitness_ma_df.set_index('Iterations', inplace=True)

        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(1,2,1)
        fitness_mt_df.plot(ax=ax1)
        ax1.set_title('SA: min_temp')
        ax1.set(xlabel = 'Iteration', ylabel='Fitness')
        ax1.grid()
        ax2 = fig.add_subplot(1, 2, 2)
        fitness_ma_df.plot(ax=ax2)
        ax2.set_title('SA: decay')
        ax2.set(xlabel='Iteration', ylabel='Fitness')
        ax2.grid()
        fig.savefig('SA_Hyperparamter.png')
    def mimic_hppm(self):
        problem_fit=mh.FlipFlopGenerator.generate(seed=self.seed, size=self.num)
        fitness_mt_lst = []
        fitness_ma_lst = []
        for iter in range(20, 1000, 200):
            best_state1, best_fitness1, fitness_curve = mh.mimic(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, keep_pct=0.1 )
            best_state2, best_fitness2, fitness_curve = mh.mimic(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, keep_pct=0.3 )
            best_state3, best_fitness3, fitness_curve = mh.mimic(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, keep_pct=0.5 )
            best_state4, best_fitness4, fitness_curve = mh.mimic(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, keep_pct=0.8 )
            fitness_mt_lst.append([iter, best_fitness1, best_fitness2, best_fitness3, best_fitness4])
        fitness_mt_df = pd.DataFrame(fitness_mt_lst, columns=['Iterations', 'keep_pct 0.1', 'keep_pct 0.3', 'keep_pct 0.5', 'keep_pct 0.8'])
        fitness_mt_df.set_index('Iterations', inplace=True)
        for iter in range(20, 1000, 200):
            best_state1, best_fitness1, fitness_curve = mh.mimic(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_size=50 )
            best_state2, best_fitness2, fitness_curve = mh.mimic(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_size=100 )
            best_state3, best_fitness3, fitness_curve = mh.mimic(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_size=200 )
            best_state4, best_fitness4, fitness_curve = mh.mimic(problem_fit, max_attempts=10, max_iters=iter, random_state=self.state, pop_size=300 )
            fitness_ma_lst.append([iter, best_fitness1, best_fitness2, best_fitness3, best_fitness4])
        fitness_ma_df = pd.DataFrame(fitness_ma_lst, columns=['Iterations', 'pop_size 50', 'pop_size 100', 'pop_size 200', 'pop_size 300'])
        fitness_ma_df.set_index('Iterations', inplace=True)
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(1,2,1)
        fitness_mt_df.plot(ax=ax1)
        ax1.set_title('MIMIC: keep_pct')
        ax1.set(xlabel = 'Iteration', ylabel='Fitness')
        ax1.grid()
        ax2 = fig.add_subplot(1, 2, 2)
        fitness_ma_df.plot(ax=ax2)
        ax2.set_title('MIMIC: pop_size')
        ax2.set(xlabel='Iteration', ylabel='Fitness')
        ax2.grid()
        fig.savefig('MIMIC_Hyperparamter.png')

    def hyperparametertuning(self):
        self.rhc_hppm()
        self.ga_hppm()
        self.sa_hppm()
        self.mimic_hppm()

if __name__=='__main__':
    Opt = OptAlgos()
    # Opt.compare_algos()
    # Opt.fitness_iter_plot(problem_fit=mh.ContinuousPeaksGenerator.generate(seed=5, size=80), problem_name='ContinuousPeaks')
    # Opt.fitness_iter_plot(problem_fit=mh.TSPGenerator.generate(seed=5, number_of_cities=50), problem_name='TSP')
    Opt.fitness_size_plot(problem_name='TSP')
    # Opt.rhc_hppm()
    # Opt.ga_hppm()
    # Opt.sa_hppm()
    # Opt.mimic_hppm()
    # Opt.fitness_curve()


