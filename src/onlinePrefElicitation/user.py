import src.onlinePrefElicitation.relFeedback.bayes_logistic as bl
from sklearn import linear_model
import numpy as np
import sys
from src.constants import BST_SOLUTIONS, BST_MIN_TIME, BST_MIN_TREASURE
from src.ols.utils import create_3D_pareto_front

import math

sys.path.insert(0, '..')


class User(object):
    def __init__(self, num_objectives=2, noise_pct=0.1, env=None, random_state=None, weights=None, num_virtual_comps=10):
        # Hack to normalize utilites to 0 - 1 range
        # This works only for the BountyfulSeaTreasureEnv
        # TODO: Find another way to normalize that is not hardcoded.

        self.random_state = random_state
        self.num_objectives = num_objectives
        self.num_virtual_comps = num_virtual_comps

        # Save all comparaisons between policies
        # As well as their outcomes (i.e preferences)
        self.comparisons = []
        self.outcomes = []

        if weights is not None:
            assert len(weights) == num_objectives
            self.hidden_weights = weights

        else:
            self.hidden_weights = self.random_state.uniform(0.0, 1, num_objectives)
            self.hidden_weights /= np.sum(self.hidden_weights)
        
        # Trick to add noise in % to the utilites
        # Deep sea treasure
        if self.num_objectives == 2:
            utilities = [self.hidden_weights[0] * sol[0] + self.hidden_weights[1] * sol[1] for sol in BST_SOLUTIONS]
        
        # Synt 3 obj
        elif (self.num_objectives == 3) and (env != "minecart"):
            utilities = [self.hidden_weights[0] * sol[0] + self.hidden_weights[1] * sol[1] + self.hidden_weights[2] * sol[2] for sol in create_3D_pareto_front()]

        # Minecart
        else:
            utilities = [1.11, -1]


        utility_range = max(utilities) - min(utilities)
        self.std_noise = (noise_pct / 100) * utility_range
        

    def get_utility(self, values, with_noise=True):
        noise = self.random_state.normal(0, self.std_noise)
        utility = 0
        for i in range(self.num_objectives):
            utility += values[i] * self.hidden_weights[i]
        if with_noise:
            utility += noise
        
        return utility
    

    def save_comparison(self, p1, p2, result):
        """
        p1 and p2 are the values of the policies
        result is 1 if user preferes p1 over p2, 0 otherwise 
        """
        diff = p1 - p2
        for _ in range(20):
            self.comparisons.append(diff)
            self.outcomes.append(float(result))

        dominated = p2 if result else p1
        dominant = p1 if list(dominated) == list(p2) else p2
        for _ in range(self.num_virtual_comps):
            if self.num_objectives == 3:
                synthetic_p = [np.random.uniform(0,dominated[i]) for i in range(len(dominated))]
            elif self.num_objectives == 2:
                synthetic_p = [
                    np.random.uniform(BST_MIN_TREASURE, dominated[0]),
                    np.random.uniform(BST_MIN_TIME, dominated[1])
                ]

            if np.random.uniform() < 0.5:
                diff = synthetic_p - dominant 
                self.comparisons.append(diff)
                self.outcomes.append(0)
            else:
                diff = dominant -  synthetic_p
                self.comparisons.append(diff)
                self.outcomes.append(1)
        
        # print(self.comparisons)

        # print("Current dataset: ")
        # print(self.comparisons)
        # print(self.outcomes)

    def compare(self, p1, p2, with_noise=True):
        """
        Compare the policies p1 and p2 and returns the prefered and rejected ones
        
        """
        scalar_p1 = self.get_utility(p1, with_noise=with_noise)
        scalar_p2 = self.get_utility(p2, with_noise=with_noise)
        res = scalar_p1 >= scalar_p2  # 1 if p1 > p2 else 0

        prefered, rejected = (p1, p2) if res else (p2, p1)
        u_pref, u_rej = (scalar_p1, scalar_p2) if res else (scalar_p2, scalar_p1)

        self.save_comparison(p1, p2, res)
        return prefered, rejected, u_pref, u_rej

    def current_map(self, weights=None):
        if len(self.outcomes) > 0:
            
            if False:
                clf = linear_model.LogisticRegression(C=1e5)
                clf.fit(self.comparisons, self.outcomes, )
                unnorm_w = clf.coef_[0]

            else:
            
                bnd = (0, 1)
                bnd_list = []
                for _ in np.arange(self.num_objectives):
                    bnd_list.append(bnd)

                if weights is not None:
                    # w_prior = weights
                    w_prior = np.ones(len(self.hidden_weights)) / len(self.hidden_weights)


                else:
                    w_prior = np.ones(len(self.hidden_weights)) / len(self.hidden_weights)
                    # w_prior = np.zeros(self.num_objectives)


                # H_prior_diag = np.ones(self.num_objectives)*0.05
                H_prior_diag = np.ones(len(self.hidden_weights)) * (1.0 / 0.33)   ** 2
                # H_prior_diag = np.array([2] * self.num_objectives)

                w_fit, H_fit = bl.fit_bayes_logistic(
                    np.array(self.outcomes),
                    np.array(self.comparisons),
                    w_prior,
                    H_prior_diag,
                )
                # print(1 / H_fit)

                # print(samples) 
                # exit()

                unnorm_w = w_fit
            unnorm_w[unnorm_w < 0] = 0 # No negative values

            sum_w = sum(unnorm_w)
            if sum_w == 0:
                sum_w = 1e-16
            norm_w_posterior =  unnorm_w / sum_w

            self.mean_w = unnorm_w
            self.H = H_fit

            return norm_w_posterior, w_fit, H_fit
            # return norm_w_posterior, 0, 0


    def sample_weight_vector(self):
        w_vec = self.mean_w
        h_vec = self.H

        if h_vec is None:
            return w_vec
        w_sample = []
        for i in range(len(w_vec)):
            stdev = 1.0 / math.sqrt(h_vec[i])
            # print("stdev "+str(i)+" "+str(stdev))
            ws = self.random_state.normal(w_vec[i], stdev)
            w_sample.append(ws)
        w_sample = np.array(w_sample)
        return w_sample / sum(w_sample)
