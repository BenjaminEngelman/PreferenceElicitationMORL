import sys
sys.path.insert(0, '..')

import numpy as np
import bayes_logistic as bl



class User(object):
    def __init__(self, num_objectives=2, std_noise=0.001, random_state=1):
        # Hack to normalize utilites to 0 - 1 range
        # This works only for the BountyfulSeaTreasureEnv
        # TODO: Find another way to normalize that is not hardcoded.
        self.max_utility = 1  # Don't normalize

        self.random_state = random_state

        self.num_objectives = num_objectives
        self.hidden_weights = self.random_state.uniform(0.0, 1, num_objectives)
        self.hidden_weights /= np.sum(self.hidden_weights)
        self.std_noise = std_noise

        # Save all comparaisons between policies 
        # As well as their outcomes (i.e preferences)
        self.comparisons = []
        self.outcomes = []

    def get_utility(self, values, with_noise=True):
        noise = self.random_state.uniform(0, self.std_noise)
        utility = 0
        for i in range(self.num_objectives):
            utility += values[i] * self.hidden_weights[i]

        utility /= self.max_utility
        if with_noise:
            utility += noise

        return utility

    def compare(self, p1, p2, with_noise=True):
        """
        Compare the policies p1 and p2 and returnrs
            1 if it prefers p1
            0 if it prefers p2 
        """
        scalar_p1 = self.get_utility(p1, with_noise=with_noise)
        scalar_p2 = self.get_utility(p2, with_noise=with_noise)
        return scalar_p1 >= scalar_p2

    def current_map(self):
        if len(self.previous_outcomes) > 0:
            # try :
            # clf = linear_model.LogisticRegression(C=1e5)
            # clf.fit(self.previous_comparisons, self.previous_outcomes)
            # unnorm_w = clf.coef_[0]
            w_prior = np.ones(len(self.weights)) / len(self.weights)
            H_prior_diag = np.ones(len(self.weights)) * (1.0 / 0.33) ** 2
            w_fit, H_fit = bl.fit_bayes_logistic(np.array(self.previous_outcomes),
                                                 np.array(
                                                     self.previous_comparisons),
                                                 w_prior,
                                                 H_prior_diag)
            unnorm_w = w_fit
            # except:
            #     unnorm_w = np.array([random.random() for x in range(len(self.weights))])
            #     w_fit = unnorm_w
            #     H_fit = None
            sum_w = sum(unnorm_w)
            return unnorm_w / sum_w, w_fit, H_fit
        else:
            result = np.ones(len(self.weights))
            for i in range(len(self.weights)):
                result[i] = result[i] / float(len(self.weights))
            return result, result, None
