import bayes_logistic as bl
import numpy as np
import sys
sys.path.insert(0, '..')


class User(object):
    def __init__(self, num_objectives=2, std_noise=0.001, random_state=1, weights=None):
        # Hack to normalize utilites to 0 - 1 range
        # This works only for the BountyfulSeaTreasureEnv
        # TODO: Find another way to normalize that is not hardcoded.
        self.max_utility = 1  # Don't normalize

        self.random_state = random_state

        if weights != None:
            self.hidden_weights = weights

        else:
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

    def save_comparison(self, p1, p2, result):
        """
        p1 and p2 are the values of the policies
        result is 1 if user preferes p1 over p2, 0 otherwise 
        """
        diff = p1 - p2
        self.comparisons.append(diff)
        self.outcomes.append(float(result))
        # print("Current dataset: ")
        # print(self.comparisons)
        # print(self.outcomes)

    def compare(self, p1, p2, with_noise=True):
        """
        Compare the policies p1 and p2 and returns the prefered and rejected ones
        
        """
        scalar_p1 = self.get_utility(p1.returns, with_noise=with_noise)
        scalar_p2 = self.get_utility(p2.returns, with_noise=with_noise)
        res = scalar_p1 >= scalar_p2  # 1 if p1 > p2 else 0
        prefered, rejected = (p1, p2) if res else (p2, p1)
        self.save_comparison(p1.returns, p2.returns, res)
        return prefered, rejected

    def current_map(self, weights):
        if len(self.outcomes) > 0:
            # try :
            # clf = linear_model.LogisticRegression(C=1e5)
            # clf.fit(self.previous_comparisons, self.previous_outcomes)
            # unnorm_w = clf.coef_[0]
            w_prior = np.ones(len(self.hidden_weights)) / \
                len(self.hidden_weights)
            # w_prior = weights
            H_prior_diag = np.ones(
                len(self.hidden_weights)) * (1.0 / 0.33) ** 2
            w_fit, H_fit = bl.fit_bayes_logistic(np.array(self.outcomes),
                                                 np.array(self.comparisons),
                                                 w_prior,
                                                 H_prior_diag)
            unnorm_w = w_fit
            # except:
            #     unnorm_w = np.array([random.random() for x in range(len(self.weights))])
            #     w_fit = unnorm_w
            #     H_fit = None
            sum_w = sum(unnorm_w)
            return unnorm_w / sum_w
        else:
            result = np.ones(len(self.hidden_weights))
            for i in range(len(self.hidden_weights)):
                result[i] = result[i] / float(len(self.hidden_weights))
            return result
