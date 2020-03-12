import math
import numpy as np
from scipy.stats import norm

"""
Reprodutction of the code of the Paper:
"Bayesian Preference Elicitation for Multiobjective Engineering Design Optimization"
W.I.P 
"""


class PrefEl(object):
    def __init__(self, n_obj):
        self.n_obj = n_obj
        self.sigma = np.identity(n_obj)
        self.data = []
        self.indif = []
        self.strict = []

    def addStrictPref(self, x, y):
        """
        :param x: The prefered option
        :param y: The other option
        :return: The strict preferences
        """
        self.strict.append([x, y])
        return self.strict

    def addIndifPref(self, x, y):
        """
        :param x: The first option
        :param y: The second option
        :return: The indiferent preferences
        """
        self.indif.append([x, y])
        return self.indif

    def calculateLogProb(self, x):
        log_prob = 0

        for strict_pref in self.strict:
            log_prob += self.getLogStrictProb(x, strict_pref[0] , strict_pref[1])
        for indif_pref in self.indif:
            log_prob += self.getLogIndifProb(x, indif_pref[0] , indif_pref[1])

        log_prob += self.logPrior(x)

    def getLogStrictProb(self, x, a, b):
        d = a - b
        varAlongD = (d * self.sigma * d.T)[0]
        meanAlongD = d.reshape(-1,).dot(x.reshape(-1,))

        return norm.logcdf(meanAlongD, 0, np.sqrt(varAlongD))

    def getLogIndifProb(self, x, a, b):
        d = a - b
        varAlongD = (d * self.sigma * d.T)[0]
        meanAlongD = d.reshape(-1,).dot(x.reshape(-1,))
        return np.log(norm.cdf(meanAlongD + 0.5, 0, np.sqrt(varAlongD)) - norm.cdf(meanAlongD - 0.5, 0, np.sqrt(varAlongD)))

    def logPrior(self, x):
        """
        For normal Prior
        :param x:
        :return:
        """
        log_prior = 0.0
        for _ in range(self.n_obj):
            # Prior is Normal(0, 1)
            log_prior += norm.pdf(x, 0, 1)
        return log_prior


