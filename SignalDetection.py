from statistics import NormalDist
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math


class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
        self.hit_rate = hits / (hits + misses)
        self.false_alarm_rate = falseAlarms / (falseAlarms + correctRejections)
        self.hit_dist = NormalDist().inv_cdf(self.hit_rate)
        self.false_dist = NormalDist().inv_cdf(self.false_alarm_rate)

    def d_prime(self):
        d = self.hit_dist - (self.false_dist)
        return d

    def criterion(self):
        c = (-0.5) * ((self.hit_dist) + (self.false_dist))
        return c

    def __add__(self, other):
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.falseAlarms + other.falseAlarms, self.correctRejections + other.correctRejections)

    def __mul__(self, scalar):
        return SignalDetection(self.hits * scalar, self.misses * scalar, self.falseAlarms * scalar, self.correctRejections * scalar)

    def plot_hit_false(self):
        x = [0, self.hit_rate, 1]
        y = [0, self.false_alarm_rate, 1]
        plt.plot(x, y, 'b')
        plt.plot(self.hit_rate, self.false_alarm_rate, 'bo')
        plt.xlabel("Hit rate")
        plt.ylabel("False alarm rate")
        plt.title("ROC curve")
        plt.show()

    def plot_std(self):
        x = np.arange(-4, 4, 0.01)
        # N
        plt.plot(x, scipy.stats.norm.pdf(x, 0, 1), 'b', label="N")
        # S
        plt.plot(x, scipy.stats.norm.pdf(x, self.d_prime(), 1), 'r', label="S")
        # C
        plt.axvline((self.d_prime()/2) + self.criterion(),
                    color='black', linestyle='--').set_label("C")
        # D
        plt.plot([self.d_prime(), 0], [0.4, 0.4], 'k', label="D")

        plt.xlabel("Decision variable")
        plt.ylabel("Probability")
        plt.title("Signal Detection Theory")
        plt.legend()
        plt.show()

    @staticmethod
    def simulate(dPrime, criteriaList, signalCount, noiseCount):
        s_list = []
        for i in range(len(criteriaList)):
            k = criteriaList[i] + (dPrime / 2)
            hit_rate = 1 - scipy.stats.norm.cdf(k - dPrime)
            false_alarm_rate = 1 - scipy.stats.norm.cdf(k)
            hits = np.random.binomial(signalCount, hit_rate)
            misses = signalCount - hits
            falseAlarm = np.random.binomial(noiseCount, false_alarm_rate)
            correctRejections = noiseCount - falseAlarm
            s_list.append(SignalDetection(
                hits, misses, falseAlarm, correctRejections))
        return s_list

    @staticmethod
    def plot_roc(sdtList):
        x = list()
        y = list()
        for i in range(len(sdtList)):
            x.append(sdtList[i].false_alarm_rate)
            y.append(sdtList[i].hit_rate)
        plt.plot([0, 1], [0, 1], ls="--", c=".3")
        plt.scatter(x, y, c='k')
        plt.grid()
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit rate")
        plt.title("ROC curve")

    def nLogLikelihood(self, hitRate, falseAlarmRate):
        likelihood = (-(self.hits) * (math.log(hitRate))) - (self.misses * (math.log(1 - hitRate))) - (
            self.falseAlarms * (math.log(falseAlarmRate))) - (self.correctRejections * (math.log(1 - falseAlarmRate)))
        return likelihood

    @staticmethod
    def rocCurve(falseAlarmRate, a):
        # compute the predicted hit rate based on false alarm rate
        hitRate = scipy.stats.norm.cdf(
            a + scipy.stats.norm.ppf(falseAlarmRate))
        return (hitRate)

    @staticmethod
    def rocLoss(a, sdtList):
        # calculate the log-liklihood for each predicted hit rate to the observed false alarm rate
        loss = 0
        for sdt in sdtList:
            loss += sdt.nLogLikelihood(SignalDetection.rocCurve(
                sdt.false_alarm_rate, a), sdt.false_alarm_rate)
        return (loss)

    @staticmethod
    def fit_roc(sdtList):
        # get the optimal value for a that minimuzes the loss function
        optimization_result = scipy.optimize.minimize(
            fun=SignalDetection.rocLoss, x0=0, args=(sdtList,))
        # result is a_hat which should be close to d'
        a_hat = optimization_result.x[0]
        # Get the fitted data to plot
        falsesAlarms = np.arange(0, 1, 0.01)
        hitRates = SignalDetection.rocCurve(falsesAlarms, a_hat)
        # plot the fitted cirve
        plt.plot(falsesAlarms, hitRates, 'r')
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit rate")
        plt.title("ROC curve")
        plt.show()
        return (a_hat)
