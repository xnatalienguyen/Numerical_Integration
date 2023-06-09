#!/usr/bin/python3.8

import numpy as np
import unittest
import matplotlib.pyplot as plt

from SignalDetection import SignalDetection


class TestSignalDetection(unittest.TestCase):
    """
    Test suite for SignalDetection class.
    """

    def test_d_prime_zero(self):
        """
        Test d-prime calculation when hits and false alarms are 0.
        """
        sd = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_d_prime_nonzero(self):
        """
        Test d-prime calculation when hits and false alarms are nonzero.
        """
        sd = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_zero(self):
        """
        Test criterion calculation when hits and false alarms are both 0.
        """
        sd = SignalDetection(5, 5, 5, 5)
        expected = 0
        obtained = sd.criterion()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_criterion_nonzero(self):
        """
        Test criterion calculation when hits and false alarms are nonzero.
        """
        sd = SignalDetection(15, 10, 15, 5)
        expected = -0.463918426665941
        obtained = sd.criterion()
        self.assertAlmostEqual(obtained, expected, places=10)

    def test_addition(self):
        """
        Test addition of two SignalDetection objects.
        """
        sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
        expected = SignalDetection(3, 2, 3, 4).criterion()
        obtained = sd.criterion()
        self.assertEqual(obtained, expected)

    def test_multiplication(self):
        """
        Test multiplication of a SignalDetection object with a scalar.
        """
        sd = SignalDetection(1, 2, 3, 1) * 4
        expected = SignalDetection(4, 8, 12, 4).criterion()
        obtained = sd.criterion()
        self.assertEqual(obtained, expected)

    def test_simulate_single_criterion(self):
        """
        Test SignalDetection.simulate method with a single criterion value.
        """
        dPrime = 1.5
        criteriaList = [0]
        signalCount = 1000
        noiseCount = 1000

        sdtList = SignalDetection.simulate(
            dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 1)
        sdt = sdtList[0]

        self.assertEqual(sdt.hits, sdtList[0].hits)
        self.assertEqual(sdt.misses, sdtList[0].misses)
        self.assertEqual(sdt.falseAlarms, sdtList[0].falseAlarms)
        self.assertEqual(sdt.correctRejections, sdtList[0].correctRejections)

    def test_simulate_multiple_criteria(self):
        """
        Test SignalDetection.simulate method with multiple criterion values.
        """
        dPrime = 1.5
        criteriaList = [-0.5, 0, 0.5]
        signalCount = 1000
        noiseCount = 1000
        sdtList = SignalDetection.simulate(
            dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 3)
        for sdt in sdtList:
            self.assertLessEqual(sdt.hits,  signalCount)
            self.assertLessEqual(sdt.misses,  signalCount)
            self.assertLessEqual(sdt.falseAlarms,  noiseCount)
            self.assertLessEqual(sdt.correctRejections,  noiseCount)

    def test_nLogLikelihood(self):
        """
        Test case to verify nLogLikelihood calculation for a SignalDetection object.
        """
        sdt = SignalDetection(10, 5, 3, 12)
        hit_rate = 0.5
        false_alarm_rate = 0.2
        expected_nll = - (10 * np.log(hit_rate) +
                          5 * np.log(1-hit_rate) +
                          3 * np.log(false_alarm_rate) +
                          12 * np.log(1-false_alarm_rate))
        self.assertAlmostEqual(sdt.nLogLikelihood(hit_rate, false_alarm_rate),
                               expected_nll, places=6)

    def test_rocLoss(self):
        """
        Test case to verify rocLoss calculation for a list of SignalDetection objects.
        """
        sdtList = [
            SignalDetection(8, 2, 1, 9),
            SignalDetection(14, 1, 2, 8),
            SignalDetection(10, 3, 1, 9),
            SignalDetection(11, 2, 2, 8),
        ]
        a = 0
        expected = 99.3884
        self.assertAlmostEqual(SignalDetection.rocLoss(
            a, sdtList), expected, places=4)

    def test_integration(self):
        """
        Test case to verify integration of SignalDetection simulation and ROC fitting.
        """
        dPrime = 1
        sdtList = SignalDetection.simulate(dPrime, [-1, 0, 1], 1e7, 1e7)
        aHat = SignalDetection.fit_roc(sdtList)
        self.assertAlmostEqual(aHat, dPrime, places=2)
        plt.close()


if __name__ == '__main__':
    unittest.main()
