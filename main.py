import unittest
from TestSignalDetection import TestSignalDetection
from SignalDetection import SignalDetection


def main():
    # Show example output
    print("Example output for different uses of the newly addded class methods:")
    # run simulation
    dPrime = 1.5
    criteriaList = [0, 0.5, -0.9, -1.1, 0.2, -0.1, 0.4, 0, 1.6, -1.4]
    signalCount = 1000
    noiseCount = 1000
    sdtList = SignalDetection.simulate(
        dPrime, criteriaList, signalCount, noiseCount)
    # Now plot data from list of SDT objects
    print("\tPrinting plot_roc and fit_roc...")
    SignalDetection.plot_roc(sdtList)
    SignalDetection.fit_roc(sdtList)

    # run the unit tests
    print("\nNow Running Unit Tests...")
    unittest.main()


if (__name__ == "__main__"):
    main()
