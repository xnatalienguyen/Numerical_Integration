import numpy as np


class Metropolis:
    def __init__(self, logTarget, initialState, stepSize=1.0):
        self.logTarget = logTarget
        self.currentState = initialState
        self.stepSize = stepSize
        self.acceptanceRate = None
        self.samples = []

    def _accept(self, proposal):
        logProposal = self.logTarget(proposal)
        logCurrent = self.logTarget(self.currentState)
        logAlpha = logProposal - logCurrent
        if np.log(np.random.rand()) < logAlpha:
            self.currentState = proposal
            return True
        else:
            return False

    def adapt(self, blockLengths):
        for blockLength in blockLengths:
            accepts = 0
            proposals = 0
            for i in range(blockLength):
                proposal = np.random.normal(
                    loc=self.currentState, scale=self.stepSize)
                if self._accept(proposal):
                    accepts += 1
                proposals += 1
            acceptanceRate = accepts / proposals
            if acceptanceRate > 0.5:
                self.stepSize *= 1.1
            else:
                self.stepSize *= 0.9
            self.acceptanceRate = acceptanceRate
        return self

    def sample(self, nSamples):
        self.samples = [self.currentState]
        for i in range(nSamples):
            proposal = np.random.normal(
                loc=self.currentState, scale=self.stepSize)
            if self._accept(proposal):
                self.samples.append(proposal)
            else:
                self.samples.append(self.currentState)
        return self

    def summary(self):
        n = len(self.samples)
        mean = np.mean(self.samples)
        std = np.std(self.samples, ddof=1)
        ci025 = np.percentile(self.samples, 2.5)
        ci975 = np.percentile(self.samples, 97.5)
        return {'mean': mean, 'std': std, 'c025': ci025, 'c975': ci975}
