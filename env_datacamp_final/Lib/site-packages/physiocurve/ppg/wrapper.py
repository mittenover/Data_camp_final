from physiocurve.pressure import Pressure


class PPG(Pressure):
    def __init__(self, values, samplerate):
        super().__init__(values, samplerate)

    def sqi(self):
        raise NotImplementedError
