from physiocurve.common import estimate_samplerate
from physiocurve.ecg import ECG as ECGNp


class ECG(ECGNp):
    def __init__(self, series, samplerate=None):
        self._series = series
        if samplerate is None:
            samplerate = estimate_samplerate(series)

        super().__init__(series.to_numpy(), samplerate)

    @property
    def idxpwave(self):
        return self._series.index[self.argpwave]

    @property
    def idxqwave(self):
        return self._series.index[self.argqwave]

    @property
    def idxrwave(self):
        return self._series.index[self.argrwave]

    @property
    def idxswave(self):
        return self._series.index[self.argswave]

    @property
    def idxtwave(self):
        return self._series.index[self.argtwave]

    @property
    def pwaves(self):
        return self._series.iloc[self.argpwave]

    @property
    def qwaves(self):
        return self._series.iloc[self.argqwave]

    @property
    def rwaves(self):
        return self._series.iloc[self.argrwave]

    @property
    def swaves(self):
        return self._series.iloc[self.argswave]

    @property
    def twaves(self):
        return self._series.iloc[self.argtwave]
