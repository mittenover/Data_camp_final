import numpy as np
from physiocurve.common import estimate_samplerate
from physiocurve.pressure import Pressure as PressureNp


class Pressure(PressureNp):
    def __init__(self, series, samplerate=None):
        self._series = series
        if samplerate is None:
            samplerate = estimate_samplerate(series)

        super().__init__(series.to_numpy(), samplerate)

    @property
    def idxfeet(self):
        locs = np.ma.masked_equal(self.argfeet, 0)
        return self._series.index[locs]

    @property
    def feet(self):
        return self._series.loc[self.idxfeet]

    @property
    def idxtanfeet(self):
        locs = np.ma.masked_equal(self.argtanfeet, 0)
        return self._series.index[locs]

    @property
    def tanfeet(self):
        return self._series.loc[self.idxtanfeet]

    @property
    def idxdia(self):
        locs = np.ma.masked_equal(self.argdia, 0)
        return self._series.index[locs]

    @property
    def diastolics(self):
        return self._series.loc[self.idxdia]

    @property
    def idxsys(self):
        locs = np.ma.masked_equal(self.argsys, 0)
        return self._series.index[locs]

    @property
    def systolics(self):
        return self._series.loc[self.idxsys]

    @property
    def idxdic(self):
        locs = np.ma.masked_equal(self.argdic, 0)
        return self._series.index[locs]

    @property
    def dicrotics(self):
        return self._series.loc[self.idxdic]
