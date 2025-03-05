from functools import cached_property

import numpy as np
from physiocurve.pressure import foot, incycle


class Pressure:
    def __init__(self, values, samplerate):
        self._values = values
        self._samplerate = samplerate

    @property
    def samplerate(self):
        return self._samplerate

    @cached_property
    def argfeet(self):
        """Return the numerical index of feet."""
        return foot.find_derivative_feet(self._values, self._samplerate)

    @cached_property
    def argtanfeet(self):
        """Return the numerical index of feet with intersection tangent method."""
        return foot.find_tangent_feet(self._values, self.argdia, self.argsys)

    @cached_property
    def _argdiasys(self):
        return incycle.find_dia_sys(self._values, self._samplerate, self.argfeet)

    @property
    def argdia(self):
        dia, _ = self._argdiasys
        return dia

    @property
    def argsys(self):
        _, sys = self._argdiasys
        return sys

    @cached_property
    def argdic(self):
        return incycle.find_dicrotics(self._values, *self._argdiasys)

    @cached_property
    def means(self):
        feet = self.argfeet
        cycles = zip(feet, feet[1:])
        means = [np.mean(self._values[beg:end]) for beg, end in cycles]
        return np.array(means)

    @cached_property
    def heartrate(self):
        dfeet = np.diff(self.argfeet)
        dfeet_s = dfeet / self._samplerate
        # Result in beats per minute
        return 60 / dfeet_s

    @cached_property
    def sqi(self):
        return incycle.calc_quality_index(
            self._values,
            self.argdia,
            self.argsys,
            self.means,
            self.heartrate,
            self._samplerate,
        )

    @cached_property
    def nra(self):
        values = self._values
        idxok = self.argdic.astype(bool)
        dics = values[self.argdic[idxok]]

        # Remove invalid dicrotic samples from all arrays
        sysidx = np.zeros(self.argsys.shape[0], dtype=bool)
        sysidx[: idxok.shape[0]] = idxok
        syss = values[self.argsys[sysidx]]

        diaidx = np.zeros(self.argdia.shape[0], dtype=bool)
        diaidx[: idxok.shape[0]] = idxok
        dias = values[self.argdia[diaidx]]

        result = np.where((syss - dias) != 0, (dics - dias) / (syss - dias), 0)
        return result
