import numpy as np

# from physiocurve.common import diagnostics


# Second derivative method
def find_derivative_feet(values, samplerate):
    fstderiv = np.diff(values, 1)
    fstderiv = np.append(fstderiv, np.nan)
    sndderiv = np.diff(fstderiv, 1)
    sndderiv = np.append(sndderiv, np.nan)
    sndderiv = sndderiv * (fstderiv > 0)
    risingStarts, risingStops = sliding_search_zois(sndderiv, samplerate)
    outlen = min([len(x) for x in (risingStarts, risingStops)])
    outarr = np.zeros(outlen, dtype=np.int64)
    for i in range(outlen):
        start = risingStarts[i]
        stop = risingStops[i]
        outarr[i] = start + np.argmax(sndderiv[start:stop])
    return outarr


def roll_sum(a, w):
    n = len(a) - w + 1
    buf = np.zeros(len(a))
    for i in range(n):
        buf[i] = np.nansum(a[i : i + w])
    return buf


def roll_quant7(a, w):
    n = len(a) - w + 1
    buf = np.zeros(len(a))
    for i in range(n):
        buf[i] = np.nanquantile(a[i : i + w], 0.7)
    return buf


def sliding_search_zois(a, samplerate, sumcoef=4, quantcoef=3):
    winsum = samplerate // sumcoef
    winquant = int(samplerate * quantcoef)
    sq = a**2
    integral = roll_sum(sq, winsum)
    thres = roll_quant7(integral, winquant)
    risings = integral > thres
    risingvar = np.diff(risings.astype(np.int8))
    risingStarts = np.flatnonzero(risingvar > 0)
    risingStops = np.flatnonzero(risingvar < 0)
    risingStops = risingStops[risingStops > risingStarts[0]]
    return (risingStarts, risingStops)


# Intersecting tangents method
def find_tangent_feet(values, argdia, argsys):
    return np.rint(find_tangent_intersections(values, argdia, argsys)).astype(np.int64)


def find_tangent_intersections(values, argdia, argsys):
    outlen = min([len(x) for x in (argdia, argsys)])
    outarr = np.zeros(outlen, dtype=np.float64)

    deriv = np.diff(values, 1)
    deriv = np.append(deriv, np.nan)

    slope_intervals = np.column_stack((argdia, argsys))

    for i, (diaidx, sysidx) in enumerate(slope_intervals):
        slope, tangent_b = calc_tangent(values, deriv, diaidx, sysidx)
        if slope == 0:
            continue
        diay = values[diaidx]
        foot_x = (diay - tangent_b) / slope
        outarr[i] = foot_x

        # Tangent diagnostics
        # beg_interval, end_interval, diay, foot_x, slope, tangent_b
        # diagnostics['tangents'].append([diaidx, sysidx, diay, foot_x, slope, tangent_b])
    return outarr


def calc_tangent(values, deriv, beg_interval, end_interval):
    if beg_interval == 0 or end_interval == 0:
        return (0, 0)
    deriv_sample = deriv[beg_interval:end_interval]
    if len(deriv_sample) == 0:
        return (0, 0)

    max_slope_idx = np.argmax(deriv_sample)

    # Take a 5 sample wide segment around max_slope
    deriv_subsample_x = np.arange(max_slope_idx - 2, max_slope_idx + 3, step=1)
    deriv_subsample_y = np.take(deriv_sample, deriv_subsample_x, mode="clip")

    tangent_x = beg_interval + deriv_subsample_x.mean()
    tangent_y = np.take(values, beg_interval + deriv_subsample_x, mode="clip").mean()

    slope = deriv_subsample_y.mean()
    tangent_b = tangent_y - slope * tangent_x

    return (slope, tangent_b)
