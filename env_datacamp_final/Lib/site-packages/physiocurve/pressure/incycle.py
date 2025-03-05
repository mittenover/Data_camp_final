import numpy as np


def find_dia_sys(values, samplerate, feetidx):
    starts = feetidx[0:-1]
    stops = feetidx[1:]
    windowsize = int(0.05 * samplerate)
    nvalues = len(starts)
    dia = np.zeros(nvalues, dtype=np.int64)
    sys = np.zeros(nvalues, dtype=np.int64)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        duration = stop - start
        diastop = start - duration
        dia[i] = find_local_extreme(values, start, diastop, windowsize)
        sys[i] = find_local_extreme(values, start, stop, windowsize, searchmax=True)
    return (dia, sys)


def distance(l1, l2, p):
    return np.cross(l2 - l1, p - l1)  # / np.linalg.norm(l2 - l1)


def find_dicrotics(values, diaidx, sysidx):
    # Start from sys
    for n, d in enumerate(diaidx):  # noqa: B007
        if d > sysidx[0]:
            break
    else:
        # No break condition -> no cycle
        return np.zeros(0, dtype=int)
    diaidx = diaidx[n:]

    nvalues = len(diaidx)
    dics = np.zeros(nvalues, dtype=np.int64)
    for i, (di, si) in enumerate(zip(diaidx, sysidx)):
        zoi = values[si:di]
        if not len(zoi):
            continue
        p1 = np.array([0, zoi[0]])
        p2 = np.array([di - si, zoi[-1]])

        zoiidx = np.arange(len(zoi))
        p3 = np.vstack((zoiidx, zoi)).T

        d = distance(p1, p2, p3)
        if not len(d):
            continue
        dics[i] = si + np.argmin(d)
    return dics


def find_local_extreme(a, begin, end, windowsize, searchmax=False):
    if searchmax:
        fextr = np.max
        fargextr = np.argmax
        fcomp = np.less_equal
        curext = -np.inf
    else:  # searchmin
        fextr = np.min
        fargextr = np.argmin
        fcomp = np.greater_equal
        curext = np.inf
    direction = np.sign(end - begin)
    if direction == 0:
        return a

    step = windowsize * direction
    nwindows = int((end - begin) / step)
    starts = begin + np.arange(nwindows) * step
    if not len(starts):
        return 0
    stops = starts + step

    goodzoi = np.zeros(1, dtype=a.dtype)
    for start, stop in zip(starts, stops):
        if direction > 0:
            newzoi = a[start:stop]
        else:
            newzoi = a[stop:start]
        if not len(newzoi):
            continue
        newext = fextr(newzoi)
        if fcomp(newext, curext):
            break  # Past local extreme
        curext = newext
        goodzoi = newzoi
    ret = start - step + fargextr(goodzoi)
    if direction < 0:
        ret = ret - len(goodzoi)
    return ret


def calc_quality_index(values, diaidx, sysidx, map, hr, samplerate):
    # Sun et al. Computers in Cardiology 2006
    # Not sure if this is implemented correctely, especially the slopes part
    startidx = diaidx[:-1]
    endidx = diaidx[1:]
    sqi = np.zeros(len(diaidx), dtype=bool)

    sys = values[sysidx]
    dia = values[diaidx]
    pp = sys - dia
    dsys = np.diff(sys)
    ddia = np.diff(dia)
    ddt = np.diff(np.diff(dia) / samplerate)
    dvalues = np.diff(values)

    params = zip(startidx, endidx, sys, dia, map, pp, dsys, ddia, ddt, hr)
    for (
        i,
        (istart, istop, isys, idia, imean, ipp, idsys, iddia, iddt, ihr),
    ) in enumerate(params):
        criteria = np.zeros(9)
        criteria[0] = isys > 300
        criteria[1] = idia < 20
        criteria[2] = imean < 30 or imean > 200
        criteria[3] = ihr < 20 or ihr > 200
        criteria[4] = ipp < 20
        slopes = dvalues[istart:istop]
        negslopes = slopes[slopes < 0]
        negslope = np.nanmean(negslopes) if len(negslopes > 0) else 0
        criteria[5] = negslope < -40
        criteria[6] = abs(idsys) > 20
        criteria[7] = abs(iddia) > 20
        criteria[8] = abs(iddt) > 0.66
        sqi[i] = np.any(criteria)
    return sqi
