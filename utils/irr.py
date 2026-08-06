from typing import List

import numpy as np
import scipy.optimize as optimize


def npv(cf: List, rate: float = 0.1) -> float:
    """
    Returns net present value of given cashflows

    Args:
        cf (List): List of lists with dates and cashflows
        rate (float, optional): Interest rate to calculate net present value. Defaults to 0.1.

    Returns:
        float: Value representing net present value of given cashflows
    """

    if len(cf) >= 2:
        first_date = min([x[0] for x in cf])
        dcf = [x[1] * (1 / ((1 + rate) ** ((x[0] - first_date).days / 365))) for x in cf]
        return sum(dcf)
    elif len(cf) == 1:
        return cf[0][1]
    else:
        return 0


def irr(cf: List) -> float:
    """
    Calculates internal rate of return for given cash flows

    Args:
        cf (List): List of lists with dates and cashflows

    Returns:
        float: Value representing net present value of given cashflows
    """

    def f(x): return npv(cf, rate=x)
    r = optimize.root(f, [0])
    if r.get('success'):
        return round(r.get('x')[0], 4)
    else:
        return np.NaN
