import numba
import numpy as np

@numba.njit()
def calculate_days_to_maturity_1d(
    temp_ts:np.ndarray,
    t_base:float,
    required_gdd:float,
    max_tolerable_temp:float,
    min_tolerable_temp:float,
):
    N = temp_ts.shape[0]
    days_to_maturity = np.zeros(shape=temp_ts.shape, dtype=float)
    gdd_at_maturity = np.zeros(shape=temp_ts.shape, dtype=float)

    for start_index in numba.prange(N):
        GDD_i = 0
        for iter_index in numba.prange(start_index, N):
            cur_temp = temp_ts[iter_index]
            if cur_temp >= max_tolerable_temp or cur_temp <= min_tolerable_temp:
                days_to_maturity[start_index] = np.inf
                gdd_at_maturity[start_index] = np.inf
                break
            if cur_temp - t_base > 0:
                GDD_i += cur_temp - t_base
            days_to_maturity[start_index] += 1
            gdd_at_maturity[start_index] = GDD_i
            if GDD_i > required_gdd:
                break
        if GDD_i < required_gdd:
            days_to_maturity[start_index] = np.inf

    return days_to_maturity, gdd_at_maturity


@numba.njit()
def calculate_days_to_maturity(
    temp_ts:np.ndarray,
    t_base:float,
    required_gdd:float,
    max_tolerable_temp:float,
    min_tolerable_temp:float,
):
    """
    temp_ts is a 3d array - (timestamps, height, width)
    """
    n_ts, height, width = temp_ts.shape
    days_to_maturity = np.zeros(shape = temp_ts.shape, dtype=float)
    gdd_at_maturity = np.zeros(shape = temp_ts.shape, dtype=float)
    for h in numba.prange(height):
        for w in numba.prange(width):
            days_to_maturity[:, h, w], gdd_at_maturity[:, h, w] = \
            calculate_days_to_maturity_1d(
                temp_ts = temp_ts[:, h, w],
                t_base = t_base,
                required_gdd = required_gdd,
                max_tolerable_temp = max_tolerable_temp,
                min_tolerable_temp = min_tolerable_temp,
            )
    
    return days_to_maturity, gdd_at_maturity
