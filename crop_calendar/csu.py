import numba
import numpy as np


@numba.njit(parallel=True)
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

    for start_index in range(N):
        GDD_i = 0
        for iter_index in range(start_index, N):
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


@numba.njit(parallel=True)
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

    temp_ts_2d = temp_ts.reshape(n_ts, height*width)
    N = height * width    

    days_to_maturity_2d = np.zeros(shape = temp_ts_2d.shape, dtype=float)
    gdd_at_maturity_2d = np.zeros(shape = temp_ts_2d.shape, dtype=float)

    for i in numba.prange(N):
        days_to_maturity_2d[:, i], gdd_at_maturity_2d[:, i] = \
        calculate_days_to_maturity_1d(
            temp_ts = temp_ts_2d[:, i],
            t_base = t_base,
            required_gdd = required_gdd,
            max_tolerable_temp = max_tolerable_temp,
            min_tolerable_temp = min_tolerable_temp,
        )
    
    days_to_maturity = days_to_maturity_2d.reshape(n_ts, height, width)
    gdd_at_maturity = gdd_at_maturity_2d.reshape(n_ts, height, width)

    return days_to_maturity, gdd_at_maturity
