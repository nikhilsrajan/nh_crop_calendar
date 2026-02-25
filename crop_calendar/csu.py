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
    days_to_maturity = np.zeros(shape=temp_ts.shape)
    gdd_at_maturity = np.zeros(shape=temp_ts.shape, dtype=float)

    for start_index in range(N):
        GDD_i = 0
        for iter_index in range(start_index, N):
            cur_temp = temp_ts[iter_index]
            if cur_temp >= max_tolerable_temp or cur_temp <= min_tolerable_temp:
                days_to_maturity[start_index] = -1
                gdd_at_maturity[start_index] = np.inf
                break
            if cur_temp - t_base > 0:
                GDD_i += cur_temp - t_base
            days_to_maturity[start_index] += 1
            gdd_at_maturity[start_index] = GDD_i
            if GDD_i > required_gdd:
                break
        if GDD_i < required_gdd:
            days_to_maturity[start_index] = -1

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

    days_to_maturity_2d = np.zeros(shape = temp_ts_2d.shape)
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


@numba.njit(parallel=True)
def total_prec_in_days_to_maturity_1d(
    cumsum_prec_ts:np.ndarray, 
    days_to_maturity_ts:np.ndarray,
):
    total_prec_in_d2m = np.full(shape=cumsum_prec_ts.shape, fill_value=np.nan)
    N = cumsum_prec_ts.shape[0]
    for start_index in numba.prange(N):
        d2m = days_to_maturity_ts[start_index]
        upto_index = d2m + start_index
        if upto_index < N and d2m != -1:
            total_prec_in_d2m[start_index] = cumsum_prec_ts[upto_index] - cumsum_prec_ts[start_index]
    return total_prec_in_d2m


@numba.njit(parallel=True)
def total_prec_in_days_to_maturity(cumsum_prec_ts:np.ndarray, days_to_maturity:np.ndarray):
    n_ts, height, width = cumsum_prec_ts.shape

    N = height * width

    cumsum_prec_ts_2d = cumsum_prec_ts.reshape(n_ts, N)
    days_to_maturity_2d = days_to_maturity.reshape(n_ts, N)
    total_prec_in_d2m_2d = np.full(shape=(n_ts, N), fill_value=np.nan)

    for i in numba.prange(N):
        total_prec_in_d2m_2d[:, i] = \
        total_prec_in_days_to_maturity_1d(
            cumsum_prec_ts = cumsum_prec_ts_2d[:, i],
            days_to_maturity_ts = days_to_maturity_2d[:, i]
        )
    
    total_prec_in_d2m = total_prec_in_d2m_2d.reshape(n_ts, height, width)

    return total_prec_in_d2m


@numba.njit(parallel=True)
def lookup_1d(
    value_arr:np.ndarray, 
    shift_arr:np.ndarray,
):
    value_at_index_arr = np.full(shape=shift_arr.shape, fill_value=np.nan)
    N = shift_arr.shape[0]
    for start_index in numba.prange(N):
        shift = shift_arr[start_index]
        lookup_index = start_index + shift
        if lookup_index < N and shift != -1:
            value_at_index_arr[start_index] = value_arr[lookup_index]
    return value_at_index_arr


def lookup(
    value_arr:np.ndarray, 
    shift_arr:np.ndarray,
):
    n_ts, height, width = shift_arr.shape

    N = height * width

    value_arr_2d = value_arr.reshape(n_ts, N)
    shift_arr_2d = shift_arr.reshape(n_ts, N)
    value_at_index_arr_2d = np.full(shape=(n_ts, N), fill_value=np.nan)

    for i in numba.prange(N):
        value_at_index_arr_2d[:, i] = \
        lookup_1d(
            value_arr = value_arr_2d[:, i],
            shift_arr = shift_arr_2d[:, i]
        )
    
    value_at_index_arr = value_at_index_arr_2d.reshape(n_ts, height, width)

    return value_at_index_arr
