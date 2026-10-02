"""
Functions for calculating streamflow (Q) metrics and signatures.
These functions analyze various aspects of streamflow time series including:
- Seasonal characteristics (snowmelt season CV, sum)
- Statistical properties (skewness, variance, coefficient of variation)
- Timing metrics (peak timing, half flow date, inflection points)
- Flow distribution metrics (high flow frequency, peak distribution)
- Flow regime indicators (flashiness, baseflow, amplitude)
"""

from ast import List
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from hydrosignatures import flashiness_index as fi
from hydrosignatures.baseflow import baseflow
import HydroErr as he


def calc_snowmeltseason_cv(q: pd.Series, melt_period = [4, 5, 6, 7]) -> float:
    """
    Calculate the coefficient of variation (CV) for the snowmelt season.
    
    Args:
        q (pd.Series): Streamflow time series
        melt_period: Months to include in snowmelt season (list) or DatetimeIndex for specific dates
    
    Returns:
        float: Coefficient of variation for the snowmelt season
    """
    if isinstance(melt_period, list):
        q_season = q.loc[q.index.month.isin(melt_period)]
    elif isinstance(melt_period, pd.DatetimeIndex):
        q_season = q.loc[melt_period]
    return q_season.std() / q_season.mean()

    

def calc_snowmeltseason_sum(q: pd.Series, melt_period = [4, 5, 6, 7]) -> float:
    """
    Calculate the total streamflow during the snowmelt season.
    
    Args:
        q (pd.Series): Streamflow time series
        months (list): Months to include in snowmelt season (default: [4,5,6,7])
    
    Returns:
        float: Total streamflow during snowmelt season
    """
    if isinstance(melt_period, list):
        q_season = q.loc[q.index.month.isin(melt_period)]
    elif isinstance(melt_period, pd.DatetimeIndex):
        q_season = q.loc[melt_period]
    return q_season.sum()

def calc_q_skew(q: pd.Series) -> float:
    """
    Calculate the skewness of the streamflow time series.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        float: Skewness coefficient
    """
    return q.skew()

def calc_q_var(q: pd.Series) -> float:
    """
    Calculate the variance of the streamflow time series.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        float: Variance
    """
    return q.var()

def calc_q_cov(q: pd.Series) -> float:
    """
    Calculate the coefficient of variation (CV) of the streamflow time series.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        float: Coefficient of variation (std/mean)
    """
    return q.std() / q.mean()

def calc_t_qmax(q: pd.Series) -> int:
    """
    Calculate the timing of maximum streamflow relative to October 1st.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        int: Days since October 1st of the previous year
    """
    t_qmax = q.idxmax()
    return (t_qmax - pd.Timestamp(f"{t_qmax.year-1}-10-01")).days

def calc_half_flow_date(q: pd.Series) -> int:
    """
    Calculate the half flow date (HFD) - the date when half of the annual flow has passed.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        int: Index of the half flow date, or np.nan if not found
    """
    q_half_sum = 0.5 * np.sum(q)
    q_cumsum = np.cumsum(q)
    hfd_aux = np.where(q_cumsum > q_half_sum)[0]
    return hfd_aux[0] if len(hfd_aux) > 0 else np.nan
def calc_hfd_meltseason(q: pd.Series, melt_period = [4, 5, 6, 7]) -> int:
    """
    Calculate the half flow date (HFD) for the snowmelt season.
    
    Args:
        q (pd.Series): Streamflow time series
        melt_period: Months to include in snowmelt season (list) or DatetimeIndex for specific dates
    
    Returns:
        int: Index of the half flow date, or np.nan if not found
    """
    if isinstance(melt_period, list):
        q_season = q.loc[q.index.month.isin(melt_period)]
    elif isinstance(melt_period, pd.DatetimeIndex):
        q_season = q.loc[melt_period]
    q_half_sum = 0.5 * np.sum(q_season)
    q_cumsum = np.cumsum(q_season)
    hfd_aux = np.where(q_cumsum > q_half_sum)[0]
    return hfd_aux[0] if len(hfd_aux) > 0 else np.nan

def calc_half_flow_interval(q: pd.Series) -> int:
    """
    Calculate the time span between quarter and three-quarter flow dates.
    
    The interval is defined as the time between:
    - When cumulative discharge reaches 25% of annual total
    - When cumulative discharge reaches 75% of annual total
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        int: Number of days between quarter and three-quarter flow dates
    """
    q_quarter = 0.25 * np.sum(q)
    q_three_quarters = 0.75 * np.sum(q)
    q_cumsum = np.cumsum(q)
    hfi_aux1 = np.where(q_cumsum > q_quarter)[0]
    hfi_aux2 = np.where(q_cumsum > q_three_quarters)[0]
    return hfi_aux2[0] - hfi_aux1[0]

def calc_quarter_flow_interval(q: pd.Series) -> int:
    """
    Calculate the time span between half and three-quarter flow dates.
    
    The interval is defined as the time between:
    - When cumulative discharge reaches 50% of annual total
    - When cumulative discharge reaches 75% of annual total
    """
    q_quarter = 0.5 * np.sum(q)
    q_three_quarters = 0.75 * np.sum(q)
    q_cumsum = np.cumsum(q)
    hfi_aux1 = np.where(q_cumsum > q_quarter)[0]
    hfi_aux2 = np.where(q_cumsum > q_three_quarters)[0]
    return hfi_aux2[0] - hfi_aux1[0]

def calc_q_melt_onset(q: pd.Series) -> int:
    """ From Cayan et al. 2001: Changes in the Onset of Spring
in the Western United States"""
    Qmean = q.mean()
    cum_departure_from_mean = np.cumsum(q - Qmean)
    Qmelt_onset =np.argmin(cum_departure_from_mean)
    return Qmelt_onset

def calc_melt_onset_sharpness(q: pd.Series) -> float:
    """ 	- Snowmelt onset sharpness
	
		Define as the ratio of flow increase in the 2 weeks after snowmelt onset vs. the previous 2 weeks. Error in this “sharpness index” reveals whether the model captures how abruptly SWE is released.
""" 
    Qmelt_onset = calc_q_melt_onset(q)
    Qmelt_onset_2weeks = q.iloc[Qmelt_onset:Qmelt_onset+14]
    Qmelt_onset_2weeks_mean = Qmelt_onset_2weeks.mean()
    Qmelt_onset_2weeks_std = Qmelt_onset_2weeks.std()
    return Qmelt_onset_2weeks_mean / Qmelt_onset_2weeks_std

def calc_high_flow_freq(q: pd.Series, pct: float = 0.9) -> float:
    """
    Calculate the frequency of high flow events.
    
    Args:
        q (pd.Series): Streamflow time series
        pct (float): Threshold percentile for high flows (default: 0.9)
    
    Returns:
        float: Frequency of high flow events (days exceeding threshold / total days)
    """
    q_threshold = pct * q.max()
    return np.sum(q > q_threshold) / len(q)

def calc_peak_distribution(q: pd.Series, slope_range: tuple = (0.1, 0.5), fit_log_space: bool = False) -> float:
    """
    Calculate the peak flow distribution metric.
    
    Args:
        q (pd.Series): Streamflow time series
        slope_range (tuple): Range for slope calculation (default: (0.1, 0.5))
        fit_log_space (bool): Whether to fit in log space (default: False)
    
    Returns:
        float: Peak distribution metric
    """
    peaks, _ = find_peaks(q)
    q_peak = q.iloc[peaks]
    q_peak_sorted = np.sort(q_peak)[-1::-1]
    q90 = np.quantile(q, 0.9)
    q50 = np.quantile(q, 0.5)
    return (q90 - q50) / 0.4

def calc_flashiness_index(q: pd.Series) -> float:
    """
    Calculate the flashiness index using the hydrosignatures package.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        float: Flashiness index
    """
    return fi(q).item()

def calc_q_inflection(q: pd.Series) -> int:
    """
    Calculate the inflection point in the cumulative streamflow.
    
    Uses a 30-day smoothing window and finds the maximum of the second derivative
    of the cumulative flow, which is related to the timing of maximum SWE.
    
    Args:
        q (pd.Series): Streamflow time series
    
    Returns:
        int: Index of the first inflection point
    """
    q_cumsum = np.cumsum(q).squeeze().rolling(window=30).mean()
    first_derivative = np.gradient(q_cumsum)
    second_derivative = np.gradient(first_derivative)
    return np.nanargmax(second_derivative)

def peakfilter_mask(array: np.ndarray, std_multiplier: float = 2.0, 
                   quantile_threshold: float = 0.95, 
                   window_size: int = 4) -> np.ndarray:
    """
    Create a mask for peak filtering of streamflow data.
    
    Args:
        array (np.ndarray): Input array
        std_multiplier (float): Multiplier for standard deviation threshold
        quantile_threshold (float): Quantile threshold for high values
        window_size (int): Number of consecutive points to mark around peaks
    
    Returns:
        np.ndarray: Boolean mask for peak filtering
    """
    # Handle edge cases
    if len(array) == 0:
        return np.array([], dtype=bool)
    if len(array) == 1:
        return np.array([False], dtype=bool)
    
    # Calculate positive differences
    diff = np.diff(array, prepend=array[0])  # More intuitive than prepend=0
    posdif = np.where(diff > 0, diff, 0)
    
    # Skip if no positive differences
    if np.sum(posdif) == 0:
        return array > np.quantile(array, quantile_threshold)
    
    # Calculate threshold based on positive differences only
    posdif_std = np.std(posdif[posdif > 0])  # Only use actual positive values
    if posdif_std == 0:
        return array > np.quantile(array, quantile_threshold)
    
    # Create mask for significant increases
    significant_increase = posdif > (std_multiplier * posdif_std)
    
    # Create windowed mask more efficiently
    mask_increase = np.zeros_like(array, dtype=bool)
    for i in range(window_size):
        mask_increase |= np.roll(significant_increase, -i)
    
    # High value mask
    high_values = array > np.quantile(array, quantile_threshold)
    
    return mask_increase | high_values

def peakfilter(q_array: pd.Series) -> pd.Series:
    """
    Filter peak flows from the streamflow time series.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        pd.Series: Filtered streamflow with peaks removed
    """
    q = q_array.squeeze().copy()
    mask = peakfilter_mask(q)
    filtered = np.where(mask, q, np.nan)
    q[mask] = np.nan
    return q

def peak_flow_efficiency(q_obs, q_sim, melt_months) -> float:
    """ Peak flow efficiency from Griessinger2016
    Peak flows are defined as exceeding 1.5x the mean flow during the melt season"""

    q_obs_filtered = q_obs.loc[q_obs.index.month.isin(melt_months)]
    q_sim_filtered = q_sim.loc[q_sim.index.month.isin(melt_months)]
    mean_flow = q_obs_filtered.mean()
    peak_flow_mask = q_obs_filtered > 1.5 * mean_flow
    peak_flow_obs = q_obs_filtered[peak_flow_mask]
    peak_flow_sim = q_sim_filtered[peak_flow_mask]
    PFE = 1- (np.sum(np.abs(peak_flow_obs - peak_flow_sim)) / np.sum(peak_flow_obs))
    return PFE


def baseflow_filter(q_array: pd.Series) -> pd.Series:
    """
    Extract baseflow from the streamflow time series using the hydrosignatures package.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        pd.Series: Baseflow component
    """
    q = q_array.squeeze().copy()
    bf = baseflow(q)
    if np.all(bf == q):
        print("Warning: Baseflow equals total flow. Try using q_array[:,0]")
    return bf

def calc_q_amplitude(q_array: pd.Series) -> float:
    """
    Calculate the amplitude of the streamflow time series.
    
    Uses a 30-day rolling mean to smooth the data before calculating amplitude.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        float: Flow amplitude (max - min of smoothed series)
    """
    q = q_array.squeeze().copy()
    q_smoothed = q.rolling(window=30).mean()
    return q_smoothed.max() - q_smoothed.min()

def calc_qstart_qend(q_array: pd.Series) -> int:
    """
    Calculate the start of the streamflow season.
    
    Uses a 30-day smoothing window and finds the intersection with the 5th percentile
    of the smoothed flow.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        int: Index of flow start, or np.nan if not found
    """
    q = q_array.squeeze().copy()
    window = 30
    q_smoothed = q.rolling(window=window).mean()
    pct = np.nanpercentile(q_smoothed,50)
    intercepts = np.where(np.diff(np.sign(q_smoothed - pct)))[0]
    intercepts = intercepts[intercepts > window]
    
    if len(intercepts) == 0:
        return [np.nan, np.nan]
    elif len(intercepts) == 2:
        return intercepts
    elif len(intercepts) == 1:
        print("Warning: Only one intercept with Q05")
        return [np.nan,np.nan]
        # return intercepts[0] if q_smoothed[intercepts[0]] > q_smoothed[intercepts[0]-1] else np.nan
    else:  # len(intercepts) > 2
        distances = np.diff(intercepts)
        return intercepts[np.argmax(distances)],intercepts[np.argmax(distances)+1]

def calc_qstart(q_array: pd.Series) -> int:
    """
    Calculate the start of the streamflow season.
    """
    return calc_qstart_qend(q_array)[0]

def calc_qend(q_array: pd.Series) -> int:
    """
    Calculate the end of the streamflow season.
    """
    return calc_qstart_qend(q_array)[1]



def calc_t_qrise(q_array: pd.Series) -> int:
    """
    Calculate the time between flow start and maximum flow.
    
    Args:
        q_array (pd.Series): Streamflow time series
    
    Returns:
        int: Number of days between flow start and maximum
    """
    qstart = calc_qstart(q_array)
    q = q_array.squeeze().copy()
    q_smoothed = q.rolling(window=30).mean()
    t_qmax = np.nanargmax(q_smoothed)
    return t_qmax - qstart

def calc_viney_bias(q_obs, q_sim) -> float:
    """Bias from Viney et al . 2009 """
    B = (q_sim.sum() - q_obs.sum()) / q_obs.sum()
    #BIAS(q,̃ q) = 5|log[1 + B(q,̃ q)] |2.5
    return 5*np.abs(np.log(1 + B))**2.5

def calc_viney_NSE_bias(q_obs, q_sim) -> float:
    """NSE-bias from Viney et al . 2009 """
    BIAS = calc_viney_bias(q_obs, q_sim)
    NSE = he.nse(q_obs, q_sim)
    return NSE - BIAS