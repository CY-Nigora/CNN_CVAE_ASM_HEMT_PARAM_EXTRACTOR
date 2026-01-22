import numpy as np

# Global RNG (replace with np.random.default_rng(seed) for reproducibility)
_RNG = np.random.default_rng()

# Threshold: consider a range "large" if it spans at least this many decades
_LARGE_DECADES = 3.0  # >=3 decades => large range

# Beta shape parameters for low-end bias (in log space).
# a < 1 biases toward the lower end; b >= 1 suppresses the upper end.
_BETA_A = 0.4
_BETA_B = 2.0

# Mixing ratio for large ranges: 70% global log-uniform + 30% low-end bias
_MIX_P_LOGUNI = 0.7


def _is_large_range(low: float, high: float) -> bool:
    """Decide if [low, high] is a large range by approximate decade span."""
    if low == high:
        return False
    if low > 0:
        span = np.log10(high) - np.log10(low)
    elif high < 0:
        # Fully negative range: compare magnitudes
        span = np.log10(abs(low)) - np.log10(abs(high))
    else:
        # Crosses zero: sum log1p spans on both sides
        span = np.log10(1 + abs(low)) + np.log10(1 + abs(high))
    return span >= _LARGE_DECADES


def _sample_log_uniform_pos(low: float, high: float) -> float:
    """Log-uniform on (low, high) with low > 0."""
    t = _RNG.uniform(np.log10(low), np.log10(high))
    return 10 ** t


def _sample_log_beta_pos(low: float, high: float, a=_BETA_A, b=_BETA_B) -> float:
    """Low-end biased sampling on (low, high) with low > 0 (Beta in log10 space)."""
    u = _RNG.beta(a, b)
    t = np.log10(low) + u * (np.log10(high) - np.log10(low))
    return 10 ** t


def _sample_log1p_uniform_nonneg(low: float, high: float) -> float:
    """Log1p-uniform on [low, high] with 0 <= low < high (naturally denser near 0)."""
    t = _RNG.uniform(np.log1p(low), np.log1p(high))
    return np.expm1(t)


def _sample_log1p_beta_nonneg(low: float, high: float, a=_BETA_A, b=_BETA_B) -> float:
    """Low-end biased sampling on [low, high] using Beta in log1p space (0 <= low)."""
    u = _RNG.beta(a, b)
    t = np.log1p(low) + u * (np.log1p(high) - np.log1p(low))
    return np.expm1(t)


def _sample_negative_only(low: float, high: float, large: bool) -> float:
    """
    Sampling for a fully negative range [low, high] (low < high <= 0).
    Sample magnitude in [|high|, |low|] (positive range) then apply negative sign.
    Low-end bias means smaller magnitude (closer to 0).
    """
    a, b = abs(high), abs(low)  # a <= b
    if a == b:
        return -a
    if large:
        # 70% log-uniform + 30% low-end bias on magnitude
        if _RNG.random() < _MIX_P_LOGUNI:
            m = _sample_log_uniform_pos(a, b)
        else:
            m = _sample_log_beta_pos(a, b)
    else:
        m = _sample_log_uniform_pos(a, b)
    return -m


def _sample_cross_zero(low: float, high: float, large: bool) -> float:
    """
    Sampling for a range crossing zero [low, high] with low < 0 < high.
    Choose side proportional to its log1p span, then sample on that side.
    """
    neg_span = np.log1p(abs(low))
    pos_span = np.log1p(high)
    total = neg_span + pos_span
    if total <= 0:
        return 0.0

    choose_pos = (_RNG.random() < (pos_span / total))
    if choose_pos:
        if large:
            if _RNG.random() < _MIX_P_LOGUNI:
                x = _sample_log1p_uniform_nonneg(0.0, high)
            else:
                x = _sample_log1p_beta_nonneg(0.0, high)
        else:
            x = _sample_log1p_uniform_nonneg(0.0, high)
        return x
    else:
        # Negative side: sample magnitude then apply negative sign
        if large:
            if _RNG.random() < _MIX_P_LOGUNI:
                m = _sample_log1p_uniform_nonneg(0.0, abs(low))
            else:
                m = _sample_log1p_beta_nonneg(0.0, abs(low))
        else:
            m = _sample_log1p_uniform_nonneg(0.0, abs(low))
        return -m


def _sample_value(low: float, high: float) -> float:
    """Sample a single value from [low, high] following the requested strategy."""
    if low == high:
        return float(low)

    large = _is_large_range(low, high)

    if low > 0:
        # Positive-only range
        if large:
            return (_sample_log_uniform_pos(low, high)
                    if _RNG.random() < _MIX_P_LOGUNI
                    else _sample_log_beta_pos(low, high))
        else:
            return _sample_log_uniform_pos(low, high)

    if high < 0:
        # Negative-only range
        return _sample_negative_only(low, high, large)

    # Crosses zero
    return _sample_cross_zero(low, high, large)


def param_random_generator(param_range: dict):
    """
    Generate a random parameter set for the HEMT model.
    Strategy per parameter (decided by its range):
      - Large ranges (>= _LARGE_DECADES decades):
            70% global log-uniform + 30% low-end biased (Beta in log-space).
      - Small ranges:
            global log-uniform.
      - Ranges including 0 or negatives are handled automatically (log1p space
        for nonnegative parts; magnitude sampling for negatives; cross-zero split
        by span).
    Returns stringified values to match the original interface.
    """
    var_dict = {}
    for key, (low, high) in param_range.items():
        v = _sample_value(float(low), float(high))
        var_dict[key] = f"{v:.5e}"  # change to `v` if you prefer floats
    return var_dict
