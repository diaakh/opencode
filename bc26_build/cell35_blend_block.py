# ---- soundscape-adapted student rank-blend (config-driven, low weight) ----
# Blend each loaded member into the base scores in PER-CLASS PERCENTILE-RANK
# space (matching the base's xSED rank-blend convention).  Each member enters
# at its configured weight (renormalized so base keeps weight (1 - sum_w)).
# No-op if no member ran (commit/dry-run scale-test or all artifacts missing).
_m_members_blended = []
if ("_m_member_probs" in dir()) and len(_m_member_probs) > 0:
    _m_base_arr = _v237_pre[_v237_cols].to_numpy("float32")
    _m_rank_base = _v237_pd.DataFrame(_m_base_arr).rank(axis=0, pct=True).to_numpy("float32")
    # collect valid members (align row count, sane weight)
    _m_use = []
    _m_wsum = 0.0
    for _name, (_probs, _w) in _m_member_probs.items():
        if _w <= 0.0:
            print(f"[members] blend skip {_name}: weight<=0")
            continue
        if _probs.shape[0] != _m_base_arr.shape[0]:
            print(f"[members] aligning {_name} rows {_probs.shape[0]} -> base {_m_base_arr.shape[0]}")
            _probs = _probs[: _m_base_arr.shape[0]]
        assert _probs.shape == _m_base_arr.shape, (
            f"member {_name} shape {_probs.shape} != base {_m_base_arr.shape}")
        _m_use.append((_name, _probs, _w))
        _m_wsum += _w
    if _m_use and _m_wsum > 0.0:
        # base keeps (1 - sum_w); guard against over-weighting.
        if _m_wsum >= 1.0:
            print(f"[members] WARNING: member weight sum {_m_wsum:.3f} >= 1; clamping to 0.5 total")
            _scale = 0.5 / _m_wsum
            _m_use = [(n, p, w * _scale) for (n, p, w) in _m_use]
            _m_wsum = 0.5
        _m_blended = (1.0 - _m_wsum) * _m_rank_base
        for _name, _probs, _w in _m_use:
            _rank_m = _v237_pd.DataFrame(_probs).rank(axis=0, pct=True).to_numpy("float32")
            _m_blended = _m_blended + _w * _rank_m
            _corr = float(_m_np.corrcoef(_m_rank_base.ravel(), _rank_m.ravel())[0, 1])
            print(f"[members] blend {_name}: w={_w:.3f} corr(base,member)={_corr:.4f} "
                  f"(lower corr = more orthogonal)")
            _m_members_blended.append(_name)
        _m_diff = float(_m_np.abs(_m_blended - _m_rank_base).mean())
        print(f"[members] blended {_m_members_blended} (sum_w={_m_wsum:.3f}, "
              f"base_w={1.0 - _m_wsum:.3f})  mean|Δrank|={_m_diff:.5f}")
        _v237_pre[_v237_cols] = _m_np.clip(_m_blended, 0.0, 1.0).astype("float32")
    else:
        print("[members] rank-blend skipped (no valid weighted members); base unchanged.")
else:
    print("[members] rank-blend skipped (no member probs); base unchanged.")
# ---------------------------------------------------------------------------
