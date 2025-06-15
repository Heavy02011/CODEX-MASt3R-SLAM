try:
    import mast3r_slam_backends as _impl
except ImportError:  # pragma: no cover - fallback for CPU-only devices
    from . import cpu as _impl

gauss_newton_rays = _impl.gauss_newton_rays
gauss_newton_calib = _impl.gauss_newton_calib
iter_proj = _impl.iter_proj
refine_matches = _impl.refine_matches

__all__ = [
    "gauss_newton_rays",
    "gauss_newton_calib",
    "iter_proj",
    "refine_matches",
]
