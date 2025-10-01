from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Sequence
import numpy as np
import tensorflow as tf

# Keras 3 uses `keras` top-level; fallback for older tf.keras
try:
    import keras
except ImportError:
    from tensorflow import keras  # type: ignore


def keras_to_savedmodel(model_path: str, custom_objects=None) -> str:
    """
    Load a .keras model and save a TensorFlow SavedModel right next to it
    (using the same base name). Returns the SavedModel directory path.
    """
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"No such file: {model_path}")

    model = keras.models.load_model(
        str(model_path), custom_objects=custom_objects)

    out_dir = model_path.with_suffix("")  # strip .keras
    savedmodel_dir = str(out_dir)

    # Prefer Keras 3 export() if available; otherwise fallback
    if hasattr(model, "export"):
        model.export(savedmodel_dir)
    else:
        tf.saved_model.save(model, savedmodel_dir)

    return savedmodel_dir


def _infer_input_batch(model: "keras.Model") -> Tuple[Sequence[Tuple[int, ...]], Sequence[tf.DType]]:
    """
    Try to infer one or more input shapes and dtypes from the model.
    Returns (shapes, dtypes). Unknown/None dims become 1.
    """
    shapes = []
    dtypes = []
    if getattr(model, "inputs", None):
        for tensor in model.inputs:
            shape = [d if (d is not None and d != 0)
                     else 1 for d in tensor.shape.as_list()]
            shapes.append(tuple(shape))
            # Try to preserve dtype; fallback to float32
            try:
                dtypes.append(tensor.dtype)
            except Exception:
                dtypes.append(tf.float32)
        return shapes, dtypes

    # Fallback for some subclassed models: try to read from config
    try:
        cfg = model.get_config()
        if isinstance(cfg, dict):
            # crude scan for batch_input_shape
            for lyr in cfg.get("layers", []):
                bi = lyr.get("config", {}).get("batch_input_shape")
                if bi:
                    shape = [d if (d is not None and d != 0)
                             else 1 for d in bi]
                    shapes.append(tuple(shape))
                    dtypes.append(tf.float32)
                    break
    except Exception:
        pass

    if not shapes:
        raise ValueError(
            "Could not infer input shape(s); please pass sample inputs explicitly.")
    return shapes, dtypes


def _make_random_inputs(
    shapes: Sequence[Tuple[int, ...]],
    dtypes: Sequence[tf.DType],
    rng: np.random.Generator,
) -> Sequence[np.ndarray]:
    xs = []
    for shape, dt in zip(shapes, dtypes):
        np_dtype = tf.as_dtype(
            dt).as_numpy_dtype if dt is not None else np.float32
        # Use standard normal for floats, integers default to zeros
        if np.issubdtype(np_dtype, np.floating):
            x = rng.standard_normal(shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer):
            x = rng.integers(low=0, high=10, size=shape, dtype=np_dtype)
        else:
            # Fallback to float32
            x = rng.standard_normal(shape).astype(np.float32)
        xs.append(x)
    return xs


def _to_keras_inputs(xs: Sequence[np.ndarray]) -> Union[np.ndarray, list]:
    return xs[0] if len(xs) == 1 else list(xs)


def _flatten_to_numpy(y: Any) -> Tuple[np.ndarray, ...]:
    """
    Convert arbitrary Keras/TensorFlow outputs (tensor, list/tuple/dict of tensors)
    into a deterministic tuple of numpy arrays for comparison.
    """
    def to_np(t):
        if isinstance(t, (np.ndarray,)):
            return t
        try:
            return t.numpy()
        except Exception:
            return np.array(t)

    if isinstance(y, dict):
        return tuple(to_np(y[k]) for k in sorted(y.keys()))
    if isinstance(y, (list, tuple)):
        return tuple(to_np(t) for t in y)
    return (to_np(y),)


def _compare_outputs(a: Tuple[np.ndarray, ...], b: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    if len(a) != len(b):
        return {"ok": False, "reason": f"num_outputs_mismatch: {len(a)} vs {len(b)}"}
    max_abs = 0.0
    max_rel = 0.0
    for i, (ai, bi) in enumerate(zip(a, b)):
        if ai.shape != bi.shape:
            return {"ok": False, "reason": f"shape_mismatch@{i}: {ai.shape} vs {bi.shape}"}
        ai = np.asarray(ai)
        bi = np.asarray(bi)
        diff = np.abs(ai - bi)
        denom = np.maximum(1e-12, np.maximum(np.abs(ai), np.abs(bi)))
        max_abs = max(max_abs, float(diff.max()))
        max_rel = max(max_rel, float((diff / denom).max()))
    return {"ok": True, "max_abs_diff": max_abs, "max_rel_diff": max_rel}


def _load_savedmodel_for_inference(savedmodel_dir: str) -> Tuple[Optional["keras.Model"], Optional[Any]]:
    """
    Try loading the SavedModel as a Keras model. If that fails, fall back to
    tf.saved_model.load returning a callable signature.
    Returns (keras_model_or_None, serving_fn_or_None).
    """
    # Try keras loader first
    try:
        km = keras.models.load_model(savedmodel_dir)
        return km, None
    except Exception:
        pass

    # Fall back to TF SavedModel callable
    obj = tf.saved_model.load(savedmodel_dir)
    # serving_default is the common signature name
    sig = getattr(obj, "signatures", {})
    if "serving_default" in sig:
        return None, sig["serving_default"]
    # If no signature, try direct call (rare)
    if hasattr(obj, "__call__"):
        return None, obj.__call__
    raise RuntimeError(
        "SavedModel loaded but no usable serving function was found.")


def test_keras_to_savedmodel_parity(
    model_path: str,
    *,
    custom_objects: Optional[Dict[str, Any]] = None,
    sample_inputs: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    trials: int = 3,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    seed: int = 1234,
    warmup: bool = True,
) -> Dict[str, Any]:
    """
    Export a SavedModel next to a .keras file, reload it, and verify inference parity.

    Parameters
    ----------
    model_path : str
        Path to .keras file.
    custom_objects : dict, optional
        Custom layer mappings needed to load the Keras model.
    sample_inputs : np.ndarray | list[np.ndarray], optional
        Concrete input(s) to test with. If None, shapes are inferred and random
        inputs are generated for each trial.
    trials : int
        Number of random trials (ignored if sample_inputs is provided).
    atol, rtol : float
        Absolute/relative tolerances for comparison.
    seed : int
        RNG seed for reproducible random inputs.
    warmup : bool
        If True, do a warmup inference before measuring parity (helps JIT/stable kernels).

    Returns
    -------
    dict
        A report with export path, per-trial results, and overall pass/fail.
    """
    model_path = str(Path(model_path).expanduser().resolve())
    # Load baseline Keras model
    base_model = keras.models.load_model(
        model_path, custom_objects=custom_objects)

    # Prepare inputs
    rng = np.random.default_rng(seed)
    if sample_inputs is None:
        shapes, dtypes = _infer_input_batch(base_model)
        # build N trials of randomized inputs

        def gen():
            for _ in range(max(1, trials)):
                xs = _make_random_inputs(shapes, dtypes, rng)
                yield _to_keras_inputs(xs)
        inputs_iter = list(gen())
    else:
        if isinstance(sample_inputs, np.ndarray):
            inputs_iter = [sample_inputs]
        else:
            # assume list/sequence for multi-input models
            inputs_iter = [sample_inputs]

    # Optional warmup on baseline model
    if warmup:
        _ = base_model.predict(inputs_iter[0], verbose=0)

    # Export SavedModel
    savedmodel_dir = keras_to_savedmodel(
        model_path, custom_objects=custom_objects)

    # Load SavedModel for inference
    km, serving_fn = _load_savedmodel_for_inference(savedmodel_dir)

    # Inference helper for SavedModel
    def sm_predict(x):
        if km is not None:
            return km.predict(x, verbose=0)
        # serving_default expects dict/named tensors or raw tensor if single input
        if isinstance(x, (list, tuple)):
            # Heuristic: map to positional names input_0, input_1, ...
            feed = {f"input_{i}": tf.convert_to_tensor(
                xi) for i, xi in enumerate(x)}
            y = serving_fn(**feed)
        elif isinstance(x, dict):
            y = serving_fn(**{k: tf.convert_to_tensor(v)
                           for k, v in x.items()})
        else:
            # single input
            y = serving_fn(tf.convert_to_tensor(x))
        # Convert StructuredOutputs to numpy-like
        return {k: v for k, v in y.items()} if hasattr(y, "items") else y

    # Compare per trial
    results = []
    overall_ok = True
    for i, x in enumerate(inputs_iter):
        base_out = base_model.predict(x, verbose=0)
        sm_out = sm_predict(x)

        base_flat = _flatten_to_numpy(base_out)
        sm_flat = _flatten_to_numpy(sm_out)

        cmp_res = _compare_outputs(base_flat, sm_flat)
        # tolerance check
        if cmp_res.get("ok"):
            ok = (cmp_res["max_abs_diff"] <= atol) or (
                cmp_res["max_rel_diff"] <= rtol)
            if not ok:
                cmp_res["ok"] = False
                cmp_res["reason"] = (
                    f"tolerance_exceeded: max_abs={cmp_res['max_abs_diff']:.3g} "
                    f"(atol={atol}), max_rel={cmp_res['max_rel_diff']:.3g} (rtol={rtol})"
                )
        results.append({"trial": i, **cmp_res})
        overall_ok = overall_ok and results[-1]["ok"]

    report = {
        "model_path": model_path,
        "savedmodel_dir": savedmodel_dir,
        "trials": len(results),
        "atol": atol,
        "rtol": rtol,
        "results": results,
        "passed": overall_ok,
        "loader_used": "keras" if km is not None else "tf.saved_model.load",
    }
    return report


# Example:
# rep = test_keras_to_savedmodel_parity(
#     "my_model.keras",
#     trials=5,
#     atol=1e-5,
#     rtol=1e-4,
#     # sample_inputs=np.random.randn(1, 224, 224, 3).astype("float32"),  # optional
# )
# print(rep)
