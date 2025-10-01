import os
import io
import json
import hashlib
import shutil
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

# Keras 3 uses `keras` top-level; works with TF-Keras as well
try:
    import keras
except ImportError:
    # fallback for older TF-Keras imports
    from tensorflow import keras  # type: ignore

import tensorflow as tf  # tf is still needed for SavedModel/TFLite bits


def _sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_text(path: Union[str, Path], text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def _write_json(path: Union[str, Path], obj: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _pip_freeze() -> str:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        return out.strip()
    except Exception as e:
        return f"pip freeze failed: {e!r}"


def _safe_mkdir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _infer_dummy_input(model: "keras.Model") -> Optional[np.ndarray]:
    """
    Try to infer a dummy input batch for sanity checks. Returns None if shape cannot be inferred.
    """
    try:
        # Keras 3: model.inputs could be empty for subclassed models until built.
        if getattr(model, "inputs", None):
            # Choose the first input; build a small batch
            shape = model.inputs[0].shape  # TensorShape
            # Convert to concrete list with None -> 1
            dims = [d if d is not None else 1 for d in shape.as_list()]  # type: ignore
            if len(dims) == 0:
                return None
            if dims[0] == 0:
                dims[0] = 1
            x = np.random.default_rng(1234).standard_normal(dims).astype(np.float32)
            return x
        # If not built, try to build with a common input shape found in model config
        cfg = model.get_config() if hasattr(model, "get_config") else None
        if isinstance(cfg, dict):
            # Try to find batch_input_shape
            for lyr in cfg.get("layers", []):
                bi = lyr.get("config", {}).get("batch_input_shape")
                if bi:
                    dims = [d if (d is not None and d != 0) else 1 for d in bi]
                    x = np.random.default_rng(1234).standard_normal(dims).astype(np.float32)
                    return x
    except Exception:
        pass
    return None


def archive_keras_model(
    model_path: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    custom_objects: Optional[Dict[str, Any]] = None,
    sample_input: Optional[np.ndarray] = None,
    include_onnx: bool = False,
    include_tflite: bool = False,
) -> Path:
    """
    Load a .keras model and archive it into multiple formats + metadata for long-term reuse.

    Parameters
    ----------
    model_path : str | Path
        Path to an existing `.keras` file (Keras v3 native format) or a Keras-compatible model file.
    out_dir : str | Path | None
        Output directory (created if missing). If None, a timestamped folder is created next to the model.
    custom_objects : dict | None
        Mapping for custom layers/ops, e.g., {'MyLayer': MyLayer}. Needed to load certain models.
    sample_input : np.ndarray | None
        Optional concrete input batch for inference parity checks. If None, a dummy input is attempted.
    include_onnx : bool
        If True, try exporting ONNX (requires `tf2onnx` installed).
    include_tflite : bool
        If True, try exporting TFLite (best-effort; may not support all ops).

    Returns
    -------
    Path
        The path to the archive output directory.
    """
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if out_dir is None:
        out_dir = model_path.with_suffix("").parent / f"{model_path.stem}_archive_{timestamp}"
    out_dir = _safe_mkdir(out_dir)

    logs_dir = _safe_mkdir(Path(out_dir) / "logs")
    meta_dir = _safe_mkdir(Path(out_dir) / "meta")
    files_dir = _safe_mkdir(Path(out_dir) / "files")

    status = {
        "model_input": str(model_path),
        "created_utc": timestamp,
        "python_version": sys.version,
        "tensorflow_version": getattr(tf, "__version__", "unknown"),
        "keras_version": getattr(keras, "__version__", "unknown"),
        "exports": {},
        "notes": [],
    }

    # Keep a pristine copy of the source
    try:
        src_copy = files_dir / model_path.name
        if src_copy.resolve() != model_path:
            shutil.copy2(str(model_path), str(src_copy))
        status["exports"]["original_copy"] = {"path": str(src_copy), "sha256": _sha256(src_copy)}
    except Exception as e:
        status["notes"].append(f"Copy original failed: {e!r}")

    # Load the model
    try:
        model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
        status["loaded"] = True
    except Exception as e:
        status["loaded"] = False
        status["error"] = f"Failed to load model: {e!r}"
        _write_json(Path(out_dir) / "status.json", status)
        raise

    # Save model summary
    try:
        s = io.StringIO()
        model.summary(print_fn=lambda x: s.write(x + "\n"))
        _write_text(logs_dir / "model_summary.txt", s.getvalue())
    except Exception as e:
        status["notes"].append(f"Model summary failed: {e!r}")

    # Determine an input batch for sanity checks
    x_test = sample_input if sample_input is not None else _infer_dummy_input(model)
    baseline_out = None
    if x_test is not None:
        try:
            baseline_out = model.predict(x_test, verbose=0)
            status["sanity_check"] = {"baseline_inference": "ok"}
        except Exception as e:
            status["sanity_check"] = {"baseline_inference": f"failed: {e!r}"}
    else:
        status["sanity_check"] = {"baseline_inference": "skipped (no input shape)"}

    # 1) Re-save as `.keras` (fresh re-pack)
    try:
        keras_path = files_dir / "model_repacked.keras"
        model.save(str(keras_path))
        status["exports"]["keras"] = {"path": str(keras_path), "sha256": _sha256(keras_path)}
    except Exception as e:
        status["exports"]["keras"] = {"error": f"{e!r}"}

    # 2) TensorFlow SavedModel (inference graph)
    try:
        # In Keras 3, model.export() writes an inference-only SavedModel with signatures.
        savedmodel_dir = files_dir / "savedmodel"
        if hasattr(model, "export"):
            model.export(str(savedmodel_dir))
        else:
            # Fallback for older TF-Keras
            tf.saved_model.save(model, str(savedmodel_dir))
        status["exports"]["savedmodel"] = {"path": str(savedmodel_dir)}
    except Exception as e:
        status["exports"]["savedmodel"] = {"error": f"{e!r}"}

    # 3) HDF5 Keras file
    try:
        h5_path = files_dir / "model.h5"
        model.save(str(h5_path))  # saving with .h5 uses HDF5 format when supported
        status["exports"]["h5"] = {"path": str(h5_path), "sha256": _sha256(h5_path)}
    except Exception as e:
        status["exports"]["h5"] = {"error": f"{e!r}"}

    # 4) JSON architecture + 5) Weights (HDF5) + 6) Weights (NumPy NPZ)
    try:
        json_cfg = model.to_json()
        json_path = files_dir / "model_config.json"
        _write_text(json_path, json_cfg)
        status["exports"]["json_config"] = {"path": str(json_path), "sha256": _sha256(json_path)}
    except Exception as e:
        status["exports"]["json_config"] = {"error": f"{e!r}"}

    try:
        weights_h5 = files_dir / "weights.h5"
        model.save_weights(str(weights_h5))
        status["exports"]["weights_h5"] = {"path": str(weights_h5), "sha256": _sha256(weights_h5)}
    except Exception as e:
        status["exports"]["weights_h5"] = {"error": f"{e!r}"}

    try:
        weights = {w.name.replace(":", "_"): w.numpy() for w in model.weights}
        weights_npz = files_dir / "weights_npz.npz"
        np.savez_compressed(weights_npz, **weights)
        status["exports"]["weights_npz"] = {"path": str(weights_npz), "sha256": _sha256(weights_npz)}
    except Exception as e:
        status["exports"]["weights_npz"] = {"error": f"{e!r}"}

    # 7) Optional: ONNX export (requires tf2onnx)
    if include_onnx:
        try:
            import tf2onnx  # noqa: F401
            onnx_path = files_dir / "model.onnx"
            # Convert via from_keras
            # Newer tf2onnx supports keras model object directly:
            import tf2onnx.convert as convert

            spec = None
            if x_test is not None:
                # Build concrete input spec from sample
                spec = [tf.TensorSpec(x_test.shape, tf.as_dtype(x_test.dtype), name="input_0")]
            model_proto, _ = convert.from_keras(model, input_signature=spec, output_path=str(onnx_path))
            status["exports"]["onnx"] = {"path": str(onnx_path), "sha256": _sha256(onnx_path)}
        except Exception as e:
            status["exports"]["onnx"] = {"error": f"{e!r}"}

    # 8) Optional: TFLite export (best-effort)
    if include_tflite:
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            tflite_path = files_dir / "model.tflite"
            tflite_path.write_bytes(tflite_model)
            status["exports"]["tflite"] = {"path": str(tflite_path), "sha256": _sha256(tflite_path)}
        except Exception as e:
            status["exports"]["tflite"] = {"error": f"{e!r}"}

    # Environment + metadata
    try:
        env_txt = _pip_freeze()
        _write_text(meta_dir / "environment_freeze.txt", env_txt)
    except Exception as e:
        status["notes"].append(f"pip freeze failed: {e!r}")

    # Basic README
    readme = f"""# Model Archive

Created: {timestamp} (UTC)

## Source
- Original: {model_path}

## Formats Included
- Keras: model_repacked.keras
- TensorFlow SavedModel: ./savedmodel/
- HDF5: model.h5
- JSON config: model_config.json
- Weights (HDF5): weights.h5
- Weights (NumPy .npz): weights_npz.npz
- Optional: ONNX (if requested): model.onnx
- Optional: TFLite (if requested): model.tflite

## Reproducibility
- Python: {sys.version.split()[0]}
- TensorFlow: {getattr(tf, "__version__", "unknown")}
- Keras: {getattr(keras, "__version__", "unknown")}
- Full environment: ./meta/environment_freeze.txt
- Checksums: see status.json

## Notes
- Prefer SavedModel for serving longevity; .keras is the canonical Keras v3 format with full fidelity.
- JSON+weights provide a framework-light fallback if high-level loaders break in the future.
- If you have custom layers, keep their source code and version pinned alongside this archive.

"""
    _write_text(Path(out_dir) / "README.txt", readme)

    # Light sanity: compare predictions across some exports (when possible)
    sanity_report = {}
    if x_test is not None and baseline_out is not None:
        def _compare(name: str, pred: Any) -> str:
            try:
                # Convert to numpy for comparison
                a = np.array(baseline_out)
                b = np.array(pred)
                if a.shape != b.shape:
                    return f"shape_mismatch: {a.shape} vs {b.shape}"
                diff = np.max(np.abs(a - b))
                return f"max_abs_diff={float(diff):.6g}"
            except Exception as e_:
                return f"compare_failed: {e_!r}"

        # H5 round-trip
        try:
            m_h5 = keras.models.load_model(str(files_dir / "model.h5"), custom_objects=custom_objects)
            p_h5 = m_h5.predict(x_test, verbose=0)
            sanity_report["h5_vs_baseline"] = _compare("h5", p_h5)
        except Exception as e:
            sanity_report["h5_vs_baseline"] = f"load_or_pred_failed: {e!r}"

        # Keras repack round-trip
        try:
            m_k = keras.models.load_model(str(files_dir / "model_repacked.keras"), custom_objects=custom_objects)
            p_k = m_k.predict(x_test, verbose=0)
            sanity_report["keras_vs_baseline"] = _compare("keras", p_k)
        except Exception as e:
            sanity_report["keras_vs_baseline"] = f"load_or_pred_failed: {e!r}"

    status["sanity_report"] = sanity_report

    # Final status dump
    _write_json(Path(out_dir) / "status.json", status)

    return Path(out_dir)


# Example usage:
# archive_dir = archive_keras_model("path/to/model.keras",
#                                   include_onnx=True,
#                                   include_tflite=False)
# print("Archived at:", archive_dir)