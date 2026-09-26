"""Compat shims for brax-on-new-JAX. Import before brax.training.*.

brax 0.14.2 calls jax.device_put_replicated (4 callsites), which jax 0.11
removed. Single-device replacement: put on that one device. This track runs
on one RTX 4060, so replication == placement; anything else fails loud.
"""

import jax
import jax.numpy as jp
import numpy as np


def _device_put_replicated(value, devices=None):
  # Old semantics: replicate across devices AND add a leading device axis
  # (brax's _unpmap later squeezes axis 0). Single-GPU exact equivalent:
  # one leading dim + placement on that device.
  devs = list(devices) if devices is not None else list(jax.local_devices()[:1])
  assert len(devs) == 1, f"single-GPU shim cannot replicate to {len(devs)} devices"

  def _expand(x):
    if isinstance(x, (jax.Array, np.ndarray)):
      return jp.expand_dims(x, axis=0)
    try:
      return jp.expand_dims(jp.asarray(x), axis=0)
    except Exception:
      return x

  return jax.device_put(jax.tree.map(_expand, value), devs[0])


try:
  jax.device_put_replicated
except AttributeError:
  jax.device_put_replicated = _device_put_replicated
