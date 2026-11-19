"""LTX adapter: pure layout helpers + construction guards (all CPU, no GPU/torch)."""
import numpy as np
import pytest

from gentrack.models.ltx import (
    LTXVideoAdapter,
    latent_frames,
    temporal_pack_split,
    valid_ltx_length,
)


def test_valid_ltx_length():
    assert valid_ltx_length(1) and valid_ltx_length(9) and valid_ltx_length(49)
    assert valid_ltx_length(121)
    assert not valid_ltx_length(0)
    assert not valid_ltx_length(8)
    assert not valid_ltx_length(50)


def test_latent_frames_causal_vae():
    assert latent_frames(1) == 1
    assert latent_frames(9) == 2
    assert latent_frames(49) == 7
    assert latent_frames(121) == 16
    assert latent_frames(49, temporal_ratio=4) == 13


def test_temporal_pack_split_disjoint_contiguous():
    cond, target = temporal_pack_split(7)
    assert cond == list(range(7))
    assert target == list(range(7, 14))
    assert set(cond).isdisjoint(target)


def test_adapter_rejects_invalid_config():
    with pytest.raises(ValueError):
        LTXVideoAdapter("x", frames=50)            # not 8k+1
    with pytest.raises(ValueError):
        LTXVideoAdapter("x", frames=49, size=500)  # not % 32
    with pytest.raises(ValueError):
        LTXVideoAdapter("x", frames=49, cond_mode="bogus")


def test_video_mode_methods_raise_without_gpu():
    a = LTXVideoAdapter("x", frames=49, size=512)   # construction must not import torch
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError):
        a.perceive(frame)
    with pytest.raises(NotImplementedError):
        a.edit(frame, frame)
    with pytest.raises(NotImplementedError):
        a.generate_clip([frame])
    with pytest.raises(NotImplementedError):
        a.temporal([frame])
