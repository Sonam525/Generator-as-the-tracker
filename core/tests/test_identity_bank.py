"""IdentityBank state logic: momentum update when visible, FREEZE under occlusion,
motion extrapolation, drop after max_misses, and query exposing occluded tracks."""
import numpy as np

from gentrack.models.identity_bank import IdentityBank


def _e(*v):
    return np.asarray(v, dtype=np.float32)


def test_new_track_then_momentum_update():
    b = IdentityBank(momentum=0.75)
    b.update(1, {5: (_e(1, 0), (10, 10, 4, 8))})
    assert 5 in b.colors() and len(b) == 1
    # second sighting blends embeddings toward the new one (unit-norm)
    b.update(2, {5: (_e(0, 1), (12, 10, 4, 8))})
    emb = b.tracks[5].embedding
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-5)
    assert emb[0] > emb[1] > 0           # still closer to the original (momentum 0.75)
    assert b.tracks[5].velocity == (2.0, 0.0)  # moved +2 in x over 1 frame


def test_freeze_under_occlusion():
    b = IdentityBank(momentum=0.5, max_misses=50)
    b.update(1, {7: (_e(1, 0), (0, 0, 4, 8))})
    b.update(2, {7: (_e(1, 0), (2, 0, 4, 8))})       # velocity (2,0)
    frozen = b.tracks[7].embedding.copy()
    b.update(3, {})                                   # 7 occluded
    assert b.tracks[7].misses == 1
    assert np.allclose(b.tracks[7].embedding, frozen)  # embedding NOT updated under occlusion
    assert b.predicted_box(7)[0] == 4.0                # box extrapolated by velocity (2->4)


def test_occluded_track_stays_in_query():
    b = IdentityBank()
    b.update(1, {3: (_e(1, 0), (0, 0, 4, 8))})
    b.update(2, {})                                    # occluded but must persist
    colors = {c for _, c in b.query()}
    assert colors == {3}


def test_drop_after_max_misses():
    b = IdentityBank(max_misses=2)
    b.update(1, {9: (_e(1, 0), (0, 0, 4, 8))})
    b.update(2, {})   # miss 1
    b.update(3, {})   # miss 2
    assert 9 in b.colors()
    b.update(4, {})   # miss 3 > max_misses -> dropped
    assert 9 not in b.colors() and len(b) == 0


def test_assign_first_frame_new_tracks():
    b = IdentityBank()
    out = b.assign([(_e(1, 0), (0, 0, 10, 10)), (_e(0, 1), (50, 0, 10, 10))], 1)
    assert None not in out and len(set(out)) == 2 and len(b) == 2


def test_assign_matches_next_frame_by_motion():
    b = IdentityBank()
    c = b.assign([(_e(1, 0), (0, 0, 10, 10))], 1)[0]
    out2 = b.assign([(_e(1, 0), (1, 0, 10, 10))], 2)  # overlapping box -> same track
    assert out2[0] == c and len(b) == 1


def test_assign_reacquires_by_appearance_across_gap():
    b = IdentityBank(max_misses=50)
    c = b.assign([(_e(1, 0), (0, 0, 10, 10))], 1)[0]
    b.assign([], 2)
    b.assign([], 3)
    assert b.tracks[c].misses == 2
    # reappears far away (IoU 0) but SAME appearance -> re-binds by appearance
    out = b.assign([(_e(1, 0), (100, 100, 10, 10))], 4, w_app=0.7, w_motion=0.3, match_thr=0.6)
    assert out[0] == c and b.tracks[c].misses == 0


def test_assign_distinct_appearance_starts_new_track():
    b = IdentityBank(max_misses=50)
    c = b.assign([(_e(1, 0), (0, 0, 10, 10))], 1)[0]
    b.assign([], 2)
    # reappears far away with ORTHOGONAL appearance -> no match -> new track
    out = b.assign([(_e(0, 1), (100, 100, 10, 10))], 3, w_app=0.7, w_motion=0.3, match_thr=0.6)
    assert out[0] != c and len(b) == 2


def test_reacquire_resets_misses():
    b = IdentityBank(max_misses=50)
    b.update(1, {2: (_e(1, 0), (0, 0, 4, 8))})
    b.update(2, {})
    b.update(3, {})
    assert b.tracks[2].misses == 2
    b.update(4, {2: (_e(1, 0), (6, 0, 4, 8))})  # reappears
    assert b.tracks[2].misses == 0 and not b.tracks[2].occluded
