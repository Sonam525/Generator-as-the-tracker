"""Build the project-page videos and posters in docs/media from the generated windows.

    python docs/make_page_media.py <volume_mirror>

<volume_mirror> is a local copy of these volume folders, same relative paths:
  ltx_ds/fullseq/{vid}_ov9/{windows.json,rgb/*.mkv}
  checkpoints/{ltx_arm_a_chain,ltx_arm_b_indep,ltx_arm_c_chain_ckptA}/{vid}/cleaned/wNN/*.png
  dancetrack/val/{vid}/gt/gt.txt
for vid in dancetrack0065 and dancetrack0019.

Display only: the MP4s are lossy H.264 and must never be decoded as ID-maps. Each frame is
taken from the window that owns it, as in video_generator/ltx_arm_eval.py (overlaps split at
their midpoint), and the 512x512 generation is shown at the native 16:9 aspect.
"""
import glob
import json
import math
import os
import sys
from collections import Counter

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
from gentrack.palette import build  # noqa: E402

PAL = build(48)
FPS = 20  # DanceTrack frame rate


def font(size):
    for f in ("C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default(size=size)


class Video:
    def __init__(self, vid):
        self.vid = vid
        meta = json.load(open(f"{SRC}/ltx_ds/fullseq/{vid}_ov9/windows.json"))
        self.wins = sorted(meta["windows"], key=lambda w: w["idx"])
        self.total = meta["total_frames"]
        s = [self.wins[0]["start"]]
        for k in range(1, len(self.wins)):
            s.append((self.wins[k]["start"] + self.wins[k - 1]["end"]) // 2 + 1)
        e = [s[k + 1] - 1 for k in range(len(self.wins) - 1)] + [self.wins[-1]["end"]]
        self.own = list(zip(s, e))
        self._rgb = {}

    def owner(self, gf):
        return next(k for k, (a, b) in enumerate(self.own) if a <= gf <= b)

    def rgb(self, gf):
        k = self.owner(gf)
        if k not in self._rgb:
            self._rgb = {k: [f.to_ndarray(format="rgb24") for f in
                             av.open(f"{SRC}/ltx_ds/fullseq/{self.vid}_ov9/rgb/{self.wins[k]['tag']}.mkv").decode(video=0)]}
        return self._rgb[k][gf - self.wins[k]["start"]]

    def idmap(self, arm, gf):
        k = self.owner(gf)
        files = sorted(glob.glob(f"{SRC}/checkpoints/{arm}/{self.vid}/cleaned/w{k:02d}/*.png"))
        return np.asarray(Image.open(files[gf - self.wins[k]["start"]]).convert("RGB"))

    def gt(self):
        rows = {}
        for ln in open(f"{SRC}/dancetrack/val/{self.vid}/gt/gt.txt"):
            p = ln.split(",")
            rows.setdefault(int(p[1]), {})[int(p[0])] = tuple(map(float, p[2:6]))
        return rows


def panel(a, size, nearest=False, label=None, sub=None):
    im = Image.fromarray(a).resize(size, Image.NEAREST if nearest else Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    if label:
        f = font(max(13, size[1] // 20))
        x0, y0, x1, y1 = d.textbbox((10, 8), label, font=f)
        d.rounded_rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), 5, fill=(0, 0, 0, 170))
        d.text((10, 8), label, font=f, fill=(255, 255, 255))
    if sub:
        f = font(max(11, size[1] // 26))
        x0, y0, x1, y1 = d.textbbox((10, 0), sub, font=f)
        y = size[1] - (y1 - y0) - 14
        d.rounded_rectangle((x0 - 6, y - 4, x1 + 6, y + (y1 - y0) + 6), 5, fill=(0, 0, 0, 150))
        d.text((10, y - y0 + 0), sub, font=f, fill=(230, 230, 230))
    return im


def grid(panels, cols, gap=4):
    w, h = panels[0].size
    rows = math.ceil(len(panels) / cols)
    W, H = cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap
    W, H = W + W % 2, H + H % 2
    out = Image.new("RGB", (W, H), (0, 0, 0))
    for i, p in enumerate(panels):
        out.paste(p, ((i % cols) * (w + gap), (i // cols) * (h + gap)))
    return out


class Writer:
    def __init__(self, name, fps):
        self.path = os.path.join(OUT, name)
        self.c = av.open(self.path, "w", options={"movflags": "+faststart"})
        self.s = None
        self.fps = fps

    def add(self, im):
        if self.s is None:
            self.s = self.c.add_stream("libx264", rate=self.fps)
            self.s.width, self.s.height = im.size
            self.s.pix_fmt = "yuv420p"
            self.s.options = {"crf": "22", "preset": "slow", "tune": "animation"}
        for p in self.s.encode(av.VideoFrame.from_image(im)):
            self.c.mux(p)

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()
        print(f"{os.path.basename(self.path)}: {os.path.getsize(self.path) / 1e6:.1f} MB")


def hero():
    v = Video("dancetrack0065")
    side, stack = Writer("hero_0065.mp4", FPS), Writer("hero_0065_stacked.mp4", FPS)
    for gf in range(1, v.total + 1):
        k = v.owner(gf)
        a = panel(v.rgb(gf), (640, 360), label="Input video", sub=f"frame {gf} / {v.total}")
        b = panel(v.idmap("ltx_arm_a_chain", gf), (640, 360), nearest=True, label="Generated ID-map",
                  sub=f"window {k + 1} of {len(v.wins)}")
        side.add(grid([a, b], 2))
        stack.add(grid([a, b], 1))
        if gf == 351:
            grid([a, b], 2).save(os.path.join(OUT, "hero_0065.jpg"), quality=88)
            grid([a, b], 1).save(os.path.join(OUT, "hero_0065_stacked.jpg"), quality=88)
    side.close()
    stack.close()


def factorial():
    v = Video("dancetrack0065")
    cells = [(None, "Input video"),
             ("ltx_arm_b_indep", "Independent windows"),
             ("ltx_arm_c_chain_ckptA", "Chained, without continuation training"),
             ("ltx_arm_a_chain", "Chained, with continuation training (ours)")]
    w = Writer("factorial_0065.mp4", FPS)
    for gf in range(1, v.total + 1):
        ps = []
        for arm, lab in cells:
            a = v.rgb(gf) if arm is None else v.idmap(arm, gf)
            ps.append(panel(a, (480, 270), nearest=arm is not None, label=lab,
                            sub=f"frame {gf}" if arm is None else None))
        w.add(grid(ps, 2))
        if gf == 351:
            grid(ps, 2).save(os.path.join(OUT, "factorial_0065.jpg"), quality=88)
    w.close()


def reacq(tid=6, around=504, pad=40, fps=10):
    """One re-acquisition event of the paper's Figure (dancetrack0019, track 6)."""
    v = Video("dancetrack0019")
    track = v.gt()[tid]
    a = max(f for f in track if f < around) + 1          # first hidden frame
    b = min(f for f in track if f > around)               # reappearance
    W0, H0 = (1280, 720)

    def colour_in(gf):  # majority palette colour inside the GT box at 512^2
        x, y, w, h = track[gf]
        m = v.idmap("ltx_arm_a_chain", gf)
        sx, sy = 512 / W0, 512 / H0
        crop = m[int(y * sy):int((y + h) * sy), int(x * sx):int((x + w) * sx)].reshape(-1, 3)
        idx = np.argmin(((crop[:, None, :].astype(int) - PAL[None].astype(int)) ** 2).sum(-1), 1)
        c = Counter(int(i) for i in idx if i != 0)
        return c.most_common(1)[0][0] if c else 0

    before = Counter(colour_in(f) for f in range(a - 10, a)).most_common(1)[0][0]
    after = Counter(colour_in(f) for f in range(b, b + 10)).most_common(1)[0][0]
    print(f"reacq: hidden frames {a}-{b - 1} ({b - a}), colour before {before}, after {after}",
          "-> SAME" if before == after and before else "-> DIFFERENT")

    def centre(gf):
        if gf in track:
            x, y, w, h = track[gf]
        else:
            t = (gf - (a - 1)) / (b - (a - 1))
            p, q = track[a - 1], track[b]
            x, y, w, h = [p[i] + t * (q[i] - p[i]) for i in range(4)]
        return (x + w / 2) / W0, (y + h / 2) / H0, h / H0

    def ring(im, gf, colour):
        cx, cy, hh = centre(gf)
        W, H = im.size
        r = max(28, hh * H * 0.62)
        d = ImageDraw.Draw(im)
        box = (cx * W - r * 0.6, cy * H - r, cx * W + r * 0.6, cy * H + r)
        if gf in track:
            d.ellipse(box, outline=colour, width=3)
        else:  # dashed while hidden
            for s in range(0, 360, 24):
                d.arc(box, s, s + 12, fill=colour, width=3)

    side, stack = Writer("reacq_0019.mp4", fps), Writer("reacq_0019_stacked.mp4", fps)
    for gf in range(a - pad, b + pad):
        state = "hidden" if a <= gf < b else ("before the gap" if gf < a else "after the gap")
        p = panel(v.rgb(gf), (640, 360), label="Input video", sub=f"frame {gf} · {state}")
        q = panel(v.idmap("ltx_arm_a_chain", gf), (640, 360), nearest=True, label="Generated ID-map")
        ring(p, gf, (255, 214, 0))
        ring(q, gf, (255, 255, 255))
        side.add(grid([p, q], 2))
        stack.add(grid([p, q], 1))
        if gf == a + (b - a) // 2:
            grid([p, q], 2).save(os.path.join(OUT, "reacq_0019.jpg"), quality=88)
            grid([p, q], 1).save(os.path.join(OUT, "reacq_0019_stacked.jpg"), quality=88)
    side.close()
    stack.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[2:] or ["reacq", "hero", "factorial"]
    for name in which:
        globals()[name]()
