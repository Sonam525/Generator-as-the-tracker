/* The 47 identity colours of core/gentrack/palette.py (K = 48, index 0 = black background). */
const PALETTE = "00ff00,0000ff,ff0000,00ffff,ff92db,ffdb6d,006d00,24006d,2492db,924924,ff24ff,92ff92,ff006d,ffffff,dbff00,006dff,006d6d,6d496d,ff9200,b60092,ff9292,92926d,92b624,00b600,9249ff,494900,920049,b692ff,49b692,b60024,92dbff,490000,dbb6db,dbffb6,244992,db9249,ffdb00,ff6d49,002400,0000b6,490049,00ff6d,db6dff,6d49b6,002449,494949,92ff49".split(",");

function drawPalette() {
  const box = document.getElementById("palette");
  if (!box) return;
  box.appendChild(Object.assign(document.createElement("span"), { title: "0: background" }));
  PALETTE.forEach((hex, i) => {
    const s = document.createElement("span");
    s.style.background = "#" + hex;
    s.title = `${i + 1}: #${hex}`;
    box.appendChild(s);
  });
}

/* On a phone the side-by-side panels are too small, so use the stacked clips. */
const STACKED = {
  hero: { src: "media/hero_0065_stacked.mp4", poster: "media/hero_0065_stacked.jpg", w: 640, h: 724 },
  "reacq-video": { src: "media/reacq_0019_stacked.mp4", poster: "media/reacq_0019_stacked.jpg", w: 640, h: 724 },
};
function useStackedClips() {
  if (!window.matchMedia("(max-width: 640px)").matches) return;
  for (const [id, alt] of Object.entries(STACKED)) {
    const v = document.getElementById(id);
    if (!v) continue;
    v.querySelector("source").src = alt.src;
    v.poster = alt.poster;
    v.width = alt.w; v.height = alt.h;
    v.load();
  }
}

drawPalette();
useStackedClips();
