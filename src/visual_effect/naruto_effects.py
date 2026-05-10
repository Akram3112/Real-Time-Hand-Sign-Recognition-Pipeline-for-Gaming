"""
Naruto Jutsu Effects Preview
----------------------------
Requirements: pip install opencv-python numpy

Controls:
  F  →  Fire Jutsu        (Katon · Gōkakyū)
  L  →  Lightning         (Raiton · Chidori)
  W  →  Water             (Suiton · Suiryūdan)
  S  →  Shadow Clone x2   (Kage Bunshin no Jutsu)
  3  →  Shadow Clone x3
  R  →  Reset / clear all effects
  Q  →  Quit
"""

import cv2
import numpy as np
import math
import random
from collections import deque

# ─────────────────────────────────────────────
#  FIRE EFFECT
# ─────────────────────────────────────────────
class FireEffect:
    def __init__(self):
        self.particles = []
        self.active = False
        self.frame_count = 0

    def trigger(self, x, y):
        self.active = True
        self.frame_count = 0
        self._spawn_burst(x, y, intensity=3.0)

    def _spawn_burst(self, x, y, intensity=1.0):
        for _ in range(int(60 * intensity)):
            angle = random.uniform(-math.pi / 3, math.pi / 3) - math.pi / 2
            speed = random.uniform(2, 9)
            self.particles.append({
                'x': x + random.gauss(0, 20), 'y': y,
                'vx': math.cos(angle) * speed + random.gauss(0, 0.6),
                'vy': math.sin(angle) * speed,
                'life': random.randint(25, 60), 'max_life': 50,
                'r': random.randint(6, 22),
            })

    def _spawn_continuous(self, x, y):
        for _ in range(15):
            angle = random.uniform(-math.pi / 4, math.pi / 4) - math.pi / 2
            speed = random.uniform(1.5, 6)
            self.particles.append({
                'x': x + random.gauss(0, 15), 'y': y,
                'vx': math.cos(angle) * speed + random.gauss(0, 0.4),
                'vy': math.sin(angle) * speed,
                'life': random.randint(15, 40), 'max_life': 35,
                'r': random.randint(4, 14),
            })

    def draw(self, frame, cx, cy):
        if not self.particles and not self.active:
            return frame
        self.frame_count += 1
        if self.active and self.frame_count < 60:
            self._spawn_continuous(cx, cy)
        elif self.frame_count >= 60:
            self.active = False

        h, w = frame.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        alive = []
        for p in self.particles:
            t = p['life'] / p['max_life']
            cx_p, cy_p = int(p['x']), int(p['y'])
            if 0 <= cx_p < w and 0 <= cy_p < h:
                r, g, b = 255, int(220 * t), int(60 * t * t)
                rad = max(1, int(p['r'] * t))
                cv2.circle(overlay, (cx_p, cy_p), rad, (b, g, r), -1)
                cv2.circle(overlay, (cx_p, cy_p), max(1, rad // 3), (200, 240, 255), -1)
            p['x'] += p['vx'] + random.gauss(0, 0.5)
            p['y'] += p['vy']
            p['vy'] -= 0.18
            p['vx'] *= 0.98
            p['life'] -= 1
            if p['life'] > 0:
                alive.append(p)
        self.particles = alive
        glow = cv2.GaussianBlur(overlay, (25, 25), 0)
        return np.clip(frame.astype(np.float32) + glow * 1.6, 0, 255).astype(np.uint8)

    def reset(self):
        self.particles = []
        self.active = False
        self.frame_count = 0


# ─────────────────────────────────────────────
#  LIGHTNING EFFECT
# ─────────────────────────────────────────────
class LightningEffect:
    def __init__(self):
        self.active = False
        self.frame_count = 0
        self.bolts = []
        self.duration = 80

    def trigger(self, x, y):
        self.active = True
        self.frame_count = 0
        self.origin = (x, y)

    def _make_bolt(self, x1, y1, x2, y2, depth=5):
        points = [(x1, y1)]
        self._subdivide(points, x1, y1, x2, y2, depth)
        points.append((x2, y2))
        return points

    def _subdivide(self, points, x1, y1, x2, y2, depth):
        if depth == 0:
            return
        mx = (x1 + x2) / 2 + random.gauss(0, 18)
        my = (y1 + y2) / 2 + random.gauss(0, 18)
        self._subdivide(points, x1, y1, mx, my, depth - 1)
        points.append((int(mx), int(my)))
        self._subdivide(points, mx, my, x2, y2, depth - 1)

    def draw(self, frame, cx, cy):
        if not self.active:
            return frame
        self.frame_count += 1
        if self.frame_count > self.duration:
            self.active = False
            return frame
        t_norm = self.frame_count / self.duration
        h, w = frame.shape[:2]
        out = frame.copy()
        if self.frame_count % 3 == 0:
            ox, oy = self.origin
            targets = [
                (ox + random.randint(-60, 60), max(0, oy - random.randint(100, h - 20))),
                (ox + random.randint(-80, 80), max(0, oy - random.randint(80, h - 20))),
            ]
            self.bolts = [self._make_bolt(ox, oy, tx, ty) for tx, ty in targets]
            for bolt in list(self.bolts):
                for i in range(0, len(bolt), len(bolt) // 3 + 1):
                    if random.random() < 0.3:
                        bx = bolt[i][0] + random.randint(-60, 60)
                        by = bolt[i][1] + random.randint(-60, 60)
                        self.bolts.append(self._make_bolt(bolt[i][0], bolt[i][1], bx, by, depth=3))
        glow_layer = np.zeros_like(frame, dtype=np.float32)
        alpha = 1.0 - t_norm * 0.5
        for bolt in self.bolts:
            pts = np.array(bolt, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(glow_layer, [pts], False, (200, 120, 60), 8)
        glow_blur = cv2.GaussianBlur(glow_layer, (31, 31), 0)
        out = np.clip(out.astype(np.float32) + glow_blur * alpha * 1.8, 0, 255).astype(np.uint8)
        for bolt in self.bolts:
            pts = np.array(bolt, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, (230, 230, 255), 2)
        if self.frame_count < 6:
            flash_alpha = (6 - self.frame_count) / 6 * 0.5
            white = np.full_like(out, 200)
            out = cv2.addWeighted(out, 1 - flash_alpha, white, flash_alpha, 0)
        return out

    def reset(self):
        self.active = False
        self.frame_count = 0
        self.bolts = []


# ─────────────────────────────────────────────
#  WATER EFFECT
# ─────────────────────────────────────────────
class WaterEffect:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.active = False
        self.frame_count = 0
        self.drops = []
        self.ripples = []
        self.duration = 120

    def trigger(self, x, y):
        self.active = True
        self.frame_count = 0
        self.origin = (x, y)
        for i in range(32):
            angle = i / 32 * math.pi * 2
            speed = random.uniform(3, 9)
            self.drops.append({
                'x': float(x), 'y': float(y),
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 3,
                'life': random.randint(30, 55), 'max_life': 50,
            })
        self.ripples.append({'x': x, 'y': y, 'r': 5, 'life': 60})

    def draw(self, frame, cx, cy):
        if not self.active and not self.drops and not self.ripples:
            return frame
        self.frame_count += 1
        if self.frame_count > self.duration:
            self.active = False
        if self.active and self.frame_count % 25 == 0:
            self.ripples.append({'x': cx, 'y': cy, 'r': 5, 'life': 60})
            for i in range(16):
                angle = i / 16 * math.pi * 2
                speed = random.uniform(2, 6)
                self.drops.append({
                    'x': float(cx), 'y': float(cy),
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed - 2,
                    'life': random.randint(20, 40), 'max_life': 35,
                })
        if self.active:
            blue_tint = frame.copy().astype(np.float32)
            blue_tint[:, :, 0] = np.clip(blue_tint[:, :, 0] * 1.4, 0, 255)
            blue_tint[:, :, 2] = blue_tint[:, :, 2] * 0.7
            frame = cv2.addWeighted(frame, 0.82, blue_tint.astype(np.uint8), 0.18, 0)
        out = frame.copy()
        alive_ripples = []
        for rip in self.ripples:
            alpha = rip['life'] / 60
            color = (int(200 * alpha), int(180 * alpha), int(80 * alpha))
            cv2.ellipse(out, (int(rip['x']), int(rip['y'])),
                        (int(rip['r']), int(rip['r'] * 0.4)), 0, 0, 360,
                        color, max(1, int(2 * alpha)))
            rip['r'] += 4
            rip['life'] -= 1
            if rip['life'] > 0:
                alive_ripples.append(rip)
        self.ripples = alive_ripples
        overlay = np.zeros_like(out, dtype=np.float32)
        alive_drops = []
        for d in self.drops:
            t = d['life'] / d['max_life']
            cx_d, cy_d = int(d['x']), int(d['y'])
            if 0 <= cx_d < self.w and 0 <= cy_d < self.h:
                angle = math.atan2(d['vy'], d['vx'])
                cv2.ellipse(overlay, (cx_d, cy_d),
                            (max(1, int(8 * t)), max(1, int(3 * t))),
                            int(math.degrees(angle)), 0, 360,
                            (int(220 * t), int(160 * t), int(60 * t)), -1)
            d['x'] += d['vx']
            d['y'] += d['vy']
            d['vy'] += 0.25
            d['vx'] *= 0.97
            d['life'] -= 1
            if d['life'] > 0:
                alive_drops.append(d)
        self.drops = alive_drops
        glow = cv2.GaussianBlur(overlay, (15, 15), 0)
        return np.clip(out.astype(np.float32) + glow * 1.3, 0, 255).astype(np.uint8)

    def reset(self):
        self.active = False
        self.frame_count = 0
        self.drops = []
        self.ripples = []


# ─────────────────────────────────────────────
#  SHADOW CLONE EFFECT  —  Kage Bunshin no Jutsu
# ─────────────────────────────────────────────
class ShadowCloneEffect:
    """
    How it works:
    ┌─────────────────────────────────────────────────────┐
    │ 1. A rolling deque stores the last BUFFER_SIZE raw  │
    │    webcam frames (fed every frame before effects).  │
    │ 2. On trigger: 2 (or 3) clone slots are created,   │
    │    each with a horizontal offset.                   │
    │ 3. Each clone renders a DELAYED frame (pulled from  │
    │    ~8 frames ago) shifted sideways.                 │
    │ 4. Clones fade in (0→1.0 alpha, 15 frames),        │
    │    hold, then dissolve with a pixel-scatter effect. │
    │ 5. White flash ring + spark scatter on spawn.       │
    │ 6. Smoke puffs drift upward on appear & disappear. │
    └─────────────────────────────────────────────────────┘
    """

    DELAY_FRAMES = 8    # how old the ghost frame is (gives visual lag feel)
    BUFFER_SIZE  = 30   # must be > DELAY_FRAMES

    def __init__(self, w, h, num_clones=2):
        self.w = w
        self.h = h
        self.num_clones = num_clones
        self.active = False
        self.frame_count = 0
        self.duration = 180       # total frames the jutsu lasts
        self.dissolve_start = 130 # frame at which clones start to fade

        self.buffer = deque(maxlen=self.BUFFER_SIZE)
        self.clones = []
        self.smoke  = []

    # ── Public ──────────────────────────────────────────

    def feed(self, frame):
        """Feed the raw frame into the history buffer each loop."""
        self.buffer.append(frame.copy())

    def trigger(self):
        if len(self.buffer) < self.DELAY_FRAMES + 2:
            print("⚠  Not enough frame history yet — wait a moment and try again.")
            return
        self.active = True
        self.frame_count = 0
        self.clones = []
        self.smoke  = []

        if self.num_clones == 2:
            offsets = [(-220, 0), (220, 0)]
        else:
            offsets = [(-270, 0), (0, -15), (270, 0)]

        for ox, oy in offsets:
            self._spawn_clone(ox, oy)

    def draw(self, frame):
        if not self.active and not self.smoke and not self.clones:
            return frame

        self.frame_count += 1

        # Pull a past frame for the clone texture
        delay_idx = max(0, len(self.buffer) - self.DELAY_FRAMES - 1)
        ghost_frame = self.buffer[delay_idx] if self.buffer else frame

        h, w = frame.shape[:2]
        out = frame.copy().astype(np.float32)

        # ── Render each clone ────────────────────────────
        for clone in self.clones:

            # ── Alpha envelope ──────────────────────────
            if self.frame_count <= 6:
                # Fade in
                clone['alpha'] = self.frame_count / 15
            elif self.frame_count > self.dissolve_start:
                # Fade out
                t_d = (self.frame_count - self.dissolve_start) / max(1, self.duration - self.dissolve_start)
                clone['alpha'] = max(0.0, 1.0 - t_d)

            if clone['alpha'] <= 0.01:
                continue

            ox = int(clone['offset_x'])
            oy = int(clone['offset_y'])

            # ── Shift the ghost frame sideways ──────────
            M = np.float32([[1, 0, ox], [0, 1, oy]])
            shifted = cv2.warpAffine(ghost_frame, M, (w, h))

            clone_f = shifted.astype(np.float32)

            # ── Dissolve shimmer (pixel scatter) ────────
            if self.frame_count > self.dissolve_start:
                t_d = (self.frame_count - self.dissolve_start) / max(1, self.duration - self.dissolve_start)
                noise_px = int(t_d * 16)
                if noise_px > 0:
                    # Random horizontal displacement per row
                    row_shift = np.random.randint(-noise_px, noise_px + 1, (h,), dtype=np.int32)
                    xs = np.clip(
                        np.arange(w).reshape(1, -1) + row_shift.reshape(-1, 1),
                        0, w - 1
                    ).astype(np.int32)
                    ys = np.arange(h).reshape(-1, 1) * np.ones((1, w), dtype=np.int32)
                    clone_f = clone_f[ys, xs]

                    # Also randomly zero out some pixels (crumble effect)
                    mask = np.random.rand(h, w) < (t_d * 0.6)
                    clone_f[mask] = 0

            # ── Blend clone onto output ──────────────────
            a = clone['alpha']
            out = out * (1.0 - a) + clone_f * a

            # ── Spawn flash ring ─────────────────────────
            if self.frame_count <= 12:
                flash_a = (12 - self.frame_count) / 12.0
                ring_cx = w // 2 + ox
                ring_cy = h // 2 + oy
                ring_r  = int(15 + self.frame_count * 20)
                thickness = max(1, int(10 * flash_a))
                color = (255.0 * flash_a, 255.0 * flash_a, 255.0 * flash_a)
                cv2.circle(out, (ring_cx, ring_cy), ring_r, color, thickness)

            # ── Update sparks ────────────────────────────
            self._update_sparks(clone, out, w, h)

        # ── Smoke puffs ──────────────────────────────────
        self._draw_smoke(out, w, h)

        # Clamp result
        out = np.clip(out, 0, 255).astype(np.uint8)

        if self.frame_count > self.duration:
            self.active = False
            if not self.smoke:
                self.clones = []

        return out

    def reset(self):
        self.active = False
        self.frame_count = 0
        self.clones = []
        self.smoke  = []

    # ── Private helpers ──────────────────────────────────

    def _spawn_clone(self, ox, oy):
        center_x = self.w // 2 + ox
        center_y = self.h // 2 + oy

        # Sparks
        sparks = []
        for _ in range(30):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(2, 8)
            sparks.append({
                'x': center_x + random.gauss(0, 12),
                'y': center_y + random.gauss(0, 12),
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': random.randint(10, 28),
                'max_life': 22,
            })

        self.clones.append({
            'offset_x': ox,
            'offset_y': oy,
            'alpha': 0.0,
            'sparks': sparks,
        })

        # Smoke puff on materialise
        for _ in range(22):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(0.4, 2.8)
            self.smoke.append({
                'x': float(center_x + random.gauss(0, 25)),
                'y': float(center_y + random.gauss(0, 25)),
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed - 1.2,
                'r': random.randint(10, 28),
                'life': random.randint(18, 48),
                'max_life': 38,
            })

    def _update_sparks(self, clone, out, w, h):
        alive = []
        for sp in clone['sparks']:
            t = sp['life'] / sp['max_life']
            sx, sy = int(sp['x']), int(sp['y'])
            ex = int(sx + sp['vx'] * 4)
            ey = int(sy + sp['vy'] * 4)
            if 0 <= sx < w and 0 <= sy < h:
                col = (180.0 * t, 210.0 * t, 255.0 * t)
                cv2.line(out, (sx, sy), (ex, ey), col, 1)
            sp['x'] += sp['vx']
            sp['y'] += sp['vy']
            sp['vy'] += 0.25  # gravity
            sp['life'] -= 1
            if sp['life'] > 0:
                alive.append(sp)
        clone['sparks'] = alive

    def _draw_smoke(self, out, w, h):
        alive = []
        for sm in self.smoke:
            t = sm['life'] / sm['max_life']
            # Expand radius as smoke fades
            r = max(1, int(sm['r'] * (1.0 + (1.0 - t) * 1.2)))
            sx, sy = int(sm['x']), int(sm['y'])
            if 0 <= sx < w and 0 <= sy < h:
                brightness = 140.0 * t
                cv2.circle(out, (sx, sy), r,
                           (brightness, brightness, brightness + 15.0), -1)
            sm['x'] += sm['vx']
            sm['y'] += sm['vy']
            sm['life'] -= 1
            if sm['life'] > 0:
                alive.append(sm)
        self.smoke = alive


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam (index 0). Trying index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("❌ No webcam found. Exiting.")
            return

    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read first frame. Exiting.")
        cap.release()
        return

    h, w = frame.shape[:2]

    fire      = FireEffect()
    lightning = LightningEffect()
    water     = WaterEffect(w, h)
    shadow    = ShadowCloneEffect(w, h, num_clones=2)

    cx, cy = w // 2, h // 2

    print("\n🍃  Naruto Jutsu Effects")
    print("─────────────────────────────────")
    print("  F  →  Fire Jutsu          (Katon · Gōkakyū)")
    print("  L  →  Lightning           (Raiton · Chidori)")
    print("  W  →  Water               (Suiton · Suiryūdan)")
    print("  S  →  Shadow Clone  ×2    (Kage Bunshin no Jutsu)")
    print("  3  →  Shadow Clone  ×3")
    print("  R  →  Reset all effects")
    print("  Q  →  Quit")
    print("\n  (Shadow clone needs ~1 second of camera warmup)\n")

    active_label = ""
    label_timer  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror so movement feels natural

        # Feed raw frame into history BEFORE effects are applied
        shadow.feed(frame)

        # Stack all effects
        frame = water.draw(frame, cx, cy)
        frame = fire.draw(frame, cx, cy)
        frame = lightning.draw(frame, cx, cy)
        frame = shadow.draw(frame)

        # HUD strip
        hud = "F=Fire  L=Lightning  W=Water  S=Clone x2  3=Clone x3  R=Reset  Q=Quit"
        cv2.putText(frame, hud, (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        # Jutsu name banner
        if label_timer > 0:
            tx = w // 2 - 210
            # Drop shadow
            cv2.putText(frame, active_label, (tx + 2, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 3, cv2.LINE_AA)
            # Main text
            cv2.putText(frame, active_label, (tx, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 220, 80), 2, cv2.LINE_AA)
            label_timer -= 1

        cv2.imshow("Naruto Jutsu Effects", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        elif key in (ord('f'), ord('F')):
            fire.trigger(cx, cy)
            active_label = "Katon - Goukakyuu no Jutsu!"
            label_timer = 80

        elif key in (ord('l'), ord('L')):
            lightning.trigger(cx, cy)
            active_label = "Raiton - Chidori!"
            label_timer = 80

        elif key in (ord('w'), ord('W')):
            water.trigger(cx, cy)
            active_label = "Suiton - Suiryuudan no Jutsu!"
            label_timer = 80

        elif key in (ord('s'), ord('S')):
            shadow.num_clones = 2
            shadow.trigger()
            active_label = "Kage Bunshin no Jutsu!"
            label_timer = 110

        elif key == ord('3'):
            shadow.num_clones = 3
            shadow.trigger()
            active_label = "Kage Bunshin no Jutsu!  x3"
            label_timer = 110

        elif key in (ord('r'), ord('R')):
            fire.reset()
            lightning.reset()
            water.reset()
            shadow.reset()
            active_label = ""
            label_timer  = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
