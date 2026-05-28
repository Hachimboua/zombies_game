#!/usr/bin/env python3
"""
Test Environment – Interactive 1-Bit Dithered Demo
====================================================
Requires:  pip install ursina

Controls
--------
  WASD          Move
  Mouse         Look
  Space         Jump
  E             Interact with object under crosshair
  T             Toggle dither shader on/off
  ESC           Quit

Goal
----
Find the brass key, unlock the back door, claim the trophy.

Interactable object types demonstrated
--------------------------------------
  Pickup    coins, key (added to inventory, removed from world)
  Sign      shows a text message in the HUD message log
  Switch    flips state; lights up a billboard "lamp" sprite
  Door      slides open; can require an inventory key
  Trophy    end-state object; triggers a win message
"""
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from panda3d.core import (
    Texture as P3DTexture, SamplerState, LVecBase2f, LVecBase3f,
    Shader as P3DShader, AntialiasAttrib,
)
from direct.filter.FilterManager import FilterManager
from PIL import Image, ImageDraw
from pathlib import Path
from functools import lru_cache
import math
import random

HUD_SPRITES_DIR = Path(__file__).resolve().parent / 'textures' / 'sprites'


# ──────────────────────────────────────────────────────────────────────────────
# GLSL – Bayer 4×4 ordered-dither post-process shader
# ──────────────────────────────────────────────────────────────────────────────
_VERT = """
#version 140
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 uv;

void main() {
    gl_Position = p3d_Vertex;
    uv = p3d_MultiTexCoord0;
}
"""

_FRAG = """
#version 140
uniform sampler2D scene_tex;
uniform vec2      resolution;
// Two-colour palette — pixel goes to fg_color when above the Bayer threshold,
// else bg_color. Drives the day/night lighting *as actual colour*, not just
// brightness.  Default NOON = white/black.
uniform vec3      fg_color;
uniform vec3      bg_color;
in  vec2 uv;
out vec4 frag_color;

float bayer4(ivec2 p) {
    const int M[16] = int[16](
         0,  8,  2, 10,
        12,  4, 14,  6,
         3, 11,  1,  9,
        15,  7, 13,  5
    );
    return float(M[(p.y & 3) * 4 + (p.x & 3)]) / 16.0;
}

void main() {
    vec3  rgb    = texture(scene_tex, uv).rgb;
    float luma   = dot(rgb, vec3(0.299, 0.587, 0.114));
    float thresh = bayer4(ivec2(uv * resolution));
    float bit    = luma > thresh ? 1.0 : 0.0;
    frag_color   = vec4(mix(bg_color, fg_color, bit), 1.0);
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helper: spawn a brief bright spark at a hit point (bullet feedback)
# ──────────────────────────────────────────────────────────────────────────────
def _spawn_spark(pos):
    s = Entity(model='sphere', position=pos, scale=0.18, color=color.white)
    s.animate_scale(0.02, duration=0.16)
    # Give the animator a comfortable margin so destroy() never races the
    # final scale frame — that race triggers a Panda3D NodePath assert.
    invoke(destroy, s, delay=0.30)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: nearest-neighbour filtering on an entity's texture
# ──────────────────────────────────────────────────────────────────────────────
def set_nearest(entity: Entity) -> None:
    if entity.texture and hasattr(entity.texture, '_texture'):
        t = entity.texture._texture
        t.setMagfilter(SamplerState.FT_nearest)
        t.setMinfilter(SamplerState.FT_nearest)
        t.setAnisotropicDegree(0)


def planter(x, z, size=1.0, sprite='cactus'):
    pot_h = 0.55 * size
    pot_w = 0.95 * size
    Entity(model='cube', position=(x, pot_h / 2, z),
        scale=(pot_w, pot_h, pot_w),
        texture=TERRA_TEX, texture_scale=(1.0, 1.0),
        color=color.rgb(175, 110, 70), collider='box')
    Entity(model='cube', position=(x, pot_h + 0.08, z),
        scale=(pot_w * 0.78, 0.14 * size, pot_w * 0.78),
        texture=STONE_TEX, texture_scale=(1.0, 1.0),
        color=color.rgb(130, 110, 90))
    DecorSprite(sprite=sprite, position=(x, pot_h + 0.65 * size, z),
          scale=(0.95 * size, 1.25 * size), static=False)


def load_cropped_sprite(name: str):
    img = Image.open(HUD_SPRITES_DIR / f'{name}.png').convert('RGBA')
    bbox = img.getchannel('A').getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    tex = Texture(img)
    if hasattr(tex, '_texture'):
        tex._texture.setMagfilter(SamplerState.FT_nearest)
        tex._texture.setMinfilter(SamplerState.FT_nearest)
        tex._texture.setAnisotropicDegree(0)
    return tex


def load_sprite_sheet(name: str, columns: int, rows: int):
    img = Image.open(Path(__file__).resolve().parent / 'Gun Sprites' / f'{name}.png').convert('RGBA')
    bg = img.getpixel((0, 0))
    pixels = img.load()
    for y in range(img.height):
        for x in range(img.width):
            if pixels[x, y] == bg:
                pixels[x, y] = (0, 0, 0, 0)
    bbox = img.getchannel('A').getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    frame_w = img.width // columns
    frame_h = img.height // rows
    frames = []
    for row in range(rows):
        for column in range(columns):
            left = column * frame_w
            top = row * frame_h
            right = img.width if column == columns - 1 else (column + 1) * frame_w
            bottom = img.height if row == rows - 1 else (row + 1) * frame_h
            frame = img.crop((left, top, right, bottom))
            tex = Texture(frame)
            if hasattr(tex, '_texture'):
                tex._texture.setMagfilter(SamplerState.FT_nearest)
                tex._texture.setMinfilter(SamplerState.FT_nearest)
                tex._texture.setAnisotropicDegree(0)
            frames.append(tex)
    return frames


def load_detected_sprite_frames(name: str):
    img = Image.open(Path(__file__).resolve().parent / 'Gun Sprites' / f'{name}.png').convert('RGBA')
    pixels = img.load()
    row_has = [any(pixels[x, y][3] > 0 for x in range(img.width)) for y in range(img.height)]
    col_has = [any(pixels[x, y][3] > 0 for y in range(img.height)) for x in range(img.width)]

    def _runs(flags):
        ranges = []
        start = None
        for index, flag in enumerate(flags):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                ranges.append((start, index - 1))
                start = None
        if start is not None:
            ranges.append((start, len(flags) - 1))
        return ranges

    row_runs = _runs(row_has)
    col_runs = _runs(col_has)
    frames = []
    for row_start, row_end in row_runs:
        for col_start, col_end in col_runs:
            frame = img.crop((col_start, row_start, col_end + 1, row_end + 1))
            bbox = frame.getchannel('A').getbbox()
            if bbox is not None:
                frame = frame.crop(bbox)
            tex = Texture(frame)
            if hasattr(tex, '_texture'):
                tex._texture.setMagfilter(SamplerState.FT_nearest)
                tex._texture.setMinfilter(SamplerState.FT_nearest)
                tex._texture.setAnisotropicDegree(0)
            frames.append(tex)
    return frames


def _load_texture(path: Path):
    tex = Texture(Image.open(path).convert('RGBA'))
    if hasattr(tex, '_texture'):
        tex._texture.setMagfilter(SamplerState.FT_nearest)
        tex._texture.setMinfilter(SamplerState.FT_nearest)
        tex._texture.setAnisotropicDegree(0)
    return tex


@lru_cache(maxsize=None)
def load_character_animations(character: str):
    root = Path(__file__).resolve().parent / 'Free Characters with Animations For FPS game'
    # Normalize to Title case to match filenames like 'ChargerIdle.png'
    base = character[0].upper() + character[1:] if character else character
    sequences = {
        'idle':   [f'{base}Idle.png'],
        'walk':   [f'{base}Walk{i}.png' for i in range(1, 5)],
        'attack': [f'{base}Attack{i}.png' for i in range(1, 3)],
        'damage': [f'{base}Damage{i}.png' for i in range(1, 3)],
        'death':  [f'{base}Death{i}.png' for i in range(1, 5)],
    }
    return {
        state: [_load_texture(root / filename) for filename in filenames]
        for state, filenames in sequences.items()
    }


class AnimatedStateBillboard(Entity):
    def __init__(self, animations, state='idle', frame_time=0.10,
                 loop_states=None, **kwargs):
        kwargs.setdefault('model', 'quad')
        kwargs.setdefault('double_sided', True)
        super().__init__(**kwargs)
        self.frame_animations = animations
        self.state = state if state in self.frame_animations else 'idle'
        self.frame_time = frame_time
        self.loop_states = set(loop_states or ('idle', 'walk'))
        self._frame_index = 0
        self._timer = 0.0
        self._playing = True
        self.texture = self.frame_animations[self.state][0]

    def set_state(self, state, restart=False, frame_time=None):
        if state not in self.frame_animations:
            state = 'idle'
        if frame_time is not None:
            self.frame_time = frame_time
        if not restart and state == self.state and self._playing:
            return
        self.state = state
        self._frame_index = 0
        self._timer = 0.0
        self._playing = True
        self.texture = self.frame_animations[self.state][0]

    def update(self):
        frames = self.frame_animations.get(self.state)
        if not frames or len(frames) <= 1 or not self._playing:
            return
        self._timer += time.dt
        if self._timer < self.frame_time:
            return
        self._timer = 0.0
        if self._frame_index < len(frames) - 1:
            self._frame_index += 1
        elif self.state in self.loop_states:
            self._frame_index = 0
        else:
            self._playing = False
        self.texture = frames[self._frame_index]


class AnimatedSprite(Entity):
    def __init__(self, frames, frame_time=0.08, loop=True, **kwargs):
        super().__init__(model='quad', double_sided=True, **kwargs)
        self.frames = frames
        self.frame_time = frame_time
        self.loop = loop
        self._frame_index = 0
        self._timer = 0.0
        self._playing = False
        self.texture = self.frames[0]

    def play_once(self, frame_time=None):
        if frame_time is not None:
            self.frame_time = frame_time
        self._frame_index = 0
        self._timer = 0.0
        self._playing = True
        self.texture = self.frames[0]

    def update(self):
        if not self._playing or len(self.frames) <= 1:
            return
        self._timer += time.dt
        if self._timer < self.frame_time:
            return
        self._timer = 0.0
        self._frame_index += 1
        if self._frame_index >= len(self.frames):
            if self.loop:
                self._frame_index = 0
            else:
                self._frame_index = 0
                self._playing = False
        self.texture = self.frames[self._frame_index]


# ──────────────────────────────────────────────────────────────────────────────
# Y-axis Billboard sprite
# ──────────────────────────────────────────────────────────────────────────────
class BillboardSprite(Entity):
    def __init__(self, **kwargs):
        super().__init__(model='quad', double_sided=True, **kwargs)

    def update(self):
        dx = camera.world_position.x - self.world_position.x
        dz = camera.world_position.z - self.world_position.z
        self.rotation_y = math.degrees(math.atan2(dx, dz))


class DecorSprite(BillboardSprite):
    """A 2D image whose texture is looked up by name from the SPRITES dict.
       - static=False (default): always faces the player on the Y axis
       - static=True:             keeps whatever rotation_y you set after init,
                                  acts like a flat picture pinned to a wall."""

    def __init__(self, sprite, static=False, **kwargs):
        kwargs['texture'] = SPRITES[sprite]
        # Sensible defaults for an upright billboard sitting on the floor.
        kwargs.setdefault('scale', (1.5, 2.0))
        kwargs.setdefault('origin_y', -0.5)         # so y=0 sits on the floor
        super().__init__(**kwargs)
        self.static = static

    def update(self):
        if not self.static:
            super().update()


# ──────────────────────────────────────────────────────────────────────────────
# Smooth first-person controller
# ──────────────────────────────────────────────────────────────────────────────
class SmoothFPC(FirstPersonController):
    """
    Drop-in replacement for FirstPersonController with:
      - Velocity-based acceleration / friction (no instant start–stop)
      - Reduced air control
      - Exponentially smoothed mouse look
      - Custom velocity-based jump + gravity (no Y-animation fighting gravity)
      - Subtle walking head-bob
    Tune the *_TUNING* fields below to taste.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('speed', 6)
        kwargs.setdefault('jump_height', 1.8)
        kwargs.setdefault('mouse_sensitivity', Vec2(40, 40))
        super().__init__(**kwargs)

        # ── Tuning ──────────────────────────────────────────────
        self.accel_ground    = 16.0    # how fast we reach top speed
        self.accel_air       = 3.0     # mid-air responsiveness
        self.friction        = 14.0    # how fast we stop
        self.mouse_smoothing = 0.04    # seconds of lag; 0 = raw
        self.gravity_strength = 22.0
        self.bob_amount      = 0.06
        self.bob_speed       = 9.0

        # ── State ───────────────────────────────────────────────
        self._vel        = Vec3(0, 0, 0)   # smoothed horizontal velocity
        self._y_vel      = 0.0             # vertical velocity (jump + gravity)
        self._mouse_vel  = Vec2(0, 0)      # smoothed mouse delta
        self._bob_t      = 0.0
        self._cam_base_y = self.camera_pivot.y
        self.grounded    = True

        # ── Dash (Shift) ────────────────────────────────────────
        self.dash_speed       = 22.0       # m/s during the burst
        self.dash_duration    = 0.18
        self.dash_cooldown    = 1.5
        self._dash_timer      = 0.0
        self._dash_cd         = 0.0
        self._dash_dir        = Vec3(0, 0, 0)
        self.invincible_until = 0.0        # time.time() — used by PlayerHUD

    # ── Jump uses velocity, not Y-animation, so it can't fight gravity ──
    def jump(self):
        if self.grounded:
            self._y_vel   = math.sqrt(2 * self.gravity_strength * self.jump_height)
            self.grounded = False

    # ── Dash: hit Shift to burst in the current movement direction ─────
    def input(self, key):
        # Ursina may emit any of these for the Shift modifier.
        if key in ('shift', 'left shift', 'right shift', 'shift hold'):
            self._try_dash()

    def _try_dash(self):
        if self._dash_cd > 0 or self._dash_timer > 0:
            return
        # Build a horizontal wish vector from input keys; if no WASD held,
        # dash forward.
        ix = held_keys['d'] - held_keys['a']
        iz = held_keys['w'] - held_keys['s']
        wish = self.forward * iz + self.right * ix
        wish.y = 0
        if wish.length() < 0.1:
            wish = Vec3(self.forward.x, 0, self.forward.z)
        self._dash_dir         = wish.normalized()
        self._dash_timer       = self.dash_duration
        self._dash_cd          = self.dash_cooldown
        self.invincible_until  = time.time() + self.dash_duration + 0.05

    def update(self):
        # ── Smoothed mouse look ────────────────────────────────────
        raw_m = Vec2(mouse.velocity[0], mouse.velocity[1])
        m_t   = clamp(time.dt / max(self.mouse_smoothing, 1e-4), 0, 1)
        self._mouse_vel = lerp(self._mouse_vel, raw_m, m_t)
        self.rotation_y += self._mouse_vel.x * self.mouse_sensitivity[1]
        self.camera_pivot.rotation_x -= self._mouse_vel.y * self.mouse_sensitivity[0]
        self.camera_pivot.rotation_x = clamp(self.camera_pivot.rotation_x, -90, 90)

        # ── Input → world-space wish direction (XZ plane only) ─────
        ix = held_keys['d'] - held_keys['a']
        iz = held_keys['w'] - held_keys['s']
        wish = self.forward * iz + self.right * ix
        wish.y = 0
        wish = wish.normalized() if wish.length() > 1e-3 else Vec3(0, 0, 0)

        # ── Dash takes over horizontal velocity ────────────────────
        if self._dash_timer > 0:
            self._dash_timer -= time.dt
            self._vel.x = self._dash_dir.x * self.dash_speed
            self._vel.z = self._dash_dir.z * self.dash_speed
        else:
            # ── Smooth horizontal velocity ─────────────────────────
            target_v = wish * self.speed
            moving   = wish.length() > 0
            if self.grounded:
                rate = self.accel_ground if moving else self.friction
            else:
                rate = self.accel_air
            t = clamp(rate * time.dt, 0, 1)
            self._vel.x = lerp(self._vel.x, target_v.x, t)
            self._vel.z = lerp(self._vel.z, target_v.z, t)

        if self._dash_cd > 0:
            self._dash_cd -= time.dt

        # ── Apply horizontal movement w/ axis-separated collision ──
        dx = self._vel.x * time.dt
        dz = self._vel.z * time.dt
        if abs(dx) > 1e-5:
            if self._wall_in(Vec3(1 if dx > 0 else -1, 0, 0), abs(dx) + 0.25):
                self._vel.x = 0
            else:
                self.x += dx
        if abs(dz) > 1e-5:
            if self._wall_in(Vec3(0, 0, 1 if dz > 0 else -1), abs(dz) + 0.25):
                self._vel.z = 0
            else:
                self.z += dz

        # ── Vertical movement: gravity + ground-snap + ceiling stop ─
        if self.gravity:
            # Cast a long ray from above the head so we always find the floor,
            # even if the player has briefly clipped below it.
            ground = raycast(self.world_position + Vec3(0, self.height + 0.5, 0),
                             Vec3(0, -1, 0),
                             distance=self.height * 4 + 1,
                             ignore=(self,))

            if ground.hit:
                ground_y      = ground.world_point.y
                feet_distance = self.y - ground_y   # +ve = above ground

                if self._y_vel <= 0 and feet_distance <= 0.1:
                    # Touching ground (or slightly inside) and not jumping → snap.
                    if not self.grounded:
                        self.land()
                    self.grounded = True
                    self._y_vel   = 0
                    self.y        = ground_y
                else:
                    # Airborne — apply gravity, then catch fall-throughs.
                    self.grounded = False
                    self._y_vel  -= self.gravity_strength * time.dt
                    self.y       += self._y_vel * time.dt
                    if self.y < ground_y:
                        self.y       = ground_y
                        self._y_vel  = 0
                        self.grounded = True
                    elif self._y_vel > 0:
                        ceil = raycast(
                            self.world_position + Vec3(0, self.height, 0),
                            Vec3(0, 1, 0), distance=0.1, ignore=(self,))
                        if ceil.hit:
                            self._y_vel = 0
            else:
                # No floor anywhere below (edge of map) — keep falling, but
                # cap velocity so we don't accelerate forever.
                self.grounded = False
                self._y_vel  -= self.gravity_strength * time.dt
                self._y_vel   = max(self._y_vel, -50.0)
                self.y       += self._y_vel * time.dt

        # ── Head-bob ───────────────────────────────────────────────
        sf = clamp(self._vel.length() / max(self.speed, 1e-3), 0, 1)
        if self.grounded and sf > 0.1:
            self._bob_t += time.dt * self.bob_speed
            target_y     = self._cam_base_y + math.sin(self._bob_t) * self.bob_amount * sf
        else:
            self._bob_t  = 0
            target_y     = self._cam_base_y
        self.camera_pivot.y = lerp(self.camera_pivot.y, target_y,
                                   clamp(time.dt * 10, 0, 1))

    STEP_UP = 0.55     # max height of an obstacle the player can walk onto

    def _wall_in(self, direction, distance):
        """Cast rays at chest + head — anything BELOW STEP_UP is climbable
        (gravity will snap us up onto its top surface next frame)."""
        for h in (self.STEP_UP + 0.15,             # just above the step zone
                  self.height * 0.6,
                  max(self.height - 0.2, 0.4)):
            origin = self.world_position + Vec3(0, h, 0)
            if raycast(origin, direction, distance=distance, ignore=(self,)).hit:
                return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Interactable base class
# ──────────────────────────────────────────────────────────────────────────────
class Interactable(Entity):
    """
    Any entity the player can target with the crosshair and trigger with [E].
    Subclasses override `interact()`; the `prompt` string is what the HUD
    shows under the crosshair when this object is in focus.
    """
    def __init__(self, prompt='Interact', **kwargs):
        kwargs.setdefault('collider', 'box')
        super().__init__(**kwargs)
        self.prompt = prompt

    def interact(self):
        pass


# ── Pickup: 2D billboard sprite; vanishes into inventory on interact ─────────
class Pickup(Interactable):
    def __init__(self, item_name='Item', sprite=None, **kwargs):
        # Sprite name defaults to a lowercased version of the item name
        # ("Brass Key" → "brass_key"); pass `sprite=...` to override.
        sprite_name = sprite or item_name.lower().replace(' ', '_')
        kwargs['model']       = 'quad'
        kwargs['texture']     = SPRITES[sprite_name]
        kwargs['double_sided'] = True
        kwargs.setdefault('scale', (0.55, 0.55))
        super().__init__(prompt=f'Pick up {item_name}', **kwargs)
        self.item_name = item_name
        self._base_y   = self.y
        self._t        = 0.0

    def update(self):
        # Bob in place + always face the camera on the Y axis
        self._t += time.dt
        self.y = self._base_y + math.sin(self._t * 3) * 0.1
        dx = camera.world_position.x - self.world_position.x
        dz = camera.world_position.z - self.world_position.z
        self.rotation_y = math.degrees(math.atan2(dx, dz))

    def interact(self):
        inventory.add(self.item_name)
        message_log.show(f'Picked up: {self.item_name}')
        destroy(self)


# ── Sign: shows a text message in the HUD ────────────────────────────────────
class Sign(Interactable):
    def __init__(self, message='', **kwargs):
        kwargs.setdefault('model', 'cube')
        kwargs.setdefault('scale', (1.2, 0.7, 0.1))
        kwargs.setdefault('color', color.rgb(190, 170, 110))
        super().__init__(prompt='Read sign', **kwargs)
        self.message = message

    def interact(self):
        message_log.show(self.message, duration=5)


# ── Switch: toggles state; can drive connected entities ──────────────────────
class Switch(Interactable):
    def __init__(self, on_toggle=None, **kwargs):
        kwargs.setdefault('model', 'cube')
        kwargs.setdefault('scale', (0.3, 0.5, 0.1))
        kwargs.setdefault('color', color.rgb(120, 120, 120))
        super().__init__(prompt='Flip switch', **kwargs)
        self.is_on      = False
        self.on_toggle  = on_toggle      # callback(is_on: bool)
        self._handle    = Entity(parent=self, model='cube',
                                 scale=(0.5, 0.15, 1.1),
                                 position=(0, -0.15, 0),
                                 color=color.rgb(60, 60, 60))

    def interact(self):
        self.is_on = not self.is_on
        # Rotate the handle to show state.
        self._handle.animate_rotation(
            (0, 0, 30 if self.is_on else -30), duration=0.15)
        if self.on_toggle:
            self.on_toggle(self.is_on)
        message_log.show(f'Switch: {"ON" if self.is_on else "OFF"}')


# ── Door: slides open; can require a key from inventory ──────────────────────
class Door(Interactable):
    def __init__(self, key_required=None, slide_axis='x', **kwargs):
        kwargs.setdefault('model', 'cube')
        kwargs.setdefault('color', color.rgb(120, 80, 50))
        super().__init__(prompt='Open door', **kwargs)
        self.key_required = key_required
        self.is_open      = False
        self.closed_pos   = Vec3(self.position)
        # Slide door sideways into the adjacent wall when opened.
        slide = Vec3(self.scale_x, 0, 0) if slide_axis == 'x' \
                else Vec3(0, 0, self.scale_z)
        self.open_pos     = self.closed_pos + slide

    def interact(self):
        if not self.is_open and self.key_required:
            if not inventory.has(self.key_required):
                message_log.show(f'Locked. Requires: {self.key_required}.')
                return
            message_log.show(f'Unlocked with {self.key_required}.')
            self.key_required = None     # door stays unlocked

        self.is_open = not self.is_open
        target = self.open_pos if self.is_open else self.closed_pos
        self.animate_position(target, duration=0.6, curve=curve.in_out_quad)
        self.prompt = 'Close door' if self.is_open else 'Open door'


# ── Trophy: 2D billboard sprite end-state object ─────────────────────────────
class Trophy(Interactable):
    def __init__(self, **kwargs):
        kwargs['model']       = 'quad'
        kwargs['texture']     = SPRITES['trophy']
        kwargs['double_sided'] = True
        kwargs.setdefault('scale', (0.8, 1.4))
        super().__init__(prompt='Claim trophy', **kwargs)

    def update(self):
        dx = camera.world_position.x - self.world_position.x
        dz = camera.world_position.z - self.world_position.z
        self.rotation_y = math.degrees(math.atan2(dx, dz))

    def interact(self):
        message_log.show('You claimed the trophy! Demo complete.',
                         duration=8)
        destroy(self)


# ──────────────────────────────────────────────────────────────────────────────
# HUD: Inventory display
# ──────────────────────────────────────────────────────────────────────────────
class Inventory:
    def __init__(self):
        self.items = []
        self.text  = Text('', position=(-0.86, 0.42), origin=(-0.5, 0.5),
                          scale=1.0, color=color.white, parent=camera.ui)
        self._refresh()

    def add(self, item):
        self.items.append(item)
        self._refresh()

    def has(self, item):
        return item in self.items

    def _refresh(self):
        if not self.items:
            self.text.text = 'INVENTORY\n(empty)'
        else:
            self.text.text = 'INVENTORY\n' + '\n'.join(f' • {i}' for i in self.items)


# ──────────────────────────────────────────────────────────────────────────────
# HUD: Message log (auto-fading text at bottom of screen)
# ──────────────────────────────────────────────────────────────────────────────
class MessageLog(Entity):
    def __init__(self):
        super().__init__()
        self.text  = Text('', position=(0, -0.40), origin=(0, 0),
                          scale=1.2, color=color.white, parent=camera.ui)
        self.timer = 0.0

    def show(self, msg, duration=3.0):
        self.text.text = msg
        self.timer     = duration

    def update(self):
        if self.timer > 0:
            self.timer -= time.dt
            if self.timer <= 0:
                self.text.text = ''


# ──────────────────────────────────────────────────────────────────────────────
# Raycast interaction system
# ──────────────────────────────────────────────────────────────────────────────
class InteractionSystem(Entity):
    """Casts a short ray from the camera each frame. Shows a prompt under
    the crosshair when looking at an Interactable; presses E to trigger."""

    MAX_DISTANCE = 3.0

    def __init__(self, player_entity):
        super().__init__()
        self.player = player_entity
        self.target = None
        self.prompt = Text('', origin=(0, 0), position=(0, -0.04),
                           scale=1.1, color=color.white,
                           parent=camera.ui, enabled=False)

    def update(self):
        origin    = camera.world_position
        direction = camera.forward
        hit = raycast(origin, direction,
                      distance=self.MAX_DISTANCE,
                      ignore=(self.player,),
                      debug=False)

        if hit.hit and isinstance(hit.entity, Interactable):
            self.target = hit.entity
            self.prompt.text    = f'[E] {hit.entity.prompt}'
            self.prompt.enabled = True
        else:
            self.target         = None
            self.prompt.enabled = False

    def input(self, key):
        if key == 'e' and self.target is not None:
            self.target.interact()


# ──────────────────────────────────────────────────────────────────────────────
# Enemy — 2D billboard that walks toward the player and melees them
# ──────────────────────────────────────────────────────────────────────────────
class Enemy(AnimatedStateBillboard):
    """2D billboard enemy animated with the Commando frame set.
    Two flavours via `ranged=True`:
      - melee:  walks straight in and hits you on contact (default)
      - ranged: keeps distance, telegraphs a yellow flash, then hitscans
    """

    def __init__(self, sprite=None, hp=3, speed=2.0, damage=10,
                 ranged=False, fire_range=22.0, prefer_range=12.0,
                 shot_damage=6, shot_cooldown=2.0, **kwargs):
        kwargs['collider'] = 'box'
        kwargs.setdefault('scale', (1.7, 2.3))
        kwargs.setdefault('origin_y', -0.5)
        animations = load_character_animations(sprite or 'Commando')
        kwargs['texture'] = animations['idle'][0]
        super().__init__(animations=animations, state='idle', frame_time=0.10,
                         loop_states=('idle', 'walk'), **kwargs)
        self.max_hp = hp
        self.hp = hp
        self.speed = speed
        self.damage = damage
        self.attack_range = 1.6
        self.attack_cooldown = 1.0
        self._attack_timer = 0.0
        self._hurt_timer = 0.0
        self.ranged = ranged
        self.fire_range = fire_range
        self.prefer_range = prefer_range
        self.shot_damage = shot_damage
        self.shot_cooldown = shot_cooldown
        self._shot_timer = random.uniform(0.8, shot_cooldown)
        self._telegraph_t = 0.0
        self._telegraphing = False
        self._dying = False
        self._death_timer = 0.0

    def _begin_death(self):
        if self._dying:
            return
        self._dying = True
        self.collider = None
        self.set_state('death', restart=True, frame_time=0.09)
        self._death_timer = len(self.frame_animations['death']) * self.frame_time
        player_hud.register_kill(15 if self.ranged else 10)
        roll = random.random()
        if roll < 0.30:
            AmmoPickup(position=(self.x, 0.5, self.z))
        elif roll < 0.55:
            HealthPickup(position=(self.x, 0.5, self.z))

    def update(self):
        if not self.enabled:
            return

        super().update()

        if self._dying:
            self._death_timer -= time.dt
            if self._death_timer <= 0:
                destroy(self)
            return

        dx = camera.world_position.x - self.world_position.x
        dz = camera.world_position.z - self.world_position.z
        dist = math.sqrt(dx * dx + dz * dz)
        if dist > 0.01:
            self.rotation_y = math.degrees(math.atan2(dx, dz))

        speed_dt = self.speed * time.dt
        if self.ranged:
            if dist > self.prefer_range + 1:
                self._try_move((dx / dist) * speed_dt, (dz / dist) * speed_dt)
                self.set_state('walk')
            elif dist < self.prefer_range - 1.5:
                self._try_move(-(dx / dist) * speed_dt * 0.6,
                               -(dz / dist) * speed_dt * 0.6)
                self.set_state('walk')
            else:
                self.set_state('idle')
            self._update_shooting(dist)
        else:
            if dist > self.attack_range:
                self._try_move((dx / dist) * speed_dt, (dz / dist) * speed_dt)
                self.set_state('walk')
            else:
                self._attack_timer -= time.dt
                if self._attack_timer <= 0:
                    self.set_state('attack', restart=True, frame_time=0.08)
                    player_hud.take_damage(self.damage, source=self)
                    self._attack_timer = self.attack_cooldown
                else:
                    self.set_state('idle')

        if self._hurt_timer > 0:
            self._hurt_timer -= time.dt
            if self.state != 'damage':
                self.set_state('damage', restart=True, frame_time=0.07)
            self.color = color.rgb(255, 60, 60) if int(self._hurt_timer * 30) % 2 \
                else color.white
            if self._hurt_timer <= 0 and not self._telegraphing:
                self.color = color.white
        elif not self._dying:
            self.color = color.white

    def _move_axis(self, axis, delta):
        if abs(delta) < 1e-5:
            return False
        direction = Vec3(0, 0, 0)
        if axis == 'x':
            direction.x = 1 if delta > 0 else -1
        else:
            direction.z = 1 if delta > 0 else -1
        origin = self.world_position + Vec3(0, 0.9, 0)
        hit = raycast(origin, direction,
                      distance=abs(delta) + 0.5,
                      ignore=(self, player))
        if not hit.hit:
            if axis == 'x':
                self.x += delta
            else:
                self.z += delta
            return True
        return False

    def _try_move(self, dx, dz):
        moved_x = self._move_axis('x', dx)
        moved_z = self._move_axis('z', dz)
        if moved_x or moved_z:
            return
        side = getattr(self, '_sidestep', random.choice((-1, 1)))
        length = math.sqrt(dx * dx + dz * dz)
        if length > 1e-5:
            self._move_axis('x', -dz * side * 0.8)
            self._move_axis('z',  dx * side * 0.8)
        if random.random() < 0.04:
            self._sidestep = -side
        else:
            self._sidestep = side

    def _aim_target(self):
        return player.world_position + Vec3(0, 0.55, 0)

    def _update_shooting(self, dist):
        if dist > self.fire_range:
            return
        if self._telegraphing:
            self._telegraph_t -= time.dt
            self.color = color.rgb(255, 240, 80)
            if self._telegraph_t <= 0:
                self._fire_at_player()
                self._telegraphing = False
                self.color = color.white
                self._shot_timer = self.shot_cooldown
            return
        self._shot_timer -= time.dt
        if self._shot_timer <= 0:
            origin = self.world_position + Vec3(0, 1.0, 0)
            dir_ = (self._aim_target() - origin).normalized()
            hit = raycast(origin, dir_, distance=self.fire_range, ignore=(self,))
            if hit.hit and hit.entity == player:
                self._telegraphing = True
                self._telegraph_t = 0.5
                self.set_state('attack', restart=True, frame_time=0.08)
            else:
                self._shot_timer = 0.4

    def _fire_at_player(self):
        origin = self.world_position + Vec3(0, 1.0, 0)
        dir_ = (self._aim_target() - origin).normalized()
        Bullet(position=origin + dir_ * 0.6,
               direction=dir_,
               damage=self.shot_damage,
               owner=self)

    def take_damage(self, amount):
        if self._dying:
            return
        self.hp -= amount
        self._hurt_timer = 0.25
        self.set_state('damage', restart=True, frame_time=0.07)
        if self.hp <= 0:
            self._begin_death()


# ──────────────────────────────────────────────────────────────────────────────
# Gun — raycast shooter with viewmodel, muzzle flash, ammo & reload
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Boss — fat-HP enemy with a 3-shot fan spread, big drops on death
# ──────────────────────────────────────────────────────────────────────────────
class Boss(Enemy):
    SPREAD_DEG = 12.0     # fan half-angle between spread shots
    SPREAD_N   = 3

    def __init__(self, **kwargs):
        kwargs.setdefault('sprite',        'Tank')
        kwargs.setdefault('hp',            90)
        kwargs.setdefault('speed',         1.7)
        kwargs.setdefault('damage',        28)
        kwargs.setdefault('shot_damage',   10)
        kwargs.setdefault('ranged',        True)
        kwargs.setdefault('fire_range',    36.0)
        kwargs.setdefault('prefer_range',  10.0)
        kwargs.setdefault('shot_cooldown', 1.1)
        kwargs.setdefault('scale',         (2.8, 4.0))
        kwargs.setdefault('origin_y',      -0.5)
        # Ensure boss uses Tank animations and larger scale
        animations = load_character_animations(kwargs.get('sprite', 'Tank'))
        kwargs['texture'] = animations['idle'][0]
        kwargs.setdefault('scale', (3.6, 5.0))
        super().__init__(**kwargs)
        self.attack_range = 2.6
        # Eye level is higher — boss is 4 m tall
        self._fire_height = 2.4
        # Announce ourselves to the HUD
        player_hud.update_boss_hp(self.hp, self.max_hp)

    def _aim_target(self):
        return player.world_position + Vec3(0, 0.55, 0)

    def _fire_at_player(self):
        origin = self.world_position + Vec3(0, self._fire_height, 0)
        base   = (self._aim_target() - origin).normalized()
        # Build a fan of bullets in the XZ plane around `base`
        for i in range(self.SPREAD_N):
            t = (i / (self.SPREAD_N - 1)) - 0.5      # -0.5 .. +0.5
            ang = math.radians(self.SPREAD_DEG * t * 2)
            cos_, sin_ = math.cos(ang), math.sin(ang)
            d = Vec3(
                base.x * cos_ - base.z * sin_,
                base.y,
                base.x * sin_ + base.z * cos_,
            ).normalized()
            Bullet(position=origin + d * 1.2,
                   direction=d,
                   damage=self.shot_damage,
                   owner=self)

    def take_damage(self, amount):
        self.hp -= amount
        self._hurt_timer = 0.25
        self.set_state('damage', restart=True, frame_time=0.07)
        player_hud.update_boss_hp(self.hp, self.max_hp)
        if self.hp <= 0:
            self._begin_boss_death()

    def _begin_boss_death(self):
        if self._dying:
            return
        self._dying = True
        self.collider = None
        self.set_state('death', restart=True, frame_time=0.09)
        self._death_timer = len(self.frame_animations['death']) * self.frame_time
        player_hud.register_kill(300)
        player_hud.update_boss_hp(0, self.max_hp, hide=True)
        # Lavish drops in a ring around the boss corpse
        for dx, dz in [(-1.5, 0), (1.5, 0), (0, -1.5), (0, 1.5)]:
            HealthPickup(position=(self.x + dx, 0.6, self.z + dz))
        for dx, dz in [(-1.2, 1.2), (1.2, 1.2), (-1.2, -1.2), (1.2, -1.2)]:
            AmmoPickup(position=(self.x + dx, 0.6, self.z + dz))
        message_log.show('★ BOSS DEFEATED ★', 5)


# ──────────────────────────────────────────────────────────────────────────────
# Charger — fast melee enemy that periodically bursts into a brief rush.
# ──────────────────────────────────────────────────────────────────────────────
class Charger(Enemy):
    def __init__(self, **kwargs):
        # Use an existing character sprite (Slayer) for the Charger
        kwargs.setdefault('sprite',     'Slayer')
        kwargs.setdefault('hp',         3)
        kwargs.setdefault('speed',      3.0)
        kwargs.setdefault('damage',     16)
        kwargs.setdefault('ranged',     False)
        kwargs.setdefault('color',      color.rgb(220, 110, 110))
        super().__init__(**kwargs)
        self._charge_cd       = random.uniform(2.0, 4.0)
        self._charge_remaining = 0.0
        self._base_speed      = self.speed

    def update(self):
        if not self.enabled:
            return
        # Initiate a charge periodically
        if self._charge_remaining > 0:
            self._charge_remaining -= time.dt
            if self._charge_remaining <= 0:
                self.speed = self._base_speed
                if self._hurt_timer <= 0:
                    self.color = color.rgb(220, 110, 110)
        else:
            self._charge_cd -= time.dt
            if self._charge_cd <= 0:
                self._charge_cd        = random.uniform(3.0, 5.5)
                self._charge_remaining = 0.9
                self.speed             = 8.5
                self.color             = color.rgb(255, 60, 60)
        super().update()


# ──────────────────────────────────────────────────────────────────────────────
# TNT barrel — shoot it (or hit it with another explosion) and it blows up,
# damaging everything within BLAST_RADIUS. Chain reactions are automatic.
# ──────────────────────────────────────────────────────────────────────────────
class TntBarrel(Entity):
    BLAST_RADIUS = 4.5
    BLAST_DAMAGE = 35
    SELF_DAMAGE  = 0.5      # multiplier on player damage (be unkind, but not lethal)

    def __init__(self, x, z):
        super().__init__(model='cube', position=(x, 0.7, z),
                         scale=(0.95, 1.4, 0.95),
                         texture=WALL_TEX, texture_scale=(1.5, 2.0),
                         color=color.rgb(210, 60, 40),
                         collider='box')
        # Yellow "TNT" cap so it reads clearly
        self._cap = Entity(parent=self, model='cube',
                           position=(0, 0.45, 0),
                           scale=(1.05, 0.18, 1.05),
                           color=color.rgb(245, 220, 80))
        self._exploded = False

    def take_damage(self, _amount):
        # Any damage detonates — the barrel doesn't have HP
        self.explode()

    def explode(self):
        if self._exploded:
            return
        self._exploded = True
        # Take a copy of the position BEFORE we destroy ourselves so the
        # raycast / iteration below never accesses a half-dead NodePath.
        pos = Vec3(self.world_position.x, self.world_position.y,
                   self.world_position.z)

        # ── Visual: growing sphere that fades out ───────────────────
        boom = Entity(model='sphere',
                      position=pos + Vec3(0, 0.6, 0),
                      scale=0.4, color=color.rgb(255, 200, 80))
        boom.animate_scale(3.5, duration=0.30)
        boom.animate_color(color.rgba(255, 120, 30, 0), duration=0.35)
        invoke(destroy, boom, delay=0.55)        # well past animation end

        # ── AoE: enemies + other TNT barrels in radius ──────────────
        # We iterate a snapshot and filter aggressively so a chained
        # blast on the same frame can't poke a destroyed Panda3D node.
        for e in list(scene.entities):
            if e is self:
                continue
            if not getattr(e, 'enabled', False):
                continue
            if not isinstance(e, (Enemy, TntBarrel)):
                continue
            try:
                d = (e.world_position - pos).length()
            except Exception:
                continue
            if d >= self.BLAST_RADIUS:
                continue
            falloff = 1.0 - d / self.BLAST_RADIUS
            if isinstance(e, Enemy):
                e.take_damage(int(self.BLAST_DAMAGE * falloff))
            elif isinstance(e, TntBarrel) and not e._exploded:
                invoke(e.explode, delay=0.10)

        # ── Player damage (reduced) ─────────────────────────────────
        try:
            d = (player.world_position - pos).length()
            if d < self.BLAST_RADIUS:
                falloff = 1.0 - d / self.BLAST_RADIUS
                player_hud.take_damage(
                    int(self.BLAST_DAMAGE * self.SELF_DAMAGE * falloff),
                    source=self)
        except Exception:
            pass

        destroy(self)


# ──────────────────────────────────────────────────────────────────────────────
# Bullet — slow-enough-to-dodge projectile fired by ranged enemies
# ──────────────────────────────────────────────────────────────────────────────
class Bullet(Entity):
    SPEED    = 22.0
    LIFETIME = 1.8

    def __init__(self, position, direction, damage=5, owner=None):
        super().__init__(
            model='sphere',
            scale=0.18,
            color=color.rgb(255, 240, 90),     # bright → dense dither dot
            position=position,
        )
        self.direction = Vec3(direction).normalized()
        self.damage    = damage
        self.owner     = owner               # so it can't immediately re-hit shooter
        self._t        = 0.0

    HIT_RADIUS = 0.7    # proximity-to-player sphere check

    def update(self):
        if not self.enabled:
            return
        step_len = self.SPEED * time.dt

        # Hit 1 — close enough to the player's body to count as a hit?
        try:
            body = player.world_position + Vec3(0, 0.55, 0)
            close = (self.world_position - body).length() < self.HIT_RADIUS
        except Exception:
            destroy(self)
            return
        if close:
            # If the owner enemy died while the bullet was in flight, drop
            # the reference — `source=` is only used for the HUD arrow.
            src = self.owner if (self.owner is not None
                                 and getattr(self.owner, 'enabled', False)) \
                              else self
            player_hud.take_damage(self.damage, source=src)
            destroy(self)
            return

        # Hit 2 — walls / cover / props block the bullet. Ignore the player
        # here (handled above) AND the firing enemy so it can't self-hit.
        # IMPORTANT: never pass a destroyed entity to `ignore=`; Panda3D
        # asserts when its NodePath is gone.
        ignore = [self, player]
        if self.owner is not None and getattr(self.owner, 'enabled', False):
            ignore.append(self.owner)
        try:
            hit = raycast(self.world_position, self.direction,
                          distance=step_len + 0.05, ignore=tuple(ignore))
        except Exception:
            destroy(self)
            return
        if hit.hit:
            destroy(self)
            return

        self.position += self.direction * step_len
        self._t += time.dt
        if self._t > self.LIFETIME:
            destroy(self)


# ──────────────────────────────────────────────────────────────────────────────
# Auto-pickup drops — float toward the player when close and apply on contact
# ──────────────────────────────────────────────────────────────────────────────
class _AutoPickup(Entity):
    PICKUP_RADIUS = 1.5
    BOB_SPEED     = 4.0
    BOB_AMOUNT    = 0.08

    def __init__(self, sprite, **kwargs):
        kwargs['model']        = 'quad'
        kwargs['texture']      = SPRITES[sprite]
        kwargs['double_sided'] = True
        kwargs.setdefault('scale', (0.55, 0.55))
        super().__init__(**kwargs)
        self._base_y = self.y
        self._t      = random.random() * 3

    def update(self):
        if not self.enabled:
            return
        # Bob + face camera
        self._t += time.dt
        self.y = self._base_y + math.sin(self._t * self.BOB_SPEED) * self.BOB_AMOUNT
        dx = camera.world_position.x - self.world_position.x
        dz = camera.world_position.z - self.world_position.z
        self.rotation_y = math.degrees(math.atan2(dx, dz))
        # Contact check
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < self.PICKUP_RADIUS:
            self.apply()
            destroy(self)

    def apply(self):
        pass    # subclasses override


class HealthPickup(_AutoPickup):
    AMOUNT = 25
    def __init__(self, **kwargs):
        super().__init__('health_pickup', **kwargs)
    def apply(self):
        player_hud.heal(self.AMOUNT)
        message_log.show(f'+{self.AMOUNT} HP', 1.0)


class AmmoPickup(_AutoPickup):
    AMOUNT = 8
    def __init__(self, **kwargs):
        super().__init__('ammo_pickup', **kwargs)
    def apply(self):
        gun.ammo = min(gun.max_ammo, gun.ammo + self.AMOUNT)
        player_hud.update_ammo(gun.ammo, gun.max_ammo)
        message_log.show(f'+{self.AMOUNT} Ammo', 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Wave manager — spawns progressive waves of outlaws at the far end of street
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# TimeOfDay — drives a global light tint through the dither shader uniform
# Each new wave advances one phase (noon → afternoon → dusk → twilight → night),
# so by wave 5 (the boss) you're fighting at night.  After the shop, it resets.
# ──────────────────────────────────────────────────────────────────────────────
class TimeOfDay:
    """Drives a two-colour dither palette through shader uniforms.
    The dither shader emits ONE of (fg_color, bg_color) per pixel, so the
    sky and the lighting now use real colour, not just brightness."""

    # (label,    sky / window.color,        fg_color (lit),         bg_color (shadow))
    PHASES = [
        ('HIGH NOON',   color.rgb(220, 215, 195),
            LVecBase3f(1.00, 1.00, 1.00), LVecBase3f(0.00, 0.00, 0.00)),
        ('AFTERNOON',   color.rgb(240, 200, 140),
            LVecBase3f(1.00, 0.95, 0.78), LVecBase3f(0.18, 0.10, 0.05)),
        ('DUSK',        color.rgb(235, 130,  70),
            LVecBase3f(1.00, 0.68, 0.32), LVecBase3f(0.32, 0.08, 0.04)),
        ('TWILIGHT',    color.rgb(100,  80, 140),
            LVecBase3f(0.75, 0.62, 0.96), LVecBase3f(0.10, 0.05, 0.20)),
        ('NIGHT',       color.rgb( 22,  30,  60),
            LVecBase3f(0.55, 0.78, 1.00), LVecBase3f(0.02, 0.04, 0.14)),
    ]

    def __init__(self):
        self.phase = 0
        self.apply()

    def set_for_wave(self, wave):
        # Cycle 0..4 every 5 waves; boss waves (5, 10, …) land on NIGHT.
        self.phase = (max(1, wave) - 1) % len(self.PHASES)
        self.apply(announce=True)

    def reset(self):
        self.phase = 0
        self.apply()

    def apply(self, announce=False):
        label, sky, _fg, _bg = self.PHASES[self.phase]
        window.color = sky
        self.reapply()
        if announce:
            message_log.show(f'☀ {label}', 2.5)

    def reapply(self):
        """Push the current fg/bg palette to the shader. Safe to call before
        the post-processing quad has been created — does nothing in that case."""
        _, _, fg, bg = self.PHASES[self.phase]
        q = globals().get('quad')
        if q is not None:
            q.setShaderInput('fg_color', fg)
            q.setShaderInput('bg_color', bg)

    @property
    def label(self):
        return self.PHASES[self.phase][0]


# ──────────────────────────────────────────────────────────────────────────────
# Shop — opens automatically every 5th cleared wave. Spend SCORE on upgrades.
# ──────────────────────────────────────────────────────────────────────────────
class Shop(Entity):
    def __init__(self):
        super().__init__()
        self.open = False

        # Items: (key, label, cost, apply_fn, info)
        self.items = [
            ('1', 'Max Ammo  +6',        50,  self._buy_max_ammo),
            ('2', 'Fire Rate  +20%',     80,  self._buy_fire_rate),
            ('3', 'Damage  +1',         140,  self._buy_damage),
            ('4', 'Max HP  +25',        100,  self._buy_max_hp),
            ('5', 'Reload Speed  +25%',  60,  self._buy_reload),
            ('6', 'Full Heal',           40,  self._buy_heal),
        ]

        # UI panel
        self.bg     = Entity(parent=camera.ui, model='quad',
                             scale=(1.0, 0.85), color=color.rgba(10, 10, 10, 230),
                             z=0.05, enabled=False)
        self.title  = Text('★ TRADING POST ★', position=(0, 0.32),
                           origin=(0, 0), scale=2.2,
                           color=color.rgb(240, 200, 90),
                           parent=camera.ui, enabled=False)
        self.subtitle = Text('Spend score between waves', position=(0, 0.25),
                             origin=(0, 0), scale=1.1,
                             color=color.white, parent=camera.ui, enabled=False)
        self.score_lbl = Text('', position=(0, 0.18), origin=(0, 0),
                              scale=1.4, color=color.rgb(255, 230, 100),
                              parent=camera.ui, enabled=False)

        # One Text per item
        self.item_texts = []
        for i, (key, label, cost, _) in enumerate(self.items):
            t = Text('', position=(-0.32, 0.08 - i * 0.07),
                     origin=(-0.5, 0.5), scale=1.2,
                     color=color.white, parent=camera.ui, enabled=False)
            self.item_texts.append(t)

        self.footer = Text('[ENTER]  Continue to next wave',
                           position=(0, -0.36), origin=(0, 0), scale=1.3,
                           color=color.rgb(200, 200, 200),
                           parent=camera.ui, enabled=False)

    # ── Open / close ─────────────────────────────────────────────
    def open_shop(self):
        self.open = True
        for e in (self.bg, self.title, self.subtitle, self.score_lbl,
                  self.footer, *self.item_texts):
            e.enabled = True
        # Freeze player & gun while in the shop
        self.player_speed_save = player.speed
        player.speed = 0
        self._refresh()

    def close_shop(self):
        self.open = False
        for e in (self.bg, self.title, self.subtitle, self.score_lbl,
                  self.footer, *self.item_texts):
            e.enabled = False
        player.speed = getattr(self, 'player_speed_save', 6)
        # Let the wave manager start the next wave now
        wave_manager.start_next_wave()

    # ── Input ────────────────────────────────────────────────────
    def input(self, key):
        if not self.open:
            return
        if key in ('enter', 'return'):
            self.close_shop()
            return
        for shortcut, label, cost, fn in self.items:
            if key == shortcut:
                self._try_buy(label, cost, fn)
                return

    # ── Buying ───────────────────────────────────────────────────
    def _try_buy(self, label, cost, fn):
        if player_hud.score < cost:
            message_log.show(f'Not enough score for "{label}"', 1.5)
            return
        player_hud.score -= cost
        player_hud.score_text.text = f'SCORE {player_hud.score}'
        fn()
        message_log.show(f'Bought: {label}', 1.5)
        self._refresh()

    def _refresh(self):
        self.score_lbl.text = f'SCORE  {player_hud.score}'
        for t, (key, label, cost, _) in zip(self.item_texts, self.items):
            affordable = player_hud.score >= cost
            t.text  = f'[{key}]   {label:<22s}   {cost}'
            t.color = color.white if affordable else color.rgb(120, 120, 120)

    # ── Upgrade callbacks ────────────────────────────────────────
    def _buy_max_ammo(self):
        gun.max_ammo += 6
        gun.ammo      = gun.max_ammo
        player_hud.update_ammo(gun.ammo, gun.max_ammo)

    def _buy_fire_rate(self):
        gun.fire_rate = max(0.04, gun.fire_rate * 0.80)

    def _buy_damage(self):
        gun.damage += 1

    def _buy_max_hp(self):
        player_hud.max_hp += 25
        player_hud.hp      = min(player_hud.max_hp, player_hud.hp + 25)
        player_hud.hp_text.text = player_hud._hp_str()

    def _buy_reload(self):
        gun.reload_time = max(0.3, gun.reload_time * 0.75)

    def _buy_heal(self):
        player_hud.heal(player_hud.max_hp)


class WaveManager(Entity):
    # Spawn zones (x_min, x_max, z_min, z_max) — all far from the start
    SPAWN_AREAS = [
        (-18,  18,  26,  34),   # far north
        (-20, -16,  -8,  24),   # west flank
        ( 16,  20,  -8,  24),   # east flank
    ]
    INTERMISSION = 4.0           # seconds between waves

    def __init__(self):
        super().__init__()
        self.wave        = 0
        self.between     = True
        self._timer      = 1.5      # short delay before wave 1
        self.banner      = Text('', position=(0, 0.32), origin=(0, 0),
                                scale=2.6, color=color.yellow,
                                parent=camera.ui)
        self.subtitle    = Text('', position=(0, 0.22), origin=(0, 0),
                                scale=1.2, color=color.white,
                                parent=camera.ui)
        self.wave_label  = Text('WAVE 0', position=(0, 0.47), origin=(0, 1),
                                scale=1.3, color=color.white,
                                parent=camera.ui)

    def update(self):
        if player_hud.dead or shop.open:
            return
        if self.between:
            self._timer -= time.dt
            if self._timer <= 0:
                # Every 5th wave cleared → open the shop instead of going straight on
                if self.wave > 0 and self.wave % 5 == 0:
                    shop.open_shop()
                    return
                self.start_next_wave()
            return
        # Wave is running — when no enemies remain, start intermission
        remaining = sum(1 for e in scene.entities if isinstance(e, Enemy))
        if remaining == 0:
            self.between = True
            self._timer  = self.INTERMISSION
            self.banner.text   = f'WAVE {self.wave} CLEARED'
            if self.wave % 5 == 0:
                self.subtitle.text = 'Trading post opens…'
            else:
                self.subtitle.text = f'Next wave in {int(self.INTERMISSION)}s…'
            invoke(self._clear_banner, delay=self.INTERMISSION - 0.3)
            player_hud.add_score(50 + self.wave * 10)

    def start_next_wave(self):
        self.wave   += 1
        self.between = False
        self.wave_label.text = f'WAVE {self.wave}'

        # Advance the time-of-day light tint with the wave (boss waves = NIGHT)
        time_of_day.set_for_wave(self.wave)

        # ── BOSS WAVE at #5 ──────────────────────────────────────────
        if self.wave == 5:
            self.banner.text   = '★ BOSS WAVE ★'
            self.subtitle.text = 'The marshal has come for you.'
            invoke(self._clear_banner, delay=2.8)
            Boss(position=(0, 0, 30))
            # Two ranged minions on the flanks for support
            Enemy(position=(-10, 0, 26), hp=4, speed=1.7,
                  ranged=True, shot_damage=5)
            Enemy(position=( 10, 0, 26), hp=4, speed=1.7,
                  ranged=True, shot_damage=5)
            return

        # ── Normal wave ──────────────────────────────────────────────
        self.banner.text   = f'WAVE {self.wave}'
        self.subtitle.text = ''
        invoke(self._clear_banner, delay=1.5)

        # Scale up: more enemies, tougher stats, more ranged.
        # Bonus: post-boss waves (6+) get a meaningful difficulty jump.
        post_boss  = 1 if self.wave > 5 else 0
        n          = 4 + self.wave * 2 + post_boss * 2
        n_ranged   = min(self.wave, n // 2)
        n_charger  = max(0, self.wave - 1) if self.wave >= 2 else 0
        n_charger  = min(n_charger, n // 3)
        base_hp    = 2 + self.wave + post_boss
        base_speed = 1.5 + min(self.wave * 0.1, 1.2)
        for i in range(n):
            area = random.choice(self.SPAWN_AREAS)
            x = random.uniform(area[0], area[1])
            z = random.uniform(area[2], area[3])
            # Distribute roles: rangedest first, then chargers, then melee
            if i < n_ranged:
                Enemy(position=(x, 0, z),
                      hp=base_hp + 1,
                      speed=base_speed - 0.4,
                      shot_damage=4 + self.wave,
                      ranged=True)
            elif i < n_ranged + n_charger:
                Charger(position=(x, 0, z),
                        hp=base_hp,
                        damage=14 + self.wave)
            else:
                Enemy(position=(x, 0, z),
                      hp=base_hp,
                      speed=base_speed,
                      ranged=False)

    def _clear_banner(self):
        self.banner.text   = ''
        self.subtitle.text = ''

    def reset(self):
        # Called by the player on restart
        self.wave    = 0
        self.between = True
        self._timer  = 1.5
        self.wave_label.text = 'WAVE 0'
        self.banner.text     = ''
        self.subtitle.text   = ''
        time_of_day.reset()
        # Also close the shop if it happened to be open
        if shop.open:
            shop.close_shop()


class Gun(Entity):
    def __init__(self, player_entity):
        super().__init__()
        self.player        = player_entity
        self.damage        = 1
        self.fire_rate     = 0.25         # seconds between shots
        self.range         = 60
        self.max_ammo      = 12
        self.ammo          = self.max_ammo
        self.reload_time   = 1.4
        self._fire_timer   = 0.0
        self._reload_timer = 0.0
        self._reloading    = False

        # Viewmodel — cropped HUD sprite, kept out of the dither pass so it
        # reads like the source image instead of a tiny world-space quad.
        self._rest = Vec3(0.70, -0.43, 0)
        self.viewmodel = AnimatedSprite(
            parent=camera.ui,
            frames=load_detected_sprite_frames('DEAG Spritesheet'),
            frame_time=0.08,
            loop=False,
            scale=(0.57, 0.57),
            position=self._rest,
        )

        # Muzzle flash — sibling sprite, only enabled for one frame after firing.
        self.muzzle = Entity(
            parent=camera.ui,
            model='quad',
            texture=load_cropped_sprite('muzzle_flash'),
            double_sided=True,
            scale=(0.16, 0.16),
            position=self._rest + Vec3(0.10, 0.12, 0),
            enabled=False,
        )

    def update(self):
        if self._fire_timer > 0:
            self._fire_timer -= time.dt
        if self._reloading:
            self._reload_timer -= time.dt
            if self._reload_timer <= 0:
                self.ammo       = self.max_ammo
                self._reloading = False
                player_hud.update_ammo(self.ammo, self.max_ammo)

    def input(self, key):
        if player_hud.dead:
            return
        if key == 'left mouse down':
            self.shoot()
        elif key == 'r':
            self.reload()

    def shoot(self):
        if self._reloading or self._fire_timer > 0:
            return
        if self.ammo <= 0:
            message_log.show('Click! Out of ammo — press [R] to reload.', 1.0)
            return

        self.ammo        -= 1
        self._fire_timer  = self.fire_rate
        player_hud.update_ammo(self.ammo, self.max_ammo)

        self.viewmodel.play_once(frame_time=0.04)

        # Muzzle flash for one frame's-worth of time
        self.muzzle.enabled = True
        invoke(setattr, self.muzzle, 'enabled', False, delay=0.05)

        # Quick recoil kick
        self.viewmodel.animate_position(
            self._rest + Vec3(0, -0.04, 0.05), duration=0.04, curve=curve.out_quad)
        self.viewmodel.animate_position(
            self._rest, duration=0.12, delay=0.05, curve=curve.in_out_quad)

        # Raycast hit-scan
        hit = raycast(camera.world_position, camera.forward,
                      distance=self.range, ignore=(self.player, self.viewmodel,
                                                   self.muzzle))
        if hit.hit:
            # ── Bullet impact spark at the hit point (any surface) ──
            _spawn_spark(hit.world_point)

            # ── Enemy hit: check for headshot (upper third of body) ─
            if isinstance(hit.entity, Enemy):
                enemy   = hit.entity
                top_y   = enemy.world_position.y + enemy.scale_y * 0.65
                is_head = hit.world_point.y >= top_y
                dmg     = self.damage * (2 if is_head else 1)
                enemy.take_damage(dmg)
                player_hud.flash_hit_marker()
                if is_head:
                    player_hud.show_headshot()
            # ── TNT barrel: detonate it ────────────────────────────
            elif isinstance(hit.entity, TntBarrel):
                hit.entity.take_damage(1)

    def reload(self):
        if self._reloading or self.ammo >= self.max_ammo:
            return
        self._reloading    = True
        self._reload_timer = self.reload_time
        message_log.show('Reloading…', self.reload_time)


# ──────────────────────────────────────────────────────────────────────────────
# Player HUD — health bar, ammo readout, score, hit marker, damage flash
# ──────────────────────────────────────────────────────────────────────────────
class PlayerHUD(Entity):
    LOW_HP_PCT       = 0.30
    ARROW_DURATION   = 1.5
    SPAWN_POS        = Vec3(0, 2, -32)

    def __init__(self, player_entity):
        super().__init__()
        self.player     = player_entity
        self.max_hp     = 100
        self.hp         = self.max_hp
        self.score      = 0
        self.dead       = False
        self._death_ui  = []   # entities to clean up on restart

        # Health (bottom-left)
        self.hp_text = Text(self._hp_str(), position=(-0.86, -0.42),
                            origin=(-0.5, 0.5), color=color.white,
                            scale=1.4, parent=camera.ui)
        # Ammo (bottom-right)
        self.ammo_text = Text('12 / 12', position=(0.86, -0.42),
                              origin=(0.5, 0.5), color=color.white,
                              scale=1.4, parent=camera.ui)
        # Score (top-right)
        self.score_text = Text('SCORE 0', position=(0.86, 0.40),
                               origin=(0.5, 1), color=color.white,
                               scale=1.2, parent=camera.ui)

        # Full-screen red flash on damage (alpha tween)
        self.flash = Entity(parent=camera.ui, model='quad', scale=(2, 1.2),
                            color=color.rgba(255, 30, 30, 0), z=0.3)

        # Persistent low-HP vignette (pulses when HP is critical)
        self.lowhp = Entity(parent=camera.ui, model='quad', scale=(2, 1.2),
                            color=color.rgba(255, 0, 0, 0), z=0.4)

        # Hit marker for tagging an enemy
        self.hit_mark = Text('X', position=(0, 0), origin=(0, 0),
                             color=color.rgba(255, 220, 0, 0),
                             scale=2.5, parent=camera.ui)

        # Boss HP bar (top centre, hidden until a Boss is alive)
        self.boss_hp_text = Text('', position=(0, 0.42), origin=(0, 0),
                                 scale=1.4, color=color.rgb(255, 90, 90),
                                 parent=camera.ui, enabled=False)

        # ── Combo multiplier (chain kills within 3 s) ───────────────
        self.combo         = 1
        self.combo_timer   = 0.0
        self.combo_max     = 5
        self.combo_timeout = 3.0
        self.combo_text    = Text('', position=(0, 0.32), origin=(0, 0),
                                  scale=2.4, color=color.rgb(255, 215, 60),
                                  parent=camera.ui)

        # ── Dash cooldown readout (right side, above ammo) ──────────
        self.dash_text = Text('DASH READY', position=(0.86, -0.34),
                              origin=(0.5, 0.5), scale=1.0,
                              color=color.rgb(120, 230, 120),
                              parent=camera.ui)

        # Direction-of-damage arrows (up to N at once)
        self._arrows     = []
        self._max_arrows = 6

    # ── Text helpers ─────────────────────────────────────────────
    def _hp_str(self):
        bar_len = 20
        filled  = int(bar_len * max(0, self.hp) / self.max_hp)
        bar     = '#' * filled + '-' * (bar_len - filled)
        return f'HP [{bar}] {max(0, self.hp)}'

    # ── Public API ───────────────────────────────────────────────
    def take_damage(self, amount, source=None):
        if self.dead:
            return
        # Dash i-frames: ignore damage during the brief invincibility window
        if time.time() < self.player.invincible_until:
            return
        self.hp -= amount
        self.hp_text.text = self._hp_str()
        self.flash.color  = color.rgba(255, 30, 30, 160)
        self.flash.animate_color(color.rgba(255, 30, 30, 0), duration=0.35)
        if source is not None:
            self._spawn_arrow_from(source.world_position)
        if self.hp <= 0:
            self.die()

    def heal(self, amount):
        if self.dead:
            return
        self.hp = min(self.max_hp, self.hp + amount)
        self.hp_text.text = self._hp_str()

    def update_ammo(self, ammo, max_ammo):
        self.ammo_text.text = f'{ammo} / {max_ammo}'

    def add_score(self, points):
        self.score += points
        self.score_text.text = f'SCORE {self.score}'

    def flash_hit_marker(self):
        self.hit_mark.color = color.rgba(255, 220, 0, 255)
        self.hit_mark.animate_color(color.rgba(255, 220, 0, 0), duration=0.2)

    def update_boss_hp(self, hp, max_hp, hide=False):
        if hide or hp <= 0:
            self.boss_hp_text.enabled = False
            return
        bar_len = 30
        filled  = int(bar_len * max(0, hp) / max_hp)
        bar     = '#' * filled + '-' * (bar_len - filled)
        self.boss_hp_text.text    = f'★ BOSS ★ [{bar}] {max(0, hp)}'
        self.boss_hp_text.enabled = True

    # ── Combo + headshot helpers ────────────────────────────────────
    def register_kill(self, base_score):
        """Called by Enemy.take_damage when the enemy dies. Builds combo and
        awards score multiplied by it."""
        if self.combo_timer > 0:
            self.combo = min(self.combo_max, self.combo + 1)
        else:
            self.combo = 1
        self.combo_timer = self.combo_timeout
        self.add_score(base_score * self.combo)
        if self.combo > 1:
            self.combo_text.text = f'COMBO  ×{self.combo}'
        else:
            self.combo_text.text = ''

    def show_headshot(self):
        t = Text('HEADSHOT!', origin=(0, 0), position=(0, 0.06),
                 scale=2.2, color=color.rgb(255, 230, 60),
                 parent=camera.ui)
        t.animate_scale(0.6, duration=0.45)
        t.animate_color(color.rgba(255, 230, 60, 0), duration=0.5)
        invoke(destroy, t, delay=0.70)        # past both animate calls

    # ── Frame logic: low-HP pulse + arrow lifetime + combo + dash ─
    def update(self):
        # Low-HP vignette pulses red
        if not self.dead and self.hp < self.max_hp * self.LOW_HP_PCT:
            pulse = (math.sin(time.time() * 6) * 0.5 + 0.5) * 120
            self.lowhp.color = color.rgba(255, 0, 0, int(pulse))
        else:
            self.lowhp.color = color.rgba(255, 0, 0, 0)

        # Combo timer
        if self.combo_timer > 0:
            self.combo_timer -= time.dt
            if self.combo_timer <= 0:
                self.combo = 1
                self.combo_text.text = ''

        # Dash readiness readout
        cd = self.player._dash_cd
        if cd > 0:
            self.dash_text.text  = f'DASH  {cd:.1f}s'
            self.dash_text.color = color.rgb(120, 120, 120)
        else:
            self.dash_text.text  = 'DASH READY'
            self.dash_text.color = color.rgb(120, 230, 120)

        # Tick down arrows; remove expired
        for arrow in list(self._arrows):
            arrow['t'] -= time.dt
            if arrow['t'] <= 0:
                destroy(arrow['ent'])
                self._arrows.remove(arrow)
                continue
            # Keep the arrow pointing at where the attacker WAS in world space
            ax, az = arrow['wx'], arrow['wz']
            self._position_arrow(arrow['ent'],
                                 self._relative_angle(ax, az),
                                 alpha=int(255 * (arrow['t'] / self.ARROW_DURATION)))

    # ── Direction-of-damage indicators ───────────────────────────
    def _relative_angle(self, wx, wz):
        dx = wx - self.player.world_position.x
        dz = wz - self.player.world_position.z
        world_a = math.degrees(math.atan2(dx, dz))
        return ((world_a - self.player.rotation_y + 540) % 360) - 180  # -180..180

    def _position_arrow(self, ent, rel_angle_deg, alpha):
        rad    = math.radians(rel_angle_deg)
        radius = 0.18
        ent.x  = math.sin(rad) * radius
        ent.y  = math.cos(rad) * radius * 0.6   # squish for aspect
        ent.rotation_z = -rel_angle_deg
        ent.color = color.rgba(255, 60, 60, alpha)

    def _spawn_arrow_from(self, source_pos):
        # Drop oldest if at cap
        if len(self._arrows) >= self._max_arrows:
            destroy(self._arrows.pop(0)['ent'])
        ent = Text('▲', origin=(0, 0), scale=1.6, parent=camera.ui,
                   color=color.rgba(255, 60, 60, 255))
        self._arrows.append({
            'ent': ent,
            't':   self.ARROW_DURATION,
            'wx':  source_pos.x,
            'wz':  source_pos.z,
        })

    # ── Death + restart ──────────────────────────────────────────
    def die(self):
        self.dead = True
        self.player.speed = 0
        self._death_ui.append(
            Text('YOU DIED', origin=(0, 0), position=(0, 0.08), scale=5,
                 color=color.red, parent=camera.ui))
        self._death_ui.append(
            Text('press [ENTER] to restart', origin=(0, 0),
                 position=(0, -0.02), scale=1.5,
                 color=color.white, parent=camera.ui))

    def input(self, key):
        if self.dead and key in ('enter', 'return'):
            self.restart()

    def restart(self):
        # Tear down death overlay
        for e in self._death_ui:
            destroy(e)
        self._death_ui.clear()
        # Tear down damage arrows
        for arrow in self._arrows:
            destroy(arrow['ent'])
        self._arrows.clear()
        # Reset player state
        self.hp     = self.max_hp
        self.dead   = False
        self.player.speed    = 6
        self.player.position = self.SPAWN_POS
        self.player._vel     = Vec3(0, 0, 0)
        self.player._y_vel   = 0.0
        self.player.rotation_y          = 0
        self.player.camera_pivot.rotation_x = 0
        self.hp_text.text = self._hp_str()
        # Reset combo + dash
        self.combo       = 1
        self.combo_timer = 0
        self.combo_text.text = ''
        self.player._dash_cd    = 0
        self.player._dash_timer = 0
        # Reset ammo
        gun.ammo        = gun.max_ammo
        gun._reloading  = False
        gun._reload_timer = 0
        self.update_ammo(gun.ammo, gun.max_ammo)
        # Wipe all enemies + pickups + flying bullets so the next wave starts clean
        for e in list(scene.entities):
            if isinstance(e, (Enemy, HealthPickup, AmmoPickup, Bullet)):
                destroy(e)
        # Hide boss bar (Boss subclasses Enemy so it was destroyed above)
        self.boss_hp_text.enabled = False
        # Reset wave manager
        wave_manager.reset()
        message_log.show('Restarted — show them who runs this town.', 2.5)


# ══════════════════════════════════════════════════════════════════════════════
#                                  APP
# ══════════════════════════════════════════════════════════════════════════════
app = Ursina(title='Test Environment – 1-Bit Dither', vsync=True)
window.fullscreen = True                     # Enable fullscreen
window.color = color.rgb(140, 130, 110)      # dusty noon-sky beige — bright dither
base.render.setAntialias(AntialiasAttrib.MNone)

# ── Player ────────────────────────────────────────────────────────────────────
player = SmoothFPC(y=2, z=-32, origin_y=-0.5, speed=6, jump_height=1.8)
player.collider = 'box'        # explicit capsule-ish collider for the player
player.cursor.visible = False
# Face north up the street toward the enemies on spawn
player.rotation_y = 0

# ── HUD systems (created before any interactable that calls into them) ────────
inventory   = Inventory()
message_log = MessageLog()
interaction = InteractionSystem(player)

# Crosshair
Text('+', origin=(0, 0), scale=2, color=color.white, parent=camera.ui)

# Coordinates display
coords_text = Text('X: 0  Y: 0  Z: 0', position=(0.86, 0.40), origin=(1, 1),
                   scale=1.0, color=color.white, parent=camera.ui)

# Controls hint
Text('WASD Move · Shift Dash · LMB Shoot · R Reload · E Use · '
     'T Dither · Enter Restart · ESC Quit',
     position=(-0.86, -0.47), scale=0.80, color=color.white, parent=camera.ui)

# Player HUD + gun must come after textures so SPRITES exists — created later.


# ──────────────────────────────────────────────────────────────────────────────
# Hand-drawn-style textures
# ──────────────────────────────────────────────────────────────────────────────
# Procedurally generated PIL images that look hand-etched once the dither
# shader processes them. Drop your own art at  textures/floor.png  or
# textures/wall.png  to override either pattern — they're loaded if present.
#
# Tips for drawing your own:
#   - Use a small canvas (128×128 or 256×256). Nearest-neighbour upscales it.
#   - Pure black & white reads cleanest; the shader will dither greys anyway.
#   - Keep tilable: edges should wrap so seams don't show.
#
TEX_DIR = Path(__file__).parent / 'textures'
TEX_DIR.mkdir(exist_ok=True)


# Helpers for building boldly contrasted patterns that survive the dither pass.
# Key principle: HARD black + HARD white only.  Mid-greys get washed out, so
# we use pure 20/220-ish luma extremes and chunky shapes (>= 4 px features).
_BLK = (15, 15, 12)
_WHT = (235, 230, 215)


def _floor_pattern(size=128):
    """Dusty sand / dirt ground — pebbles, footprints, wagon tracks."""
    img = Image.new('RGB', (size, size), _WHT)
    d   = ImageDraw.Draw(img)
    rng = random.Random(11)

    # Dark dirt patches — chunky blotches
    for _ in range(30):
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        r      = rng.randint(4, 10)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_BLK)

    # Bright sand highlights
    for _ in range(35):
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        r      = rng.randint(2, 4)
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=_WHT)

    # Scattered pebbles
    for _ in range(25):
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        r      = rng.randint(1, 2)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_BLK)

    # Wandering wagon tracks (long lines)
    for _ in range(4):
        y = rng.randint(8, size - 8)
        prev = (0, y)
        for x in range(8, size, 12):
            ny = y + rng.randint(-4, 4)
            d.line([prev, (x, ny)], fill=_BLK, width=2)
            prev = (x, ny)
    return img


def _wall_pattern(size=128):
    """Vertical wood planks: dark seams, horizontal grain, knots."""
    img = Image.new('RGB', (size, size), _WHT)
    d   = ImageDraw.Draw(img)
    rng = random.Random(23)

    planks   = 4
    plank_w  = size // planks
    seam_w   = 5

    # Vertical plank seams
    for col in range(planks + 1):
        x = col * plank_w
        d.rectangle([x, 0, x + seam_w, size], fill=_BLK)

    # Per-plank grain streaks + occasional knot
    for col in range(planks):
        x0 = col * plank_w + seam_w + 1
        x1 = (col + 1) * plank_w - 1
        if x1 <= x0:
            continue

        # Horizontal grain streaks
        for _ in range(14):
            y      = rng.randint(0, size)
            length = rng.randint(x1 - x0 - 8, x1 - x0)
            sx     = rng.randint(x0, max(x0, x1 - length))
            d.line([(sx, y), (sx + length, y + rng.randint(-2, 2))],
                   fill=_BLK, width=1)

        # Big knot
        if rng.random() < 0.55:
            kx = (x0 + x1) // 2
            ky = rng.randint(15, size - 15)
            d.ellipse([kx - 6, ky - 5, kx + 6, ky + 5], fill=_BLK)
            d.ellipse([kx - 2, ky - 1, kx + 2, ky + 1], fill=_WHT)

        # Bright highlights / weathered spots
        for _ in range(4):
            cx = rng.randint(x0, x1)
            cy = rng.randint(0, size)
            r  = rng.randint(1, 3)
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=_WHT)
    return img


def _stone_pattern(size=128):
    """Rough-hewn stone for pedestals & capstones: blobs and chips, no grid."""
    img = Image.new('RGB', (size, size), _WHT)
    d   = ImageDraw.Draw(img)
    rng = random.Random(31)
    # Big dark blotches
    for _ in range(28):
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        r      = rng.randint(4, 11)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_BLK)
    # Small bright pits
    for _ in range(40):
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        r      = rng.randint(1, 3)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_WHT)
    # A few hard cracks
    for _ in range(5):
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        for _ in range(7):
            nx = cx + rng.randint(-14, 14)
            ny = cy + rng.randint(-14, 14)
            d.line([(cx, cy), (nx, ny)], fill=_BLK, width=2)
            cx, cy = nx, ny
    return img


def _iron_pattern(size=64):
    """Dark iron — vertical streaks of rust + rivets."""
    img = Image.new('RGB', (size, size), _BLK)
    d   = ImageDraw.Draw(img)
    # Vertical bright streaks (wear / highlights)
    rng = random.Random(41)
    for _ in range(7):
        x = rng.randint(2, size - 3)
        d.line([(x, 0), (x + rng.randint(-2, 2), size)], fill=_WHT, width=1)
    # Rivets at the corners and centre
    for cx, cy in [(6, 6), (size - 6, 6), (6, size - 6), (size - 6, size - 6),
                   (size // 2, size // 2)]:
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=_WHT)
        d.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=_BLK)
    return img


def _wood_pattern(size=128):
    """Wood plank grain — horizontal dark streaks, knot here and there."""
    img = Image.new('RGB', (size, size), _WHT)
    d   = ImageDraw.Draw(img)
    plank_h = size // 4
    # Plank seams
    for row in range(5):
        y = row * plank_h
        d.rectangle([0, y, size, y + 3], fill=_BLK)
    # Grain streaks
    rng = random.Random(53)
    for row in range(4):
        y0 = row * plank_h + 4
        y1 = (row + 1) * plank_h - 4
        for _ in range(12):
            y = rng.randint(y0, y1)
            x = rng.randint(0, size)
            length = rng.randint(20, 60)
            d.line([(x, y), (x + length, y)], fill=_BLK, width=1)
        # Occasional knot
        if rng.random() < 0.5:
            kx = rng.randint(15, size - 15)
            ky = (y0 + y1) // 2
            d.ellipse([kx - 5, ky - 4, kx + 5, ky + 4], fill=_BLK)
            d.ellipse([kx - 2, ky - 1, kx + 2, ky + 1], fill=_WHT)
    return img


def _terra_pattern(size=64):
    """Terracotta — dotted clay surface."""
    img = Image.new('RGB', (size, size), _WHT)
    d   = ImageDraw.Draw(img)
    rng = random.Random(67)
    for _ in range(70):
        cx, cy = rng.randint(0, size), rng.randint(0, size)
        r      = rng.randint(1, 3)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_BLK)
    # Horizontal banding (potter's wheel)
    for y in range(0, size, 8):
        d.line([(0, y), (size, y)], fill=_BLK, width=1)
    return img


def _make_texture(filename, generator):
    """Load textures/<filename> if it exists, else build it procedurally.
    The cache is on disk so you can edit the PNG and re-run."""
    path = TEX_DIR / filename
    if path.exists():
        pil_img = Image.open(path).convert('RGB')
    else:
        pil_img = generator()
        try:
            pil_img.save(path)
        except OSError:
            pass
    tex = Texture(pil_img)
    tex._texture.setMagfilter(SamplerState.FT_nearest)
    tex._texture.setMinfilter(SamplerState.FT_nearest)
    return tex


# Bumped to _v3 for the wild-west sand-and-plank patterns.
FLOOR_TEX = _make_texture('floor_v3.png', _floor_pattern)
WALL_TEX  = _make_texture('wall_v3.png',  _wall_pattern)
STONE_TEX = _make_texture('stone_v2.png', _stone_pattern)
IRON_TEX  = _make_texture('iron_v2.png',  _iron_pattern)
WOOD_TEX  = _make_texture('wood_v2.png',  _wood_pattern)
TERRA_TEX = _make_texture('terra_v2.png', _terra_pattern)


# ──────────────────────────────────────────────────────────────────────────────
# Billboard-sprite textures — one PNG per object type in  textures/sprites/
# ──────────────────────────────────────────────────────────────────────────────
# Each named sprite gets its own PNG file. The first run generates a labeled
# placeholder that shows the sprite name; replace any file with your own art
# (transparent PNG, ideally tall portrait for plants/trees, square for pickups).
#
SPRITES_DIR = TEX_DIR / 'sprites'
SPRITES_DIR.mkdir(exist_ok=True)

try:
    from PIL import ImageFont
    try:
        _FONT_LG = ImageFont.truetype('arialbd.ttf', 22)
    except (OSError, IOError):
        try:
            _FONT_LG = ImageFont.truetype('DejaVuSans-Bold.ttf', 22)
        except (OSError, IOError):
            _FONT_LG = ImageFont.load_default()
except ImportError:
    _FONT_LG = None


def _make_sprite_template(name, size=(128, 192)):
    """Bordered placeholder sprite with the name printed across it."""
    w, h = size
    img  = Image.new('RGBA', size, (0, 0, 0, 0))     # fully transparent
    d    = ImageDraw.Draw(img)

    # Opaque dark interior + thick white frame (silhouette of the placeholder)
    d.rectangle([2, 2, w - 3, h - 3], fill=(25, 25, 25, 235),
                outline=(245, 245, 245, 255), width=4)

    # Subtle diagonal cross (so the frame reads as "placeholder")
    d.line([(6, 6), (w - 7, h - 7)], fill=(160, 160, 160, 180), width=1)
    d.line([(w - 7, 6), (6, h - 7)], fill=(160, 160, 160, 180), width=1)

    # Name label — white box with black text in the middle
    text = name.upper()
    if _FONT_LG is not None:
        bbox = d.textbbox((0, 0), text, font=_FONT_LG)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = (w - tw) // 2, (h - th) // 2 - 4
        pad = 6
        d.rectangle([tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
                    fill=(255, 255, 255, 255))
        d.text((tx, ty), text, fill=(0, 0, 0, 255), font=_FONT_LG)
    else:
        # Last-resort fallback if PIL has no font support at all
        d.text(((w - 6 * len(text)) // 2, h // 2 - 4),
               text, fill=(0, 0, 0, 255))

    return img


def _make_sprite(name, size=(128, 192)):
    """Load textures/sprites/<name>.png, generating a labeled template if absent."""
    path = SPRITES_DIR / f'{name}.png'
    if path.exists():
        pil_img = Image.open(path).convert('RGBA')
    else:
        pil_img = _make_sprite_template(name, size)
        try:
            pil_img.save(path)
        except OSError:
            pass
    tex = Texture(pil_img)
    tex._texture.setMagfilter(SamplerState.FT_nearest)
    tex._texture.setMinfilter(SamplerState.FT_nearest)
    return tex


# Each entry:  (name, (texture_w_px, texture_h_px))
# Files end up at  D:\game\textures\sprites\<name>.png  — edit any to replace.
_SPRITE_SPECS = {
    # ── Wild-west town props (replace any PNG to reskin) ────────────────
    'saloon':         (256, 256),     # tall building facade
    'sheriff':        (256, 256),
    'bank':           (256, 256),
    'general_store':  (256, 256),
    'cactus':         (128, 192),
    'tumbleweed':     ( 96,  96),
    'hitching_post':  (128,  96),
    'water_trough':   (160,  80),
    'wagon':          (192, 128),
    'wanted_poster':  ( 96, 128),

    # ── Interactive items (still picked up / claimed) ──────────────────
    'coin':          ( 64,  64),
    'brass_key':     ( 96,  64),
    'trophy':        ( 96, 160),

    # ── Combat sprites ─────────────────────────────────────────────────
    'enemy':         (128, 192),
    'enemy_ranged':  (128, 192),   # gunslinger variant
    'charger':       (128, 192),   # red-trim rusher
    'boss':          (256, 320),   # wave-5 marshal
    'gun':           (192, 128),
    'muzzle_flash':  ( 96,  96),
    'hit_marker':    ( 48,  48),
    'health_pickup': ( 64,  64),
    'ammo_pickup':   ( 64,  64),
    'plant':         (128, 192),
}
SPRITES = {n: _make_sprite(n, sz) for n, sz in _SPRITE_SPECS.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Combat + meta systems (must come after SPRITES is built and after the
# post-processing shader is in place, so TimeOfDay can push uniforms).
# ──────────────────────────────────────────────────────────────────────────────
player_hud   = PlayerHUD(player)
gun          = Gun(player)
time_of_day  = TimeOfDay()           # ← drives the scene_tint uniform
shop         = Shop()                # ← opens every 5 waves
wave_manager = WaveManager()


# ──────────────────────────────────────────────────────────────────────────────
# World geometry — two rooms separated by an inner wall with a door gap
# ──────────────────────────────────────────────────────────────────────────────
#
#   +Z (north)                ┌─────────────────────┐
#       ▲                     │   TREASURE  ROOM    │
#       │                     │                     │
#       │                     ├──────── ░ ──────────┤  ← inner wall, door gap
#       │                     │                     │
#       │                     │     MAIN  HALL      │
#       │                     │     (start)         │
#       │                     └─────────────────────┘
#                              ←──────── X ────────→
#
ROOM_W      = 46         # street width  (X)   — wider town
HALL_D      = 38         # south half  (Z)     — player starts here
TREASURE_D  = 38         # north half  (Z)     — enemies advance from here
ROOM_H      = 10         # taller buildings, plenty of open sky
WALL_T      = 0.4        # wall thickness
DOOR_W      = 3.0        # doorway gap width

TOTAL_D = HALL_D + TREASURE_D
HALF_X  = ROOM_W / 2
Z_FRONT = -HALL_D                 # south outer wall
Z_INNER = 0                       # inner dividing wall
Z_BACK  = TREASURE_D              # north outer wall

# Floor (full footprint) — bold stone-tile texture, ~4 m per repeat
Entity(model='cube', scale=(ROOM_W, 0.2, TOTAL_D),
       position=(0, -0.1, (Z_BACK + Z_FRONT) / 2),
       texture=FLOOR_TEX,
       texture_scale=(ROOM_W / 4, TOTAL_D / 4),
       color=color.white,                  # don't tint — let pattern carry the contrast
       collider='box')

# ──────────────────────────────────────────────────────────────────────────────
# Victorian greenhouse — palette of base colours
# (Different luminances dither at different densities → varied patterns.)
# ──────────────────────────────────────────────────────────────────────────────
IRON    = color.rgb(40, 45, 30)      # dark — reads as mostly black
IRON_L  = color.rgb(75, 80, 60)      # mid-iron, slight green tint
STONE_L = color.rgb(175, 170, 150)   # bright weathered stone
STONE_D = color.rgb(110, 105, 90)    # darker stone
LEAF_D  = color.rgb(45, 70, 30)      # dark foliage
LEAF_M  = color.rgb(95, 140, 60)     # mid foliage
LEAF_L  = color.rgb(170, 200, 105)   # bright leaves catching the light
TERRA   = color.rgb(155, 75, 45)     # terracotta pot
WOOD    = color.rgb(80, 55, 35)      # weathered wood
MOSS    = color.rgb(70, 95, 50)      # mossy vine
SOIL    = color.rgb(55, 40, 25)
WATER   = color.rgb(60, 85, 65)      # stagnant fountain water

# No roof — wide open desert sky above.

# ──────────────────────────────────────────────────────────────────────────────
# Dithered border walls — tall stone perimeter enclosing the conservatory
# (Tall enough you can't see over, low enough that the iron columns + glass
# roof framework still rise above. The brick texture × dither shader gives
# every face a different ordered-dither pattern based on its tint.)
# ──────────────────────────────────────────────────────────────────────────────
BORDER_H = 5.0      # tall wood-plank perimeter — keeps the player on the street
BORDER_T = 0.4      # wall thickness

# Per-face tint — different luminances → different dither densities.
WALL_TINT_FRONT = color.rgb(220, 200, 165)
WALL_TINT_BACK  = color.rgb(180, 160, 130)
WALL_TINT_EAST  = color.rgb(200, 180, 145)
WALL_TINT_WEST  = color.rgb(165, 150, 120)

def border_wall(pos, scale, col):
    # Big bricks — one repeat covers ~3 m horizontally so they read clearly.
    horiz = max(scale[0], scale[2])
    Entity(model='cube', position=pos, scale=scale,
           texture=WALL_TEX,
           texture_scale=(horiz / 3.0, scale[1] / 1.5),
           color=col, collider='box')

# Four tall border walls
border_wall((0,        BORDER_H/2, Z_FRONT),
            (ROOM_W,   BORDER_H,   BORDER_T), WALL_TINT_FRONT)
border_wall((0,        BORDER_H/2, Z_BACK),
            (ROOM_W,   BORDER_H,   BORDER_T), WALL_TINT_BACK)
border_wall(( HALF_X,  BORDER_H/2, (Z_BACK + Z_FRONT) / 2),
            (BORDER_T, BORDER_H,   TOTAL_D),  WALL_TINT_EAST)
border_wall((-HALF_X,  BORDER_H/2, (Z_BACK + Z_FRONT) / 2),
            (BORDER_T, BORDER_H,   TOTAL_D),  WALL_TINT_WEST)

# Bright stone capstone running along the top edge of each border wall
CAP_T = 0.18
for pos, scale in [
    ((0,        BORDER_H + CAP_T/2, Z_FRONT), (ROOM_W, CAP_T, BORDER_T + 0.15)),
    ((0,        BORDER_H + CAP_T/2, Z_BACK),  (ROOM_W, CAP_T, BORDER_T + 0.15)),
    (( HALF_X,  BORDER_H + CAP_T/2, (Z_BACK + Z_FRONT) / 2),
        (BORDER_T + 0.15, CAP_T, TOTAL_D)),
    ((-HALF_X,  BORDER_H + CAP_T/2, (Z_BACK + Z_FRONT) / 2),
        (BORDER_T + 0.15, CAP_T, TOTAL_D)),
]:
    horiz = max(scale[0], scale[2])
    Entity(model='cube', position=pos, scale=scale,
           texture=STONE_TEX, texture_scale=(horiz / 1.5, 1),
           color=color.white)

# Darker plinth at the foot of each wall, anchoring it visually to the floor
PLINTH_H = 0.25
for pos, scale in [
    ((0,        PLINTH_H/2, Z_FRONT), (ROOM_W, PLINTH_H, BORDER_T + 0.2)),
    ((0,        PLINTH_H/2, Z_BACK),  (ROOM_W, PLINTH_H, BORDER_T + 0.2)),
    (( HALF_X,  PLINTH_H/2, (Z_BACK + Z_FRONT) / 2),
        (BORDER_T + 0.2, PLINTH_H, TOTAL_D)),
    ((-HALF_X,  PLINTH_H/2, (Z_BACK + Z_FRONT) / 2),
        (BORDER_T + 0.2, PLINTH_H, TOTAL_D)),
]:
    horiz = max(scale[0], scale[2])
    Entity(model='cube', position=pos, scale=scale,
           texture=STONE_TEX, texture_scale=(horiz / 1.5, 1),
           color=color.rgb(100, 95, 80))

# Back-compatibility aliases — older code still references these
LOW_H, LOW_T = BORDER_H, BORDER_T

# ──────────────────────────────────────────────────────────────────────────────
# Boardwalks — raised wooden plank strips running along the east and west walls
# ──────────────────────────────────────────────────────────────────────────────
BOARD_W   = 2.5       # how far the boardwalk juts into the street
BOARD_H   = 0.18
for side in (-1, 1):
    Entity(model='cube',
           position=(side * (HALF_X - BOARD_W / 2 - BORDER_T / 2),
                     BOARD_H / 2,
                     (Z_BACK + Z_FRONT) / 2),
           scale=(BOARD_W, BOARD_H, TOTAL_D - 2),
           texture=WALL_TEX, texture_scale=(BOARD_W / 1.0, TOTAL_D / 2),
           color=color.rgb(180, 140, 95), collider='box')

# Awning posts along the boardwalks — short wooden uprights every few metres
for side in (-1, 1):
    px = side * (HALF_X - BOARD_W - BORDER_T / 2 + 0.2)
    for pz in range(int(Z_FRONT) + 3, int(Z_BACK) - 2, 6):
        Entity(model='cube', position=(px, ROOM_H * 0.4, pz),
               scale=(0.25, ROOM_H * 0.8, 0.25),
               texture=WALL_TEX, texture_scale=(1, ROOM_H * 0.8),
               color=color.rgb(140, 95, 55), collider='box')
        # Cross-beam supporting the awning
        Entity(model='cube', position=(px + side * 0.6, ROOM_H * 0.78, pz),
               scale=(1.4, 0.2, 0.2),
               texture=WALL_TEX, texture_scale=(1.4, 0.2),
               color=color.rgb(110, 75, 40))

# ──────────────────────────────────────────────────────────────────────────────
# Decorative 2D billboard sprites
# ──────────────────────────────────────────────────────────────────────────────
# Each entry below is a single DecorSprite placed at (x, z) with the size given.
# The named texture comes from textures/sprites/<name>.png — replace any file
# to reskin every sprite of that type at once.
#
# Sprite y-position is the centre of the quad; with origin_y=-0.5 (set on the
# DecorSprite class) the bottom edge sits at y=0, so y = height/2.

# ── Town building facades — flat sprites pressed against the boardwalk posts ─
# The four building types are evenly distributed down both sides of the street.
BLDG_INSET = HALF_X - BOARD_W - BORDER_T - 0.05  # how far in from the wall
BLDG_TYPES = ['saloon', 'sheriff', 'bank', 'general_store']

# West side (negative X) — static pictures pinned facing east
for i, z in enumerate(range(int(Z_FRONT) + 5, int(Z_BACK), 10)):
    name = BLDG_TYPES[i % len(BLDG_TYPES)]
    s = DecorSprite(sprite=name, position=(-BLDG_INSET, 0, z),
                    scale=(7, 6), static=True)
    s.rotation_y = 90

# East side (positive X) — static pictures pinned facing west
for i, z in enumerate(range(int(Z_FRONT) + 9, int(Z_BACK), 10)):
    name = BLDG_TYPES[(i + 2) % len(BLDG_TYPES)]
    s = DecorSprite(sprite=name, position=(BLDG_INSET, 0, z),
                    scale=(7, 6), static=True)
    s.rotation_y = -90

# Wanted posters tacked to the wall — also static
# (these used to spin; now they read like proper paper signs)
# Note: actual wanted_poster placements are in _TOWN_PROPS below — convert there too.

# ── Street props (Y-axis billboards) ─────────────────────────────────────────
_TOWN_PROPS = [
    # Cacti scattered along the dusty edges
    ('cactus',         -10,  -20, 1.2, 2.0),
    ('cactus',          11,  -18, 1.4, 2.2),
    ('cactus',         -12,    2, 1.0, 1.7),
    ('cactus',          12,    8, 1.3, 2.0),
    ('cactus',         -11,   20, 1.1, 1.8),
    ('cactus',          10,   22, 1.2, 1.9),

    # Hitching posts in front of the saloon (west side, near start)
    ('hitching_post', -BLDG_INSET + 3.0, -15, 1.8, 1.2),
    ('hitching_post', -BLDG_INSET + 3.0,  -5, 1.8, 1.2),
    # Water troughs on the east side
    ('water_trough',   BLDG_INSET - 3.0, -10, 2.0, 1.0),
    ('water_trough',   BLDG_INSET - 3.0,  12, 2.0, 1.0),

    # Wagons mid-street as cover landmarks
    ('wagon',          -4,    0, 3.0, 1.8),
    ('wagon',           5,   16, 3.0, 1.8),

    # Tumbleweeds scattered for atmosphere
    ('tumbleweed',     -7,   -8, 0.9, 0.9),
    ('tumbleweed',      3,    6, 1.0, 1.0),
    ('tumbleweed',     -2,   18, 0.8, 0.8),

    # Wanted posters tacked to a couple of buildings
    ('wanted_poster', -BLDG_INSET + 0.3, -12, 1.0, 1.4),
    ('wanted_poster',  BLDG_INSET - 0.3,   4, 1.0, 1.4),
]
for name, x, z, sw, sh in _TOWN_PROPS:
    # Wanted posters are static (pinned to the wall); everything else
    # (cacti, troughs, wagons, tumbleweeds) keeps the Y-billboard behaviour.
    is_static = (name == 'wanted_poster')
    s = DecorSprite(sprite=name, position=(x, 0, z),
                    scale=(sw, sh), static=is_static)
    if is_static:
        # Face into the street (posters live on the wall they're nearest to)
        s.rotation_y = 90 if x < 0 else -90


# ──────────────────────────────────────────────────────────────────────────────
# 3D cover objects — crates, barrels, hay bales the player can hide behind
# (Solid colliders that block raycast bullets and player movement alike.)
# ──────────────────────────────────────────────────────────────────────────────
def crate(x, z, size=1.0):
    Entity(model='cube', position=(x, size / 2, z),
           scale=(size, size, size),
           texture=WOOD_TEX, texture_scale=(2, 2),
           color=color.rgb(185, 135, 85), collider='box')

def barrel(x, z, h=1.2):
    # Approximated as a slightly-tall cube; with the dither it reads fine
    Entity(model='cube', position=(x, h / 2, z),
           scale=(0.85, h, 0.85),
           texture=IRON_TEX, texture_scale=(2, h * 1.5),
           color=color.rgb(130, 115, 100), collider='box')

def hay_bale(x, z):
    Entity(model='cube', position=(x, 0.55, z),
           scale=(1.6, 1.1, 1.0),
           texture=FLOOR_TEX, texture_scale=(2, 1.5),
           color=color.rgb(230, 210, 130), collider='box')

# Cover layout — five "rings" of obstacles down the longer street
# Closest to player spawn
crate(-4, -30);  crate( 5, -28);  barrel(-8, -26);  barrel( 9, -24)
hay_bale( 0, -28)

# Near
crate(-3, -16);  crate( 4, -14);  barrel(-6, -10);  barrel( 8,  -8)
hay_bale( 0, -12);  hay_bale(-12, -18); hay_bale(11, -16)

# Mid street
crate(-8,  -2);  crate( 7,  -3); crate(-2,   3);   crate( 3,   5)
barrel( 0,  -4); barrel(-5,   6); barrel( 8,   2)
hay_bale(-6,   0); hay_bale( 6,   8)

# Far end
crate(-4,  14);  crate( 5,  12);  crate( 0,  20)
barrel(-7,  18); barrel( 7,  18); barrel(-2,  22)
hay_bale( 3,  19); hay_bale(-5,  10)

# Way out — useful when you push north
crate(-9,  28);  crate( 9,  30);  crate(-2,  32);  crate( 3,  34)
barrel(-13, 24); barrel( 13, 26); barrel(0,  36)
hay_bale(-5,  30); hay_bale( 6,  32)

# ──────────────────────────────────────────────────────────────────────────────
# TNT barrels — shoot to detonate. Chain reactions are automatic.
# Placed near enemy spawn lines so they're tactically interesting.
# ──────────────────────────────────────────────────────────────────────────────
for tx, tz in [
    (-6,   6), ( 6,   8), ( 0,  16),       # mid-street — good for early waves
    (-8,  22), ( 8,  22), ( 0,  28),       # north side — clustered near spawns
    ( 4,  30),                              # an extra one for chain potential
]:
    TntBarrel(tx, tz)


# ──────────────────────────────────────────────────────────────────────────────
# Saloon balconies — 3D wooden platforms with railings + stairs you can climb.
# Climb up for a vantage point on the street, or use them as hard cover.
# (Step height is set to ~0.45 m so SmoothFPC's step-up walks you straight up.)
# ──────────────────────────────────────────────────────────────────────────────
def balcony(side, center_z, platform_h=3.0, platform_w=3.0, platform_l=5.0,
            n_steps=7):
    """Build a wooden balcony + stairs against the wall on the given side.
       side: -1 = west wall, +1 = east wall.
       Stairs descend from the SOUTH end of the balcony so you climb facing north."""
    wall_x      = side * (HALF_X - BORDER_T / 2)
    platform_x  = wall_x - side * (platform_w / 2 + 0.1)
    edge_x      = platform_x - side * (platform_w / 2 - 0.07)   # inward edge
    end_z_min   = center_z - platform_l / 2
    end_z_max   = center_z + platform_l / 2

    # ── Platform deck ─────────────────────────────────────────────
    Entity(model='cube',
           position=(platform_x, platform_h, center_z),
           scale=(platform_w, 0.22, platform_l),
            texture=WOOD_TEX, texture_scale=(platform_w, platform_l),
           color=color.rgb(180, 135, 85), collider='box')

    # ── Front railing (full length, facing the street) ────────────
    Entity(model='cube',
           position=(edge_x, platform_h + 0.55, center_z),
           scale=(0.10, 1.0, platform_l),
            texture=IRON_TEX, texture_scale=(1.0, platform_l),
            color=color.rgb(115, 80, 45), collider='box')
    # Vertical posts along the front railing
    for pz in (end_z_min + 0.15, center_z, end_z_max - 0.15):
        Entity(model='cube', position=(edge_x, platform_h + 0.55, pz),
               scale=(0.18, 1.15, 0.18),
             texture=IRON_TEX, texture_scale=(1.0, 1.0),
             color=color.rgb(95, 65, 35), collider='box')

    # ── End railings (north/south sides) ──────────────────────────
    for pz, half_thick in [(end_z_max - 0.05, 0.10), (end_z_min + 0.05, 0.10)]:
        Entity(model='cube',
               position=(platform_x, platform_h + 0.55, pz),
               scale=(platform_w - 0.15, 1.0, half_thick),
             texture=IRON_TEX, texture_scale=(platform_w - 0.15, 1.0),
             color=color.rgb(115, 80, 45), collider='box')

    # ── Support pillars under the deck ────────────────────────────
    for pz in (end_z_min + 0.4, end_z_max - 0.4):
        Entity(model='cube',
               position=(edge_x, platform_h / 2, pz),
               scale=(0.32, platform_h, 0.32),
             texture=WOOD_TEX, texture_scale=(1, platform_h),
             color=color.rgb(110, 75, 45), collider='box')

    # ── Awning roof above the balcony (decorative) ────────────────
    Entity(model='cube',
           position=(platform_x, platform_h + 2.1, center_z),
           scale=(platform_w + 0.4, 0.15, platform_l + 0.3),
           color=color.rgb(95, 65, 35))
    # Two posts holding the awning up off the railing
    for pz in (end_z_min + 0.3, end_z_max - 0.3):
        Entity(model='cube',
               position=(edge_x, platform_h + 1.1, pz),
               scale=(0.16, 1.1, 0.16),
             texture=WOOD_TEX, texture_scale=(1, 1.1),
             color=color.rgb(95, 65, 35), collider='box')

    # ── Stairs at the SOUTH end (player climbs walking north) ─────
    step_h = platform_h / n_steps
    step_d = 0.55
    for i in range(n_steps):
        # i = 0 → bottom step (southernmost, lowest)
        # i = n_steps-1 → top step (northernmost, flush with platform)
        step_top_y = (i + 1) * step_h
        y_center   = step_top_y - step_h / 2
        z_pos      = (end_z_min - 0.3) - (n_steps - 1 - i) * step_d
        Entity(model='cube',
               position=(platform_x, y_center, z_pos),
               scale=(platform_w * 0.7, step_h, step_d),
               texture=WALL_TEX, texture_scale=(2, 1),
               color=color.rgb(155, 115, 70), collider='box')


# Two balconies — one each side, at different positions
balcony(side=-1, center_z=-22, platform_h=3.0)   # west, near player start
balcony(side= 1, center_z=  4, platform_h=3.4)   # east, mid-street


# ──────────────────────────────────────────────────────────────────────────────
# THE SALOON — a 3D enterable building. Walk through the east doorway to enter.
# Inside: a bar counter along the back wall, stools, a couple of tables.
# ──────────────────────────────────────────────────────────────────────────────
SAL_X, SAL_Z       = -13, -16          # building centre
SAL_W, SAL_D, SAL_H = 9.0, 7.0, 4.2    # exterior dimensions (X, Z, Y)
SAL_WALL_T         = 0.3
SAL_DOOR_W         = 2.6               # central east-facing doorway

def _saloon_wall(pos, scale, col=color.rgb(180, 130, 80)):
    horiz = max(scale[0], scale[2])
    Entity(model='cube', position=pos, scale=scale,
           texture=WALL_TEX,
           texture_scale=(horiz / 1.5, scale[1] / 1.0),
           color=col, collider='box')

# Outer walls — north / south / west are solid, east has a door gap
_saloon_wall((SAL_X,             SAL_H/2, SAL_Z - SAL_D/2),     # north
             (SAL_W,             SAL_H,   SAL_WALL_T))
_saloon_wall((SAL_X,             SAL_H/2, SAL_Z + SAL_D/2),     # south
             (SAL_W,             SAL_H,   SAL_WALL_T))
_saloon_wall((SAL_X - SAL_W/2,   SAL_H/2, SAL_Z),               # west (back)
             (SAL_WALL_T,        SAL_H,   SAL_D))

# East wall split into two segments around the door gap
_seg_d = (SAL_D - SAL_DOOR_W) / 2
for sign in (-1, 1):
    _saloon_wall((SAL_X + SAL_W/2,
                  SAL_H/2,
                  SAL_Z + sign * (SAL_DOOR_W/2 + _seg_d/2)),
                 (SAL_WALL_T, SAL_H, _seg_d))
# Door lintel above the gap
_saloon_wall((SAL_X + SAL_W/2, SAL_H - 0.55, SAL_Z),
             (SAL_WALL_T, 1.1, SAL_DOOR_W),
             col=color.rgb(150, 100, 60))

# Flat plank roof
Entity(model='cube',
       position=(SAL_X, SAL_H + 0.18, SAL_Z),
       scale=(SAL_W + 0.5, 0.35, SAL_D + 0.5),
    texture=TERRA_TEX, texture_scale=(SAL_W / 1.4, SAL_D / 1.4),
    color=color.rgb(115, 75, 40), collider='box')

# Big "SALOON" sign hanging above the door
Entity(model='cube',
       position=(SAL_X + SAL_W/2 + 0.18, SAL_H + 0.85, SAL_Z),
       scale=(0.12, 1.0, SAL_DOOR_W + 0.6),
    texture=WOOD_TEX, texture_scale=(SAL_DOOR_W + 0.6, 1.0),
    color=color.rgb(220, 175, 100))

# ── Saloon doors (batwing) — half-height, no collider so you can walk through
for sign in (-1, 1):
    Entity(model='cube',
           position=(SAL_X + SAL_W/2,
                     1.10,
                     SAL_Z + sign * (SAL_DOOR_W/2 - 0.6)),
           scale=(SAL_WALL_T * 0.7, 1.5, SAL_DOOR_W / 2 - 0.2),
            texture=WOOD_TEX, texture_scale=(1, 1.5),
           color=color.rgb(130, 90, 55))

# ── Interior: bar counter along the back (west) wall ────────────
_BAR_INNER_X = SAL_X - SAL_W/2 + 1.4
Entity(model='cube',
       position=(_BAR_INNER_X, 0.55, SAL_Z),
       scale=(1.0, 1.1, SAL_D - 1.6),
    texture=WOOD_TEX, texture_scale=(SAL_D / 1.4, 1.1),
       color=color.rgb(95, 60, 30), collider='box')
# Bar top — slightly wider polished plank
Entity(model='cube',
       position=(_BAR_INNER_X + 0.10, 1.13, SAL_Z),
       scale=(1.25, 0.08, SAL_D - 1.4),
    texture=WOOD_TEX, texture_scale=(SAL_D / 1.2, 0.5),
       color=color.rgb(150, 105, 60), collider='box')

# ── Bar stools ──────────────────────────────────────────────────
for dz in (-2.6, -1.0, 0.6, 2.2):
    Entity(model='cube',
           position=(_BAR_INNER_X + 1.05, 0.42, SAL_Z + dz),
           scale=(0.45, 0.85, 0.45),
            texture=WOOD_TEX, texture_scale=(0.5, 0.85),
           color=color.rgb(110, 75, 45), collider='box')

# ── A square table and four chairs near the door ────────────────
_TBL_X, _TBL_Z = SAL_X + 1.4, SAL_Z + 1.8
Entity(model='cube', position=(_TBL_X, 0.78, _TBL_Z),
       scale=(1.3, 0.08, 1.3),
    texture=WOOD_TEX, texture_scale=(0.8, 0.8),
       color=color.rgb(135, 95, 55), collider='box')
Entity(model='cube', position=(_TBL_X, 0.39, _TBL_Z),
       scale=(0.18, 0.78, 0.18),
       color=color.rgb(80, 55, 35))
for dx, dz in [(-0.95, 0), (0.95, 0), (0, -0.95), (0, 0.95)]:
    Entity(model='cube',
           position=(_TBL_X + dx, 0.42, _TBL_Z + dz),
           scale=(0.45, 0.85, 0.45),
            texture=WOOD_TEX, texture_scale=(0.5, 0.85),
           color=color.rgb(110, 75, 45), collider='box')

# ── Second table further back ───────────────────────────────────
Entity(model='cube',
       position=(SAL_X + 1.6, 0.78, SAL_Z - 1.6),
       scale=(1.0, 0.08, 1.0),
    texture=WOOD_TEX, texture_scale=(0.6, 0.6),
       color=color.rgb(135, 95, 55), collider='box')
Entity(model='cube',
       position=(SAL_X + 1.6, 0.39, SAL_Z - 1.6),
    scale=(0.16, 0.78, 0.16), texture=IRON_TEX,
    color=color.rgb(80, 55, 35))

# Terracotta planters break up the street edges and give the terra texture a
# visible role beyond the roofline.
planter(-BLDG_INSET + 1.6, -23, 0.9, sprite='plant')
planter( BLDG_INSET - 1.6, -9,  0.9, sprite='plant')
planter(-BLDG_INSET + 2.0,  10, 1.0, sprite='cactus')
planter( BLDG_INSET - 2.0,  24, 1.0, sprite='cactus')

# ── A free ammo crate inside the saloon — reward for going in ───
AmmoPickup(position=(_BAR_INNER_X + 0.10, 1.5, SAL_Z - SAL_D/2 + 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Pickups — a few coins glinting on the street, plus a "trophy" sheriff's badge
# at the far end of the street as a stretch goal.
# ──────────────────────────────────────────────────────────────────────────────
for cx, cz in [(-3, -8), (4, 4), (-6, 14), (6, 22), (-1, 18)]:
    Pickup(item_name='Coin', position=(cx, 0.7, cz), scale=(0.4, 0.4))

# Sheriff's badge at the far north end — pick it up after clearing the street
Trophy(position=(0, 1.5, Z_BACK - 1.5))


# Enemies are no longer hand-placed — WaveManager spawns waves of outlaws
# automatically at the far end and flanks of the street.


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing: 1-bit Bayer dither
# ──────────────────────────────────────────────────────────────────────────────
scene_tex  = P3DTexture('scene_color')
filter_mgr = FilterManager(base.win, base.cam)
quad       = filter_mgr.renderSceneInto(colortex=scene_tex)

dither_enabled = True

if quad is None:
    print('[WARNING] FilterManager could not create render buffer; dither off.')
    dither_enabled = False
else:
    scene_tex.setMagfilter(SamplerState.FT_nearest)
    scene_tex.setMinfilter(SamplerState.FT_nearest)
    dither_shader = P3DShader.make(P3DShader.SL_GLSL, _VERT, _FRAG)
    quad.setShader(dither_shader)
    quad.setShaderInput('scene_tex', scene_tex)
    quad.setShaderInput('resolution',
                        LVecBase2f(base.win.getXSize(), base.win.getYSize()))
    # NOON default palette — pure 1-bit black and white
    quad.setShaderInput('fg_color', LVecBase3f(1.00, 1.00, 1.00))
    quad.setShaderInput('bg_color', LVecBase3f(0.00, 0.00, 0.00))
    time_of_day.reapply()      # push the *current* phase if it's not noon


# ──────────────────────────────────────────────────────────────────────────────
# HUD updater — handles coordinates, dither toggle, and interaction input
# ──────────────────────────────────────────────────────────────────────────────
class HUDUpdater(Entity):
    def __init__(self, player_entity):
        super().__init__()
        self.player = player_entity
        self.dither_t_pressed = False
        self.e_pressed = False
    
    def update(self):
        # Update coordinates display
        pos = self.player.position
        coords_text.text = f'X: {pos.x:6.2f}  Y: {pos.y:6.2f}  Z: {pos.z:6.2f}'
        
        # Handle T key for dither toggle (using held_keys to avoid blocking other input)
        if 't' in held_keys:
            if not self.dither_t_pressed and quad is not None:
                global dither_enabled
                self.dither_t_pressed = True
                dither_enabled = not dither_enabled
                if dither_enabled:
                    quad.setShader(dither_shader)
                    quad.setShaderInput('scene_tex', scene_tex)
                    quad.setShaderInput('resolution',
                                        LVecBase2f(base.win.getXSize(),
                                                   base.win.getYSize()))
                    time_of_day.reapply()    # restore current scene_tint
                else:
                    quad.clearShader()
        else:
            self.dither_t_pressed = False
        
        # Handle E key for interaction (using held_keys)
        if 'e' in held_keys:
            if not self.e_pressed and interaction.target is not None:
                self.e_pressed = True
                interaction.target.interact()
        else:
            self.e_pressed = False

hud_updater = HUDUpdater(player)


# Opening message
message_log.show('Look around with the mouse. Press [E] on the sign.', 6)

app.run()
