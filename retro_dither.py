#!/usr/bin/env python3
"""
Retro 1-Bit Dithered First-Person Demo
=======================================
Requires:  pip install ursina
Controls:  WASD = move, Mouse = look, ESC = quit, T = toggle dither

Features:
  - FirstPersonController in a hand-built 3D room
  - Full-screen Bayer 4x4 ordered-dither GLSL post-process shader
  - Y-axis billboard sprites (only yaw toward the camera)
  - Nearest-neighbour texture filtering on all sprite textures
"""
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from panda3d.core import (
    Texture, SamplerState, LVecBase2f,
    Shader as P3DShader, AntialiasAttrib,
)
from direct.filter.FilterManager import FilterManager
from PIL import Image
from pathlib import Path
import math

# Asset paths
ASSET_DIR = Path(__file__).resolve().parent / 'textures'
SPRITE_DIR = ASSET_DIR / 'sprites'


def load_asset(name: str):
    return Texture(Image.open(ASSET_DIR / name).convert('RGBA'))


def load_sprite(name: str):
    return Texture(Image.open(SPRITE_DIR / f'{name}.png').convert('RGBA'))

# ──────────────────────────────────────────────────────────────────────────────
# GLSL – full-screen Bayer 4×4 ordered-dither shader
# ──────────────────────────────────────────────────────────────────────────────
#  Vertex shader: just passes the full-screen quad through unchanged.
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

#  Fragment shader:
#   1. Sample the 3-D scene colour.
#   2. Convert to Rec.601 luminance.
#   3. Threshold against the Bayer matrix position → 1-bit output.
_FRAG = """
#version 140
uniform sampler2D scene_tex;
uniform vec2      resolution;   // window size in pixels
in  vec2 uv;
out vec4 frag_color;

// 4×4 Bayer ordered-dither matrix (values 0–15, normalised to 0–1).
// Pattern tiles across the screen so no two neighbouring pixels share
// the same threshold, spreading quantisation error visually.
float bayer4(ivec2 p) {
    const int M[16] = int[16](
         0,  8,  2, 10,
        12,  4, 14,  6,
         3, 11,  1,  9,
        15,  7, 13,  5
    );
    // Wrap with bitwise AND – equivalent to mod 4 but branch-free.
    return float(M[(p.y & 3) * 4 + (p.x & 3)]) / 16.0;
}

void main() {
    vec3  rgb    = texture(scene_tex, uv).rgb;
    // Rec.601 perceptual luminance weights.
    float luma   = dot(rgb, vec3(0.299, 0.587, 0.114));
    float thresh = bayer4(ivec2(uv * resolution));
    float bit    = luma > thresh ? 1.0 : 0.0;
    frag_color   = vec4(bit, bit, bit, 1.0);
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Billboard sprite entity
# ──────────────────────────────────────────────────────────────────────────────
class BillboardSprite(Entity):
    """
    A flat quad that rotates around the world Y axis every frame so its
    front face always points toward the camera.  Pitch and roll are left at
    zero so the sprite never tilts.

    Usage:
        BillboardSprite(texture='my_sprite', position=(3, 1.5, 3), scale=2)
    """

    def __init__(self, **kwargs):
        super().__init__(model='quad', double_sided=True, **kwargs)

    def update(self):
        dx = camera.world_position.x - self.world_position.x
        dz = camera.world_position.z - self.world_position.z
        # atan2(dx, dz) gives the heading from +Z to the camera in the XZ plane.
        self.rotation_y = math.degrees(math.atan2(dx, dz))


# ──────────────────────────────────────────────────────────────────────────────
# Utility: force nearest-neighbour (pixelated) filtering on an entity texture
# ──────────────────────────────────────────────────────────────────────────────
def set_nearest(entity: Entity) -> None:
    """Disable bilinear filtering so pixel-art textures stay crisp."""
    if entity.texture and hasattr(entity.texture, '_texture'):
        t = entity.texture._texture
        t.setMagfilter(SamplerState.FT_nearest)
        t.setMinfilter(SamplerState.FT_nearest)
        t.setAnisotropicDegree(0)


# ──────────────────────────────────────────────────────────────────────────────
# Initialise app
# ──────────────────────────────────────────────────────────────────────────────
app = Ursina(title='1-Bit Dither', vsync=True)
window.color = color.rgb(45, 45, 45)

# Hard pixel edges – no MSAA blurring the dither pattern.
base.render.setAntialias(AntialiasAttrib.MNone)

# ── Player ────────────────────────────────────────────────────────────────────
player = FirstPersonController(y=1, origin_y=-0.5)
player.speed        = 5
player.mouse_sensitivity = Vec2(40, 40)
player.cursor.visible    = False

# ── Room dimensions ───────────────────────────────────────────────────────────
ROOM_W = 20      # full width / depth
ROOM_H = 6       # ceiling height
HW     = ROOM_W / 2

# ── Floor & ceiling ───────────────────────────────────────────────────────────
Entity(model='plane',
       scale=(ROOM_W, 1, ROOM_W), y=0,
    texture=load_asset('floor_v3.png'),
    texture_scale=(ROOM_W / 4, ROOM_W / 4),
    color=color.white,
       collider='box')

Entity(model='plane',
       scale=(ROOM_W, 1, ROOM_W), y=ROOM_H, rotation_x=180,
    texture=load_asset('wall_v3.png'),
    texture_scale=(ROOM_W / 4, ROOM_W / 4),
    color=color.white,
       collider='box')

# ── Walls (thin cubes – avoids rotation-convention confusion) ─────────────────
_WALL_DEFS = [
    dict(pos=(  0,   ROOM_H/2,  HW), scale=(ROOM_W, ROOM_H, 0.5)),  # north
    dict(pos=(  0,   ROOM_H/2, -HW), scale=(ROOM_W, ROOM_H, 0.5)),  # south
    dict(pos=( HW,   ROOM_H/2,   0), scale=(0.5, ROOM_H, ROOM_W)),  # east
    dict(pos=(-HW,   ROOM_H/2,   0), scale=(0.5, ROOM_H, ROOM_W)),  # west
]
for wd in _WALL_DEFS:
    Entity(model='cube', position=wd['pos'], scale=wd['scale'],
           texture=load_asset('wall_v2.png'),
           texture_scale=(ROOM_W / 4, ROOM_H / 2),
           color=color.white, collider='box')

# ── Decorative pillars ────────────────────────────────────────────────────────
for i in range(6):
    a = math.radians(i * 60)
    Entity(model='cube',
           position=(math.sin(a) * 6, ROOM_H / 4, math.cos(a) * 6),
           scale=(0.8, ROOM_H / 2, 0.8),
           color=color.rgb(90, 90, 90),
           collider='box')

# ── A raised platform in the centre ──────────────────────────────────────────
Entity(model='cube',
       position=(0, 0.25, 0), scale=(4, 0.5, 4),
       color=color.rgb(70, 70, 70),
       collider='box')

# ── Billboard sprites ─────────────────────────────────────────────────────────
# Use the actual sprite textures shipped in textures/sprites.
_SPRITES = [
    ('tree',         ( 3,  2,  3), (1.3, 2.6, 1)),
    ('cactus',       (-3,  2,  3), (1.1, 2.2, 1)),
    ('wagon',        ( 0,  2, -5), (1.8, 1.2, 1)),
    ('wanted_poster',( 5,  2, -3), (1.0, 1.4, 1)),
    ('trophy',       (-5,  2, -3), (1.0, 1.6, 1)),
]
for sprite_name, pos, scale in _SPRITES:
    sp = BillboardSprite(
        texture=load_sprite(sprite_name),
        position=pos,
        scale=scale,
        color=color.rgb(200, 200, 200),
    )
    set_nearest(sp)

# ── Crosshair ─────────────────────────────────────────────────────────────────
crosshair = Text('+', origin=(0, 0), scale=2,
                 color=color.white, parent=camera.ui)

# ── Controls hint (rendered on the 2-D UI layer – not dithered) ───────────────
Text('WASD · Move     Mouse · Look     T · Toggle Dither     ESC · Quit',
     position=(-0.84, 0.47), scale=1.1, color=color.white)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing: Bayer ordered-dither
# ──────────────────────────────────────────────────────────────────────────────
scene_tex  = Texture('scene_color')
filter_mgr = FilterManager(base.win, base.cam)
quad       = filter_mgr.renderSceneInto(colortex=scene_tex)

dither_enabled = True

if quad is None:
    print('[WARNING] FilterManager could not create render buffer.')
    print('          Dither effect will be unavailable.')
    dither_enabled = False
else:
    # Nearest-neighbour on the captured scene keeps any lo-fi look intact.
    scene_tex.setMagfilter(SamplerState.FT_nearest)
    scene_tex.setMinfilter(SamplerState.FT_nearest)

    dither_shader = P3DShader.make(P3DShader.SL_GLSL, _VERT, _FRAG)
    quad.setShader(dither_shader)
    quad.setShaderInput('scene_tex', scene_tex)
    quad.setShaderInput('resolution',
                        LVecBase2f(base.win.getXSize(), base.win.getYSize()))


# ──────────────────────────────────────────────────────────────────────────────
# Toggle dither on/off with T (useful for comparison)
# ──────────────────────────────────────────────────────────────────────────────
def input(key):
    global dither_enabled
    if key == 't' and quad is not None:
        dither_enabled = not dither_enabled
        if dither_enabled:
            quad.setShader(dither_shader)
            quad.setShaderInput('scene_tex', scene_tex)
            quad.setShaderInput('resolution',
                                LVecBase2f(base.win.getXSize(),
                                           base.win.getYSize()))
        else:
            quad.clearShader()
        print(f'Dither: {"ON" if dither_enabled else "OFF"}')


app.run()
