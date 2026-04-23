"""
Draw clean dual-student distillation framework with PIL (no matplotlib required).
Outputs: docs/figures/framework_clean.png
"""
import sys
sys.path.insert(0, "/Users/lena/dasishang/KD/.lib")

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUT = "/Users/lena/dasishang/KD/docs/figures/framework_clean.png"
W, H = 3600, 2200   # canvas  (×2 for Retina/DPI)

# ── 颜色 ────────────────────────────────────────────
TITLE_BG  = (30, 58, 95)
TITLE_FG  = (255, 255, 255)
TITLE_SUB = (147, 197, 253)

T_BG, T_ED = (235, 248, 255), (49, 130, 206)
T_TXT      = (30, 64, 175)
CNN_BG, CNN_ED = (245, 243, 255), (124, 58, 237)
CNN_TXT    = (76, 29, 149)
LR_BG, LR_ED   = (254, 243, 199), (217, 119, 6)
LR_TXT      = (146, 64, 14)
DATA_BG, DATA_ED = (230, 255, 250), (56, 178, 172)
DATA_TXT   = (0, 105, 100)
MET_BG, MET_ED = (240, 253, 244), (22, 163, 74)
MET_TXT    = (22, 101, 52)
SEP        = (226, 232, 240)
GRAY       = (100, 116, 139)
DARK       = (30, 41, 59)
WHITE      = (255, 255, 255)
ACCENT_BG  = (255, 251, 235)
ACCENT_ED  = (245, 158, 11)
KD_CLR     = (49, 130, 206)
ARR_CLR    = (51, 65, 85)

def rr(draw, xy, r=12, fill=None, outline=None, width=2):
    x, y, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)

def text_center(draw, xy, txt, fill, font, max_width=None):
    x, y = xy
    try:
        bbox = draw.textbbox((0, 0), txt, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(txt, font=font)
    if max_width and w > max_width:
        scale = max_width / w
        font = font.font_variant(size=int(font.size * scale))
        bbox = draw.textbbox((0, 0), txt, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x - w // 2, y - h // 2), txt, fill=fill, font=font)

def text_left(draw, xy, txt, fill, font):
    x, y = xy
    try:
        bbox = draw.textbbox((0, 0), txt, font=font)
        h = bbox[3] - bbox[1]
    except Exception:
        _, h = draw.textsize(txt, font=font)
    draw.text((x, y - h // 2), txt, fill=fill, font=font)

def arrow_line(draw, x1, y1, x2, y2, color, width=3):
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # arrowhead
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    aw, ah = 12, 6
    ax1 = x2 - aw * math.cos(angle) + ah * math.sin(angle)
    ay1 = y2 - aw * math.sin(angle) - ah * math.cos(angle)
    ax2 = x2 - aw * math.cos(angle) - ah * math.sin(angle)
    ay2 = y2 - aw * math.sin(angle) + ah * math.cos(angle)
    draw.polygon([(x2, y2), (ax1, ay1), (ax2, ay2)], fill=color)

def dashed_arrow(draw, x1, y1, x2, y2, color, width=2, dash=12):
    import math
    angle = math.atan2(y2 - y1, x2 - x1)
    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)
    nx = dx / dist
    ny = dy / dist
    cur = 0.0
    i = 0
    while cur < dist:
        sx = x1 + nx * cur
        sy = y1 + ny * cur
        cur += dash if i % 2 == 0 else dash // 2
        ex = x1 + nx * min(cur, dist)
        ey = y1 + ny * min(cur, dist)
        draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
        cur += dash // 2
        i += 1
    aw, ah = 10, 5
    ax1 = x2 - aw * nx + ah * ny
    ay1 = y2 - aw * ny - ah * nx
    ax2 = x2 - aw * nx - ah * ny
    ay2 = y2 - aw * ny + ah * nx
    draw.polygon([(x2, y2), (ax1, ay1), (ax2, ay2)], fill=color)

# ── 字体 ─────────────────────────────────────────────
try:
    FONT_TITLE  = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 38)
    FONT_SUB    = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    FONT_BOLD   = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
    FONT_BOLD2  = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    FONT_MAIN   = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    FONT_SMALL  = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    FONT_ITALIC = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Italic.ttf", 17)
    FONT_TINY   = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
except:
    FONT_TITLE  = ImageFont.load_default()
    FONT_SUB    = ImageFont.load_default()
    FONT_BOLD   = ImageFont.load_default()
    FONT_BOLD2  = ImageFont.load_default()
    FONT_MAIN   = ImageFont.load_default()
    FONT_SMALL  = ImageFont.load_default()
    FONT_ITALIC = ImageFont.load_default()
    FONT_TINY   = ImageFont.load_default()

img = Image.new("RGB", (W, H), WHITE)
draw = ImageDraw.Draw(img)

# ── 标题栏 ───────────────────────────────────────────
draw.rectangle([0, 0, W, 100], fill=TITLE_BG)
text_center(draw, (W//2, 30), "Dual-Student Knowledge Distillation for Encrypted Traffic Classification",
            TITLE_FG, FONT_TITLE)
text_center(draw, (W//2, 72), "Transformer teacher  |  CNN student (deployment)  |  Logistic Regression student (interpretable)",
            TITLE_SUB, FONT_SUB)

# ════════════════════════════════════════════════════════
#  主流程行  y ≈ 200 – 1350
# ════════════════════════════════════════════════════════
ROW = 800   # 主体行 Y

# 列中心 X
CX_DATA    = 240
CX_TEACHER = 780
CX_CNN     = 1320
CX_LR      = 1900
CX_MET     = 2480

BW = 300    # box width
BH = 500    # box height

# ── ① 输入 ────────────────────────────────────────────
rr(draw, [CX_DATA - BW//2, ROW - BH//2, CX_DATA + BW//2, ROW + BH//2], r=16,
   fill=DATA_BG, outline=DATA_ED, width=3)
text_center(draw, (CX_DATA, ROW - 130), "Encrypted", DATA_TXT, FONT_BOLD)
text_center(draw, (CX_DATA, ROW - 95),  "Traffic",   DATA_TXT, FONT_BOLD)
text_center(draw, (CX_DATA, ROW - 40),  "Packet sequence", DARK, FONT_MAIN)
text_center(draw, (CX_DATA, ROW + 5),   "X  (L x D)", GRAY, FONT_SMALL)
text_center(draw, (CX_DATA, ROW + 60),  "to Teacher", GRAY, FONT_SMALL)
text_center(draw, (CX_DATA, ROW + 90),  "& CNN", GRAY, FONT_SMALL)

# ── ② Transformer 教师 ──────────────────────────────
rr(draw, [CX_TEACHER - BW//2, ROW - BH//2, CX_TEACHER + BW//2, ROW + BH//2], r=16,
   fill=T_BG, outline=T_ED, width=3)
text_center(draw, (CX_TEACHER, ROW - 170), "Teacher", T_TXT, FONT_BOLD)
text_center(draw, (CX_TEACHER, ROW - 128), "Transformer", T_TXT, FONT_BOLD2)
draw.line([CX_TEACHER-90, ROW-88, CX_TEACHER+90, ROW-88], fill=T_ED, width=2)
text_center(draw, (CX_TEACHER, ROW - 55), "L x Transformer blocks", GRAY, FONT_SMALL)
text_center(draw, (CX_TEACHER, ROW - 20), "Multi-head self-attention", GRAY, FONT_SMALL)
text_center(draw, (CX_TEACHER, ROW + 25),  "Feed-forward network", GRAY, FONT_SMALL)
draw.line([CX_TEACHER-90, ROW+72, CX_TEACHER+90, ROW+72], fill=T_ED, width=2)
text_center(draw, (CX_TEACHER, ROW + 115), "z_T  (logits)", T_TXT, FONT_BOLD2)
text_center(draw, (CX_TEACHER, ROW + 160), "p_T = softmax(z_T / tau)", GRAY, FONT_ITALIC)
text_center(draw, (CX_TEACHER, ROW + 185), "tau = 2.0", GRAY, FONT_TINY)

# ── ③ CNN 学生 ───────────────────────────────────────
rr(draw, [CX_CNN - BW//2, ROW - BH//2, CX_CNN + BW//2, ROW + BH//2], r=16,
   fill=CNN_BG, outline=CNN_ED, width=3)
text_center(draw, (CX_CNN, ROW - 170), "Student-I  (CNN)", CNN_TXT, FONT_BOLD)
text_center(draw, (CX_CNN, ROW - 128), "1D CNN", CNN_TXT, FONT_BOLD2)
draw.line([CX_CNN-90, ROW-88, CX_CNN+90, ROW-88], fill=CNN_ED, width=2)
text_center(draw, (CX_CNN, ROW - 55), "Conv1d + BN + ReLU", GRAY, FONT_SMALL)
text_center(draw, (CX_CNN, ROW - 20), "Adaptive pool + Dropout", GRAY, FONT_SMALL)
text_center(draw, (CX_CNN, ROW + 25), "Lightweight deploy", GRAY, FONT_SMALL)
draw.line([CX_CNN-90, ROW+72, CX_CNN+90, ROW+72], fill=CNN_ED, width=2)
text_center(draw, (CX_CNN, ROW + 115), "z_C  (logits)", CNN_TXT, FONT_BOLD2)
text_center(draw, (CX_CNN, ROW + 160), "Student for inference", GRAY, FONT_SMALL)

# ── ④ LR 学生 ────────────────────────────────────────
rr(draw, [CX_LR - BW//2, ROW - BH//2, CX_LR + BW//2, ROW + BH//2], r=16,
   fill=LR_BG, outline=LR_ED, width=3)
text_center(draw, (CX_LR, ROW - 170), "Student-II  (LR)", LR_TXT, FONT_BOLD)
text_center(draw, (CX_LR, ROW - 128), "Logistic Regression", LR_TXT, FONT_BOLD2)
draw.line([CX_LR-90, ROW-88, CX_LR+90, ROW-88], fill=LR_ED, width=2)
text_center(draw, (CX_LR, ROW - 55), "Interpretable linear model", GRAY, FONT_SMALL)
text_center(draw, (CX_LR, ROW - 20), "y = sigmoid(Wx + b)", GRAY, FONT_ITALIC)
text_center(draw, (CX_LR, ROW + 25), "Transparent weights W", GRAY, FONT_SMALL)
draw.line([CX_LR-90, ROW+72, CX_LR+90, ROW+72], fill=LR_ED, width=2)
text_center(draw, (CX_LR, ROW + 115), "y-hat  (prediction)", LR_TXT, FONT_BOLD2)
text_center(draw, (CX_LR, ROW + 160), "Explainable decision", GRAY, FONT_SMALL)

# ── ⑤ 指标 ────────────────────────────────────────────
MW = 200
rr(draw, [CX_MET - MW//2, ROW - 300, CX_MET + MW//2, ROW + 300], r=14,
   fill=MET_BG, outline=MET_ED, width=3)
text_center(draw, (CX_MET, ROW - 220), "Metrics", MET_TXT, FONT_BOLD)
for i, label in enumerate(["Accuracy", "Model size (KB)", "Inference latency",
                             "Feature weights W",
                             "(interpretability)"]):
    y_pos = ROW - 150 + i * 58
    text_center(draw, (CX_MET, y_pos), label, GRAY, FONT_TINY)

# ── 主箭头 ───────────────────────────────────────────
# Data → Teacher
arrow_line(draw, CX_DATA + BW//2, ROW, CX_TEACHER - BW//2, ROW, DATA_ED, width=4)
# Teacher → CNN
arrow_line(draw, CX_TEACHER + BW//2, ROW, CX_CNN - BW//2, ROW, T_ED, width=4)
# CNN → LR
arrow_line(draw, CX_CNN + BW//2, ROW, CX_LR - BW//2, ROW, CNN_ED, width=4)
# LR → Metrics
arrow_line(draw, CX_LR + BW//2, ROW, CX_MET - MW//2, ROW, LR_ED, width=4)

# ── KD 虚线：Teacher 上方 → CNN 上方 ────────────────
dashed_arrow(draw, CX_TEACHER + BW//2 - 10, ROW - BH//2 + 50,
            CX_CNN - BW//2 + 10, ROW - BH//2 + 50, KD_CLR, width=2)
# 标注
kdx = (CX_TEACHER + CX_CNN) // 2
text_center(draw, (kdx, ROW - BH//2 + 28), "KD (soft target)", KD_CLR, FONT_ITALIC)
text_center(draw, (kdx, ROW - BH//2 + 50), "tau = 2.0", KD_CLR, FONT_TINY)

# ── LR 协同虚线 ─────────────────────────────────────
# Teacher (左下) → LR (左上)
dashed_arrow(draw, CX_TEACHER - BW//2 + 20, ROW + 60,
            CX_LR - BW//2 - 5, ROW + 60, LR_ED, width=2)
# CNN (右下) → LR (左下)
dashed_arrow(draw, CX_CNN + BW//2 - 10, ROW + 150,
            CX_LR - BW//2 + 5, ROW + 100, LR_ED, width=2)
text_center(draw, (CX_CNN + 80, ROW + 180), "z_T  +  z_C", LR_ED, FONT_ITALIC)

# ════════════════════════════════════════════════════════
#  分隔线 + 底部三目标
# ════════════════════════════════════════════════════════
SEP_Y = 1150
draw.line([60, SEP_Y, W-60, SEP_Y], fill=SEP, width=2)

# 三目标
GBW = 920   # goal box width
GBH = 320
GBY = 400

goals = [
    (100,  ACCENT_BG, ACCENT_ED, DATA_TXT,
     "Goal 1  Lightweight",
     ["CNN student replaces heavy Transformer",
      "Model size:  453.6 KB  →  10.0 KB",
      "Enables edge deployment"]),
    (1100, (239, 246, 255), KD_CLR, T_TXT,
     "Goal 2  Knowledge Transfer",
     ["KD loss aligns CNN with teacher logits",
      "Teacher 85.0%  →  CNN 73.5%",
      "Dark knowledge preserved in soft targets"]),
    (2100, (254, 249, 195), ACCENT_ED, (146, 64, 14),
     "Goal 3  Interpretability",
     ["LR student via transparent weights W",
      "LR co-distill:  85.4%  (0.8 KB)",
      "Decision basis directly analyzable"]),
]

for gx, gbg, ged, gtxt_color, gtitle, glines in goals:
    rr(draw, [gx, GBY, gx + GBW, GBY + GBH], r=14, fill=gbg, outline=ged, width=3)
    text_left(draw, (gx + 30, GBY + 30), gtitle, gtxt_color, FONT_BOLD)
    draw.line([gx + 20, GBY + 72, gx + GBW - 20, GBY + 72], fill=ged, width=2)
    for i, line in enumerate(glines):
        text_left(draw, (gx + 30, GBY + 95 + i * 52), line, DARK, FONT_SMALL)

# ── 图例 ─────────────────────────────────────────────
LEG_X, LEG_Y = W - 480, 130
rr(draw, [LEG_X, LEG_Y, LEG_X + 400, LEG_Y + 120], r=10,
   fill=(248, 250, 252), outline=SEP, width=2)
patches = [
    (T_BG,    T_ED,    "Teacher (Transformer)"),
    (CNN_BG,  CNN_ED,  "Student-I (CNN)"),
    (LR_BG,   LR_ED,   "Student-II (LR)"),
]
for i, (bg, ed, label) in enumerate(patches):
    px = LEG_X + 20
    py = LEG_Y + 18 + i * 36
    rr(draw, [px, py, px + 22, py + 22], r=5, fill=bg, outline=ed, width=2)
    text_left(draw, (px + 32, py + 5), label, DARK, FONT_SMALL)

# ── 底部注释 ─────────────────────────────────────────
text_center(draw, (W//2, H - 60),
    "All accuracy numbers from midterm report (March 2026).  Sizes from train_distill_packet.py.",
    GRAY, FONT_TINY)

# ── 保存 ─────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT), exist_ok=True)
img.save(OUT, "PNG", optimize=True)
print(f"Saved: {OUT}  ({W}x{H})")
