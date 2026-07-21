"""Generate a self-contained, publication-ready SVG redraw of 结构图4.png."""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from pathlib import Path

from PIL import Image


OUT = Path(__file__).resolve().parent / "figures" / "structure4_metabci_vector.svg"
SOURCE = Path(r"C:\Users\Administrator\Desktop\机器人大赛\MetaBCI相关\决赛材料\结构图4.png")

# Pixel bounds measured from the supplied 1366 x 1151 source image.
ICON_CROPS = {
    "brainstim": (78, 165, 138, 225),
    "brainda": (563, 164, 624, 225),
    "brainflow": (1048, 165, 1109, 225),
    "psychopy": (37, 426, 101, 502),
    "sine": (41, 638, 99, 697),
    "trigger_mark": (45, 786, 91, 841),
    "neuracle": (306, 430, 358, 502),
    "triggerbox": (305, 543, 362, 596),
    "online": (304, 663, 355, 718),
    "offline": (304, 786, 359, 838),
    "algorithm": (578, 429, 625, 489),
    "window": (576, 559, 628, 612),
    "stopping": (577, 674, 631, 733),
    "adaptive": (576, 837, 628, 896),
    "process": (859, 431, 912, 486),
    "websocket": (858, 548, 912, 601),
    "json": (862, 673, 908, 728),
    "time": (858, 811, 912, 866),
    "canvas": (1135, 430, 1185, 484),
    "directions": (1137, 580, 1328, 631),
    "confidence": (1137, 800, 1188, 856),
    "gamepad": (326, 998, 397, 1061),
}


def embedded_crop(key, x, y, w, h):
    """Return a self-contained SVG image element using an exact source icon."""
    with Image.open(SOURCE) as source:
        crop = source.crop(ICON_CROPS[key]).convert("RGB")
        buffer = BytesIO()
        crop.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'href="data:image/png;base64,{encoded}" '
        f'xlink:href="data:image/png;base64,{encoded}"/>'
    )


def rect(x, y, w, h, fill, stroke, rx=12, sw=1.2):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def txt(x, y, value, size=16, weight=400, anchor="start", fill="#15191d"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}">{escape(value)}</text>'
    )


def line(x1, y1, x2, y2, color="#35618a", sw=2, dash=None, arrow=False):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker = ' marker-end="url(#arrow-blue)"' if arrow else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}" fill="none"{dash_attr}{marker}/>'
    )


def arrow_head(x, y, direction, color, size=8):
    if direction == "down":
        points = ((x - size, y - size), (x + size, y - size), (x, y))
    elif direction == "up":
        points = ((x - size, y + size), (x + size, y + size), (x, y))
    elif direction == "left":
        points = ((x + size, y - size), (x + size, y + size), (x, y))
    else:
        points = ((x - size, y - size), (x - size, y + size), (x, y))
    data = " ".join(f"{px},{py}" for px, py in points)
    return f'<polygon points="{data}" fill="{color}"/>'


def path(d, color="#35618a", sw=2, dash=None, arrow_end=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    result = (
        f'<path d="{d}" stroke="{color}" stroke-width="{sw}" fill="none" '
        f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
    )
    if arrow_end:
        result += arrow_head(*arrow_end, color=color)
    return result


def badge(cx, cy, r, fill, label, size=13):
    return "".join(
        [
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>',
            txt(cx, cy + size * 0.34, label, size, 700, "middle", "#ffffff"),
        ]
    )


def direction_vector_icon(x, y, size=54):
    """Draw a square, fully vector four-direction classification icon."""
    scale = size / 54
    centers = [
        (x + 16 * scale, y + 16 * scale, "#e94f5b", "up"),
        (x + 38 * scale, y + 16 * scale, "#55ad62", "down"),
        (x + 16 * scale, y + 38 * scale, "#397ad6", "left"),
        (x + 38 * scale, y + 38 * scale, "#efb72f", "right"),
    ]
    parts = [
        f'<g id="direction-four-class">',
        f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="8" '
        f'fill="#f6f7f9" stroke="#d9dde2" stroke-width="1"/>',
    ]
    for cx, cy, color, direction in centers:
        radius = 9 * scale
        arm = 5 * scale
        head = 4 * scale
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}"/>')
        if direction == "up":
            d = (
                f'M {cx} {cy + arm} V {cy - arm} '
                f'M {cx - head} {cy - arm + head} L {cx} {cy - arm} '
                f'L {cx + head} {cy - arm + head}'
            )
        elif direction == "down":
            d = (
                f'M {cx} {cy - arm} V {cy + arm} '
                f'M {cx - head} {cy + arm - head} L {cx} {cy + arm} '
                f'L {cx + head} {cy + arm - head}'
            )
        elif direction == "left":
            d = (
                f'M {cx + arm} {cy} H {cx - arm} '
                f'M {cx - arm + head} {cy - head} L {cx - arm} {cy} '
                f'L {cx - arm + head} {cy + head}'
            )
        else:
            d = (
                f'M {cx - arm} {cy} H {cx + arm} '
                f'M {cx + arm - head} {cy - head} L {cx + arm} {cy} '
                f'L {cx + arm - head} {cy + head}'
            )
        parts.append(
            f'<path d="{d}" fill="none" stroke="#ffffff" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    parts.append("</g>")
    return "".join(parts)


def card(
    x,
    y,
    w,
    h,
    accent,
    icon_key,
    heading,
    details,
    *,
    detail_size=14.5,
    detail_gap=27,
):
    parts = [rect(x, y, w, h, "#ffffff", accent, 12, 1.05)]
    if icon_key == "directions":
        parts.append(direction_vector_icon(x + 8, y + 12, 54))
    else:
        parts.append(embedded_crop(icon_key, x + 9, y + 13, 52, 52))
    parts.append(txt(x + 67, y + 34, heading, 17, 700))
    start_y = y + 62
    for i, detail in enumerate(details):
        parts.append(
            txt(
                x + 67,
                start_y + i * detail_gap,
                detail,
                detail_size,
                400,
                fill="#2d3339",
            )
        )
    return "".join(parts)


def platform_box(x, y, w, accent, name, subtitle):
    return "".join(
        [
            rect(x, y, w, 94, "#ffffff", accent, 12, 1.4),
            embedded_crop(name, x + 15, y + 17, 58, 58),
            txt(x + 88, y + 41, name, 23, 700),
            txt(x + 88, y + 70, subtitle, 17, 500),
        ]
    )


def layer_panel(x, y, w, h, accent, soft, number, title, subtitle):
    return "".join(
        [
            rect(x, y, w, h, "#ffffff", accent, 13, 1.3),
            f'<path d="M {x+13} {y} H {x+w-13} Q {x+w} {y} {x+w} {y+13} '
            f'V {y+80} H {x} V {y+13} Q {x} {y} {x+13} {y} Z" fill="{soft}"/>',
            badge(x + 34, y + 29, 18, accent, str(number), 16),
            txt(x + 88, y + 35, title, 22, 700),
            txt(x + w / 2, y + 65, subtitle, 16, 500, "middle"),
        ]
    )


def build_svg():
    blue = "#2f73d5"
    green = "#4f9858"
    orange = "#d57825"
    rose = "#be6785"
    purple = "#7252a4"
    dark_blue = "#1f4f8f"

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" width="1366" height="1151" '
        'viewBox="0 0 1366 1151" role="img" aria-labelledby="title desc">',
        '<title id="title">MetaBCI实时脑机交互系统结构图</title>',
        '<desc id="desc">由刺激层、采集层、算法层、通信层和交互层构成的MetaBCI闭环系统。</desc>',
        '<defs>',
        '<linearGradient id="core" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#edf6ff"/><stop offset="1" stop-color="#dceaf8"/>'
        '</linearGradient>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC","PingFang SC",Arial,sans-serif;dominant-baseline:alphabetic}</style>',
        '</defs>',
        '<rect width="1366" height="1151" fill="#ffffff"/>',
    ]

    # Core platform and three MetaBCI sub-platforms.
    out += [
        rect(414, 20, 542, 96, "url(#core)", "#72a9df", 15, 1.4),
        txt(685, 60, "MetaBCI 核心平台", 29, 700, "middle"),
        txt(685, 96, "统一目录 · 标准化数据接口", 23, 500, "middle"),
        platform_box(64, 148, 271, green, "brainstim", "视觉刺激与实验范式"),
        platform_box(550, 148, 273, blue, "brainda", "模型训练与在线解码"),
        platform_box(1036, 148, 273, purple, "brainflow", "实时数据流与通信调度"),
        path("M414 68 H201 V147", "#377ac8", 1.8, "7 6", (201, 147, "down")),
        path("M685 116 V147", "#377ac8", 1.8, "7 6", (685, 147, "down")),
        path("M956 68 H1172 V147", "#377ac8", 1.8, "7 6", (1172, 147, "down")),
        path("M201 242 V281 H135 V319", "#377ac8", 1.8, "7 6", (135, 319, "down")),
        path("M685 242 V319", "#377ac8", 1.8, "7 6", (685, 319, "down")),
        path("M1172 242 V281 H959 V319", "#377ac8", 1.8, "7 6", (959, 319, "down")),
        path("M201 281 H1239 V319", "#377ac8", 1.8, "7 6", (1239, 319, "down")),
    ]

    # Layer containers.
    out += [
        layer_panel(20, 320, 228, 623, orange, "#fff0df", 1, "刺激层", "诱发 SSVEP 响应"),
        layer_panel(293, 320, 221, 623, green, "#e8f3e4", 2, "采集层", "脑电采集与同步"),
        layer_panel(565, 320, 229, 623, blue, "#e1effb", 3, "算法层", "在线解码与决策"),
        layer_panel(844, 320, 225, 623, rose, "#f6e1e8", 4, "通信层", "实时通信与调度"),
        layer_panel(1123, 320, 225, 623, purple, "#eee7f6", 5, "交互层", "脑控游戏与可视化"),
    ]

    # Stimulus layer.
    out += [
        card(27, 404, 214, 187, orange, "psychopy", "PsychoPy", ("四频率 SSVEP 刺激", "8.25 / 11.0 Hz", "13.75 / 16.5 Hz", "上 / 下 / 左 / 右")),
        card(27, 612, 214, 128, orange, "sine", "正弦灰度调制", ("刷新同步控制", "频率稳定精确")),
        card(27, 760, 214, 119, orange, "trigger_mark", "Trigger 标记", ("刺激开始同步发送", "上 / 下 / 左 / 右")),
    ]

    # Acquisition layer.
    out += [
        card(300, 404, 207, 115, green, "neuracle", "Neuracle 放大器", ("14 通道 · 250 Hz", "高输入阻抗 · 低噪声")),
        card(300, 528, 207, 109, green, "triggerbox", "TriggerBox 同步", ("刺激事件与 EEG", "精确时间同步")),
        card(300, 646, 207, 111, green, "online", "在线输入（实时）", ("TCP/IP 接口", "实时获取 EEG 数据")),
        card(300, 766, 207, 112, green, "offline", "离线输入（回放）", ("预采集数据回放", "统一数据接口")),
    ]

    # Algorithm layer.
    out += [
        card(572, 404, 215, 121, blue, "algorithm", "算法模型", ("FBTDCA 分类器", "与 FBTRCA 交叉验证", "选取性能更优模型")),
        card(
            572,
            534,
            215,
            105,
            blue,
            "window",
            "递增窗口模型",
            ("125 / 250 / 375 / 500", "采样点", "0.5 / 1.0 / 1.5 / 2.0 s"),
            detail_size=11.5,
            detail_gap=20,
        ),
        card(572, 648, 215, 145, blue, "stopping", "动态停止策略", ("固定起点递增窗口", "边际分数 + 最大置信度", "双阈值判定", "最迟 2 s 强制决策")),
        card(572, 802, 215, 125, blue, "adaptive", "自适应增强机制", ("在线去均值归一化", "动态阈值调整", "数据增强提升泛化能力")),
    ]

    # Communication layer.
    out += [
        card(851, 404, 211, 114, rose, "process", "独立解码进程", ("采集 / 解码 / 通信", "解耦并行运行")),
        card(851, 527, 211, 115, rose, "websocket", "WebSocket", ("Python 后端服务", "实时双向通信")),
        card(851, 651, 211, 116, rose, "json", "JSON 传输", ("统一数据格式", "轻量高效可靠")),
        card(851, 776, 211, 115, rose, "time", "推送内容", ("方向 / 置信度", "决策时间等信息")),
    ]

    # Interaction layer.
    out += [
        card(1130, 404, 211, 132, purple, "canvas", "HTML5 Canvas", ("迷宫 · 贪吃蛇 · 赛车", "三种交互场景")),
        card(1130, 556, 211, 199, purple, "directions", "四分类 → 控制指令", ("上 / 下 / 左 / 右", "映射游戏操作", "实时接收解码结果")),
        card(1130, 775, 211, 152, purple, "confidence", "实时置信度可视化", ("在线 / 离线模式", "自适应画布布局", "跨平台体验")),
    ]

    # Main flow arrows.
    arrow_y = 648
    for x1, x2, label in ((248, 290, "EEG"), (514, 562, "数据"), (794, 841, "决策"), (1069, 1120, "JSON")):
        out.append(
            path(
                f"M{x1} {arrow_y} H{x2}",
                dark_blue,
                2.8,
                arrow_end=(x2, arrow_y, "right"),
            )
        )
        out.append(txt((x1 + x2) / 2, arrow_y - 13, label, 15, 600, "middle", "#263f56"))

    # Application band and closed-loop feedback.
    out += [
        rect(266, 987, 838, 92, "url(#core)", "#72a9df", 15, 1.3),
        embedded_crop("gamepad", 326, 1001, 70, 62),
        txt(685, 1026, "MetaBCIWebGame 应用层", 25, 700, "middle", "#18467d"),
        txt(685, 1058, "仅负责业务流程组织 · 通过统一接口调用 · 不保留算法副本", 18, 500, "middle"),
        path(
            "M1238 943 V1121 H122 V946",
            "#7650d5",
            2.2,
            "8 7",
            (122, 946, "up"),
        ),
        txt(685, 1139, "游戏反馈与下一轮决策控制", 19, 500, "middle"),
    ]

    out.append("</svg>")
    return "\n".join(out)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_svg(), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
