from __future__ import annotations

import re

from .models import NODE_TYPES


BUTTON_WORDS = ("button", "btn", "ok", "confirm", "cancel", "close", "submit", "按钮", "确定", "取消", "关闭")
LIST_WORDS = ("list", "列表", "rank", "ranking", "scrolllist")
SCROLL_WORDS = ("scroll", "scrollview", "scrollbar", "滚动")
GRID_WORDS = ("grid", "matrix", "网格", "九宫格")
INPUT_WORDS = ("input", "field", "search", "textbox", "edit", "输入", "搜索")
TOGGLE_WORDS = ("toggle", "checkbox", "check", "radio", "switch", "勾选", "开关")
SLIDER_WORDS = ("slider", "progress", "bar", "gauge", "进度", "滑动")
BACKGROUND_WORDS = ("background", "bg", "backdrop", "mask", "底", "背景")
TEXT_WORDS = ("text", "label", "title", "name", "desc", "copy", "文本", "标题", "说明")
ICON_WORDS = ("icon", "ico", "badge", "logo", "图标")
CONTAINER_WORDS = ("panel", "dialog", "popup", "window", "modal", "container", "group", "box", "面板", "弹窗", "容器")
IMAGE_WORDS = ("image", "img", "pic", "photo", "avatar", "sprite", "图片", "头像")


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"\s+", " ", name.strip().lower())


def is_comment_layer(name: str | None) -> bool:
    return bool(name and name.strip().startswith("#"))


def infer_node_type(name: str | None, kind: str | None = None, is_group: bool = False) -> tuple[str, float]:
    normalized = normalize_name(name)
    layer_kind = (kind or "").lower()

    if layer_kind in {"type", "text"}:
        return "Text", 0.92
    if any(word in normalized for word in BUTTON_WORDS):
        return "Button", 0.88
    if any(word in normalized for word in LIST_WORDS):
        return "List", 0.86
    if any(word in normalized for word in SCROLL_WORDS):
        return "ScrollView", 0.84
    if any(word in normalized for word in GRID_WORDS):
        return "Grid", 0.84
    if any(word in normalized for word in INPUT_WORDS):
        return "Input", 0.84
    if any(word in normalized for word in TOGGLE_WORDS):
        return "Toggle", 0.82
    if any(word in normalized for word in SLIDER_WORDS):
        return "Slider", 0.82
    if any(word in normalized for word in BACKGROUND_WORDS):
        return "Background", 0.78
    if any(word in normalized for word in TEXT_WORDS):
        return "Text", 0.76
    if any(word in normalized for word in ICON_WORDS):
        return "Icon", 0.74
    if any(word in normalized for word in CONTAINER_WORDS):
        return "Container", 0.78
    if any(word in normalized for word in IMAGE_WORDS):
        return "Image", 0.7
    if is_group:
        return "Container", 0.58
    if layer_kind in {"pixel", "smartobject", "shape", "solidcolorfill", "gradientfill"}:
        return "Image", 0.55
    return "Unknown", 0.45


def infer_role(name: str | None, node_type: str) -> str | None:
    normalized = normalize_name(name)
    if not normalized:
        return None
    if "close" in normalized or "关闭" in normalized or normalized in {"x", "btn_x"}:
        return "close"
    if "ok" in normalized or "confirm" in normalized or "确定" in normalized:
        return "primary_action"
    if "cancel" in normalized or "取消" in normalized:
        return "secondary_action"
    if "back" in normalized or "返回" in normalized:
        return "back"
    if node_type == "Background":
        return "background"
    if node_type == "List":
        return "collection"
    return None


def coerce_node_type(value: str | None) -> str:
    if value in NODE_TYPES:
        return value
    return "Unknown"
