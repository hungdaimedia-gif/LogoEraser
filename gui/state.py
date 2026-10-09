"""Pure Python state and configuration manager for LogoEraser GUI.

Contains zero tkinter dependencies to allow headless execution and complete unit testing.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

from core.preset import (
    FAST_UNSUPPORTED_METHODS,
    IMAGE_EXTENSIONS,
    SUPPORTED_METHODS,
    VIDEO_EXTENSIONS,
    get_default_preset,
    load_preset,
    save_preset,
    to_abs_box,
    to_rel_box,
    validate_preset,
)


class GuiState:
    """Manages files, bounding boxes, method parameters, and video configurations."""

    def __init__(self) -> None:
        default_preset = get_default_preset()
        self.method: str = default_preset["method"]
        self.params: dict[str, Any] = copy.deepcopy(default_preset["params"])
        self.video_mode: str = default_preset["video"]["mode"]
        self.video_encoder: str = default_preset["video"]["encoder"]
        self.video_crf: int = int(default_preset["video"]["crf"])
        self.video_copy_audio: bool = bool(default_preset["video"]["copy_audio"])

        self.boxes: list[dict[str, Any]] = []
        self.selected_box_idx: int | None = None

        self.files: list[str] = []
        self.selected_file_idx: int | None = None

        self.output_dir: str = ""
        self.overwrite: bool = False

    # ---------------- File Management ----------------

    def add_files(self, paths: list[str]) -> int:
        """Add files to queue if they have valid extensions. Returns count of added files."""
        added = 0
        all_exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        for p in paths:
            norm = os.path.abspath(p)
            if not os.path.isfile(norm):
                continue
            ext = os.path.splitext(norm)[1].lower()
            if ext in all_exts and norm not in self.files:
                self.files.append(norm)
                added += 1
        if self.files and self.selected_file_idx is None:
            self.selected_file_idx = 0
        return added

    def remove_file(self, idx: int) -> str | None:
        """Remove file from list by index. Does NOT delete files on disk."""
        if 0 <= idx < len(self.files):
            removed = self.files.pop(idx)
            if self.selected_file_idx is not None:
                if len(self.files) == 0:
                    self.selected_file_idx = None
                elif self.selected_file_idx >= len(self.files):
                    self.selected_file_idx = len(self.files) - 1
            return removed
        return None

    def clear_files(self) -> None:
        """Clear all files from the queue."""
        self.files.clear()
        self.selected_file_idx = None

    def select_file(self, idx: int | None) -> str | None:
        """Select a file by index. Returns file path or None."""
        if idx is not None and 0 <= idx < len(self.files):
            self.selected_file_idx = idx
            return self.files[idx]
        self.selected_file_idx = None
        return None

    def get_selected_file(self) -> str | None:
        """Get path of currently selected file."""
        if self.selected_file_idx is not None and 0 <= self.selected_file_idx < len(self.files):
            return self.files[self.selected_file_idx]
        return None

    # ---------------- Box Management ----------------

    def add_box_abs(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> int:
        """Add a box given absolute image coordinates, converting to relative. Returns new index."""
        if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
            return -1
        abs_box = {"x": x, "y": y, "w": w, "h": h, "unit": "absolute"}
        rel_box = to_rel_box(abs_box, img_w, img_h)
        self.boxes.append(rel_box)
        new_idx = len(self.boxes) - 1
        self.selected_box_idx = new_idx
        return new_idx

    def update_box_abs(self, idx: int, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> bool:
        """Update a box at index using absolute image coordinates."""
        if not (0 <= idx < len(self.boxes)):
            return False
        if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
            return False
        abs_box = {"x": x, "y": y, "w": w, "h": h, "unit": "absolute"}
        rel_box = to_rel_box(abs_box, img_w, img_h)
        self.boxes[idx] = rel_box
        return True

    def remove_box(self, idx: int) -> bool:
        """Remove box at index. In-memory data only."""
        if 0 <= idx < len(self.boxes):
            self.boxes.pop(idx)
            if self.selected_box_idx == idx:
                self.selected_box_idx = None
            elif self.selected_box_idx is not None and self.selected_box_idx > idx:
                self.selected_box_idx -= 1
            return True
        return False

    def clear_boxes(self) -> None:
        """Clear all boxes."""
        self.boxes.clear()
        self.selected_box_idx = None

    def get_abs_boxes(self, img_w: int, img_h: int) -> list[dict[str, Any]]:
        """Get all boxes converted to absolute coordinates for given image dimensions."""
        if img_w <= 0 or img_h <= 0:
            return []
        return [to_abs_box(b, img_w, img_h) for b in self.boxes]

    # ---------------- Method & Parameters ----------------

    def set_method(self, method: str) -> None:
        """Set removal method if supported."""
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method '{method}'. Supported: {SUPPORTED_METHODS}")
        self.method = method

    def set_param(self, key: str, value: Any) -> None:
        """Set a parameter value.

        Chỉ nhận key có trong preset: key lạ bị validate_preset bỏ qua trong im lặng,
        từng làm ô "Bán kính inpaint" ghi vào 'inpaint_radius' mà không có tác dụng.
        """
        if key not in self.params:
            raise KeyError(f"Unknown preset parameter '{key}'. Known: {sorted(self.params)}")
        self.params[key] = value

    def set_video_mode(self, mode: str) -> None:
        """Set video mode ('fast' or 'quality')."""
        if mode not in ("fast", "quality"):
            raise ValueError(f"Invalid video mode '{mode}'. Must be 'fast' or 'quality'.")
        self.video_mode = mode

    def get_active_param_names(self) -> list[str]:
        """Return parameter keys applicable to the current method (for dynamic GUI controls)."""
        # dilate/feather định hình mask nên áp dụng cho mọi method dùng mask
        # (mọi method trừ overlay — overlay dán đúng khung box).
        if self.method in ("inpaint_fast", "inpaint_hq"):
            return ["dilate", "feather", "radius"]
        if self.method in ("clone_smooth", "inpaint_ai"):
            return ["dilate", "feather"]
        if self.method == "blur":
            return ["dilate", "feather", "blur_ksize"]
        if self.method == "mosaic":
            return ["dilate", "feather", "mosaic_block"]
        if self.method == "solid":
            return ["dilate", "feather", "solid_color"]
        if self.method == "overlay":
            return ["overlay_path", "overlay_opacity"]
        return []

    # ---------------- Preset Import/Export ----------------

    def to_preset_dict(self) -> dict[str, Any]:
        """Convert current state to a validated preset dictionary."""
        data = {
            "version": 1,
            "method": self.method,
            "boxes": copy.deepcopy(self.boxes),
            "params": copy.deepcopy(self.params),
            "video": {
                "mode": self.video_mode,
                "encoder": self.video_encoder,
                "crf": self.video_crf,
                "copy_audio": self.video_copy_audio,
            },
        }
        return validate_preset(data)

    def from_preset_dict(self, data: dict[str, Any]) -> None:
        """Load state from preset dictionary."""
        valid = validate_preset(data)
        self.method = valid["method"]
        self.boxes = copy.deepcopy(valid["boxes"])
        self.selected_box_idx = 0 if self.boxes else None
        self.params = copy.deepcopy(valid["params"])
        self.video_mode = valid["video"]["mode"]
        self.video_encoder = valid["video"]["encoder"]
        self.video_crf = int(valid["video"]["crf"])
        self.video_copy_audio = bool(valid["video"]["copy_audio"])

    def save_preset_file(self, path: str) -> None:
        """Save current preset to a JSON file."""
        preset_dict = self.to_preset_dict()
        save_preset(preset_dict, path)

    def load_preset_file(self, path: str) -> None:
        """Load preset from a JSON file and apply to state."""
        preset_dict = load_preset(path)
        self.from_preset_dict(preset_dict)

    # ---------------- Pre-run Validation ----------------

    def validate_for_run(self) -> list[str]:
        """Check all prerequisites before running batch processing.

        Returns a list of human-readable error messages (empty if ready).
        """
        errors: list[str] = []

        if not self.files:
            errors.append("Danh sách tệp tin đang trống. Vui lòng thêm ít nhất một ảnh hoặc video.")

        if not self.boxes:
            errors.append("Chưa chọn vùng logo nào. Vui lòng vẽ ít nhất một khung bao trên hình ảnh.")

        # Review 1b-r2 Note 4: clone_smooth + fast video mode forbidden
        has_videos = any(os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS for f in self.files)
        if has_videos and self.method in FAST_UNSUPPORTED_METHODS and self.video_mode == "fast":
            errors.append(
                f"Phương thức '{self.method}' không hỗ trợ chế độ 'fast' trên video. "
                "Vui lòng chuyển chế độ video sang 'quality' hoặc chọn phương thức khác."
            )

        if self.method == "inpaint_ai":
            from core.ai_inpaint import missing_model_message, model_available

            if not model_available():
                errors.append(missing_model_message())

        # Overlay file check
        if self.method == "overlay":
            ov_path = self.params.get("overlay_path", "")
            if not ov_path or not os.path.isfile(ov_path):
                errors.append(f"Không tìm thấy tệp ảnh phủ (overlay): '{ov_path}'")

        # Validate preset schema
        try:
            self.to_preset_dict()
        except Exception as e:
            errors.append(f"Cấu hình tham số không hợp lệ: {e}")

        return errors
