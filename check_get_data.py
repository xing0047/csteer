import argparse
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

BLINK_CHECK_SUBSETS = {
    "RRF": "Relative_Reflectance",
    "RDP": "Relative_Depth",
    "FCO": "Functional_Correspondence",
}


def _is_pil_image(x: Any) -> bool:
    try:
        from PIL import Image as PILImage

        return isinstance(x, PILImage.Image)
    except Exception:
        return False


def _short(s: Any, n: int = 120) -> str:
    txt = repr(s)
    return txt if len(txt) <= n else (txt[: n - 3] + "...")


def _check_path_exists(path: str, kind: str) -> Optional[str]:
    if not isinstance(path, str) or not path:
        return f"{kind} is missing or not a string: {path!r}"
    if not os.path.exists(path):
        return f"{kind} does not exist: {path}"
    return None


def _require_keys(item: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    errs = []
    for k in keys:
        if k not in item:
            errs.append(f"missing key: {k}")
    return errs


def _infer_image_source(item: Dict[str, Any]) -> str:
    if "pil_images" in item:
        return "pil_images"
    if "pil_image" in item:
        return "pil_image"
    if "image_path" in item:
        return "image_path"
    if "video_path" in item:
        return "video_path"
    return "unknown"


def _load_first_image_for_overlay_check(
    eval_type: str, item: Dict[str, Any]
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Returns (PIL.Image or None, meta) where meta includes the image origin.
    Designed to load exactly one representative image for each dataset type.
    """
    meta: Dict[str, Any] = {"type": eval_type, "image_source": _infer_image_source(item)}
    src = meta["image_source"]

    try:
        from PIL import Image as PILImage
    except Exception as ex:
        meta["error"] = f"PIL import failed: {ex!r}"
        return None, meta

    if src == "pil_image":
        im = item.get("pil_image")
        meta["origin"] = "item.pil_image"
        return im if _is_pil_image(im) else None, meta

    if src == "pil_images":
        ims = item.get("pil_images")
        meta["origin"] = "item.pil_images[0]"
        if isinstance(ims, list) and len(ims) > 0 and _is_pil_image(ims[0]):
            return ims[0], meta
        return None, meta

    if src == "image_path":
        path = item.get("image_path")
        meta["origin"] = "item.image_path"
        meta["image_path"] = path
        if isinstance(path, str) and os.path.exists(path):
            try:
                return PILImage.open(path).convert("RGB"), meta
            except Exception as ex:
                meta["error"] = f"failed to open image_path: {ex!r}"
                return None, meta
        return None, meta

    if src == "video_path":
        # wrapper expands this under ../DATA/Inst-It-Bench/<video_path>
        video_path = item.get("video_path")
        meta["origin"] = "item.video_path -> first frame under ../DATA/Inst-It-Bench/<video_path>/"
        meta["video_path"] = video_path
        if not isinstance(video_path, str) or not video_path:
            return None, meta
        video_root = os.path.join("..", "DATA", "Inst-It-Bench", video_path)
        meta["video_root"] = video_root
        if not os.path.isdir(video_root):
            return None, meta
        try:
            frames = sorted(os.listdir(video_root))
        except Exception as ex:
            meta["error"] = f"failed to list frames: {ex!r}"
            return None, meta
        if not frames:
            return None, meta
        first = os.path.join(video_root, frames[0])
        meta["first_frame_path"] = first
        if not os.path.exists(first):
            return None, meta
        try:
            return PILImage.open(first).convert("RGB"), meta
        except Exception as ex:
            meta["error"] = f"failed to open first frame: {ex!r}"
            return None, meta

    meta["error"] = "unknown image source"
    return None, meta


def _pil_resample_lanczos():
    from PIL import Image as PILImage

    if hasattr(PILImage, "Resampling"):
        return PILImage.Resampling.LANCZOS
    return PILImage.LANCZOS


def _stitch_pil_grid(
    images: List[Any],
    *,
    cell_w: int = 320,
    cell_h: int = 320,
    cols: Optional[int] = None,
    draw_borders: bool = False,
    border_width: int = 6,
    border_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Arrange PIL images in a grid (same cell size, thumbnail + center paste).
    Returns (PIL.Image or None, error_message or None).
    """
    try:
        from PIL import Image as PILImage
    except Exception as ex:
        return None, f"PIL import failed: {ex!r}"

    valid: List[Any] = []
    for im in images:
        if _is_pil_image(im):
            valid.append(im.convert("RGB"))
    n = len(valid)
    if n == 0:
        return None, "no valid PIL images"

    import math

    if cols is None:
        cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    canvas = PILImage.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = None
    if draw_borders:
        try:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(canvas)
        except Exception:
            draw = None
    for i, im in enumerate(valid):
        thumb = im.copy()
        thumb.thumbnail((cell_w, cell_h), _pil_resample_lanczos())
        r, c = divmod(i, cols)
        x = c * cell_w + (cell_w - thumb.width) // 2
        y = r * cell_h + (cell_h - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        if draw is not None and border_width > 0:
            # Draw border around the pasted thumbnail (not the whole cell).
            x1, y1 = x, y
            x2, y2 = x + thumb.width - 1, y + thumb.height - 1
            for k in range(border_width):
                draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=border_color, width=1)
    return canvas, None


def _load_all_video_frames_as_pil(video_root: str) -> Tuple[List[Any], Optional[str]]:
    """Load every image file in video_root (sorted) as RGB PIL images."""
    try:
        from PIL import Image as PILImage
    except Exception as ex:
        return [], f"PIL import failed: {ex!r}"

    if not os.path.isdir(video_root):
        return [], f"not a directory: {video_root}"
    try:
        names = sorted(os.listdir(video_root))
    except Exception as ex:
        return [], str(ex)

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    out: List[Any] = []
    for name in names:
        low = name.lower()
        if not any(low.endswith(e) for e in exts):
            continue
        p = os.path.join(video_root, name)
        if not os.path.isfile(p):
            continue
        try:
            out.append(PILImage.open(p).convert("RGB"))
        except Exception:
            continue
    return out, None if out else "no readable frame images"


def _open_rgb_safe(path: str):
    from PIL import Image as PILImage

    if not isinstance(path, str) or not path or not os.path.exists(path):
        return None
    try:
        return PILImage.open(path).convert("RGB")
    except Exception:
        return None


def _save_vip_source_vs_bbox(
    tname: str,
    save_dir: str,
    *,
    video_cell_w: int,
    video_cell_h: int,
) -> Dict[str, Any]:
    """ViP-Bench: save one PNG with source (noref) left, bbox (annotated) right."""
    from behaviors import get_vip_bench_data

    os.makedirs(save_dir, exist_ok=True)
    safe_name = tname.replace("/", "_")
    out: Dict[str, Any] = {
        "type": tname,
        "saved": None,
        "error": None,
        "source_path": None,
        "bbox_path": None,
    }
    try:
        item_src = get_vip_bench_data(noref=True)[0]
        item_bbox = get_vip_bench_data(noref=False)[0]
        p_src = item_src.get("image_path")
        p_bbox = item_bbox.get("image_path")
        out["source_path"] = p_src
        out["bbox_path"] = p_bbox
        im_src = _open_rgb_safe(p_src) if p_src else None
        im_bbox = _open_rgb_safe(p_bbox) if p_bbox else None
        if im_src is None:
            out["error"] = f"failed to open source: {p_src!r}"
            return out
        if im_bbox is None:
            out["error"] = f"failed to open bbox: {p_bbox!r}"
            return out
        grid, err = _stitch_pil_grid(
            [im_src, im_bbox],
            cols=2,
            cell_w=max(video_cell_w, 400),
            cell_h=max(video_cell_h, 400),
            draw_borders=True,
        )
        if err or grid is None:
            out["error"] = err or "stitch failed"
            return out
        path = os.path.join(save_dir, f"{safe_name}_source_vs_bbox.png")
        grid.save(path)
        out["saved"] = path
    except Exception as ex:
        out["error"] = repr(ex)
    return out


def _overlay_check_vip_dual() -> Dict[str, Any]:
    """Run overlay heuristic on ViP source vs bbox (first sample)."""
    from behaviors import get_vip_bench_data

    item_src = get_vip_bench_data(noref=True)[0]
    item_bbox = get_vip_bench_data(noref=False)[0]
    p_src = item_src.get("image_path")
    p_bbox = item_bbox.get("image_path")
    im_src = _open_rgb_safe(p_src) if p_src else None
    im_bbox = _open_rgb_safe(p_bbox) if p_bbox else None
    likely_src, stats_src = _overlay_likely_boxnum(im_src)
    likely_bbox, stats_bbox = _overlay_likely_boxnum(im_bbox)
    return {
        "source_path": p_src,
        "bbox_path": p_bbox,
        "source": {"likely": likely_src, "stats": stats_src},
        "bbox": {"likely": likely_bbox, "stats": stats_bbox},
    }


def _save_dual_image_path(
    *,
    tname: str,
    save_dir: str,
    getter_noref_true,
    getter_noref_false,
    left_label: str,
    right_label: str,
    video_cell_w: int,
    video_cell_h: int,
) -> Dict[str, Any]:
    """
    Save one PNG with (noref=True) image left and (noref=False) image right.
    This is for datasets whose loader returns items with image_path.
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = tname.replace("/", "_")
    out: Dict[str, Any] = {
        "type": tname,
        "saved": None,
        "error": None,
        "left_label": left_label,
        "right_label": right_label,
        "left_path": None,
        "right_path": None,
    }
    try:
        item_left = getter_noref_true(noref=True)[0]
        item_right = getter_noref_false(noref=False)[0]
        p_left = item_left.get("image_path")
        p_right = item_right.get("image_path")
        out["left_path"] = p_left
        out["right_path"] = p_right

        im_left = _open_rgb_safe(p_left) if p_left else None
        im_right = _open_rgb_safe(p_right) if p_right else None
        if im_left is None:
            out["error"] = f"failed to open left ({left_label}): {p_left!r}"
            return out
        if im_right is None:
            out["error"] = f"failed to open right ({right_label}): {p_right!r}"
            return out

        grid, err = _stitch_pil_grid(
            [im_left, im_right],
            cols=2,
            cell_w=max(video_cell_w, 400),
            cell_h=max(video_cell_h, 400),
            draw_borders=True,
        )
        if err or grid is None:
            out["error"] = err or "stitch failed"
            return out

        path = os.path.join(save_dir, f"{safe_name}_{left_label}_vs_{right_label}.png")
        grid.save(path)
        out["saved"] = path
    except Exception as ex:
        out["error"] = repr(ex)
    return out


def _overlay_check_dual_image_path(*, getter_noref_true, getter_noref_false) -> Dict[str, Any]:
    item_left = getter_noref_true(noref=True)[0]
    item_right = getter_noref_false(noref=False)[0]
    p_left = item_left.get("image_path")
    p_right = item_right.get("image_path")
    im_left = _open_rgb_safe(p_left) if p_left else None
    im_right = _open_rgb_safe(p_right) if p_right else None
    likely_left, stats_left = _overlay_likely_boxnum(im_left)
    likely_right, stats_right = _overlay_likely_boxnum(im_right)
    return {
        "left_path": p_left,
        "right_path": p_right,
        "left": {"likely": likely_left, "stats": stats_left},
        "right": {"likely": likely_right, "stats": stats_right},
    }


def _first_frame_path_from_video_root(video_root: str) -> Optional[str]:
    if not isinstance(video_root, str) or not os.path.isdir(video_root):
        return None
    try:
        names = sorted(os.listdir(video_root))
    except Exception:
        return None
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    for name in names:
        low = name.lower()
        if not low.endswith(exts):
            continue
        p = os.path.join(video_root, name)
        if os.path.isfile(p):
            return p
    return None


def _save_dual_video_mosaic(
    *,
    tname: str,
    save_dir: str,
    getter_noref_true,
    getter_noref_false,
    left_label: str,
    right_label: str,
    video_cell_w: int,
    video_cell_h: int,
    video_max_frames: int,
) -> Dict[str, Any]:
    """
    Inst-It video: stitch all frames for noref and refer, then place side-by-side.
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_name = tname.replace("/", "_")
    out: Dict[str, Any] = {
        "type": tname,
        "saved": None,
        "error": None,
        "left_label": left_label,
        "right_label": right_label,
        "left_video_path": None,
        "right_video_path": None,
        "left_video_root": None,
        "right_video_root": None,
        "left_num_frames_used": None,
        "right_num_frames_used": None,
    }
    try:
        item_left = getter_noref_true(noref=True)[0]
        item_right = getter_noref_false(noref=False)[0]
        vp_left = item_left.get("video_path")
        vp_right = item_right.get("video_path")
        out["left_video_path"] = vp_left
        out["right_video_path"] = vp_right

        root_left = os.path.join("..", "DATA", "Inst-It-Bench", vp_left) if vp_left else None
        root_right = os.path.join("..", "DATA", "Inst-It-Bench", vp_right) if vp_right else None
        out["left_video_root"] = root_left
        out["right_video_root"] = root_right

        if not root_left or not os.path.isdir(root_left):
            out["error"] = f"left video_root missing: {root_left!r}"
            return out
        if not root_right or not os.path.isdir(root_right):
            out["error"] = f"right video_root missing: {root_right!r}"
            return out

        frames_left, err_left = _load_all_video_frames_as_pil(root_left)
        frames_right, err_right = _load_all_video_frames_as_pil(root_right)
        if err_left:
            out["error"] = f"left load frames failed: {err_left}"
            return out
        if err_right:
            out["error"] = f"right load frames failed: {err_right}"
            return out

        if video_max_frames > 0 and len(frames_left) > video_max_frames:
            frames_left = frames_left[:video_max_frames]
        if video_max_frames > 0 and len(frames_right) > video_max_frames:
            frames_right = frames_right[:video_max_frames]
        out["left_num_frames_used"] = len(frames_left)
        out["right_num_frames_used"] = len(frames_right)

        mosaic_left, e1 = _stitch_pil_grid(
            frames_left,
            cell_w=video_cell_w,
            cell_h=video_cell_h,
            draw_borders=True,
        )
        mosaic_right, e2 = _stitch_pil_grid(
            frames_right,
            cell_w=video_cell_w,
            cell_h=video_cell_h,
            draw_borders=True,
        )
        if e1 or mosaic_left is None:
            out["error"] = f"left stitch failed: {e1 or 'unknown'}"
            return out
        if e2 or mosaic_right is None:
            out["error"] = f"right stitch failed: {e2 or 'unknown'}"
            return out

        side_by_side, e3 = _stitch_pil_grid(
            [mosaic_left, mosaic_right],
            cols=2,
            cell_w=max(mosaic_left.width, mosaic_right.width),
            cell_h=max(mosaic_left.height, mosaic_right.height),
            draw_borders=True,
        )
        if e3 or side_by_side is None:
            out["error"] = f"final stitch failed: {e3 or 'unknown'}"
            return out

        path = os.path.join(save_dir, f"{safe_name}_{left_label}_vs_{right_label}.png")
        side_by_side.save(path)
        out["saved"] = path
        return out
    except Exception as ex:
        out["error"] = repr(ex)
        return out


def _overlay_check_dual_video(*, getter_noref_true, getter_noref_false) -> Dict[str, Any]:
    """
    Heuristic overlay check for video: only checks the first frame of each root.
    """
    item_left = getter_noref_true(noref=True)[0]
    item_right = getter_noref_false(noref=False)[0]
    vp_left = item_left.get("video_path")
    vp_right = item_right.get("video_path")

    root_left = os.path.join("..", "DATA", "Inst-It-Bench", vp_left) if vp_left else None
    root_right = os.path.join("..", "DATA", "Inst-It-Bench", vp_right) if vp_right else None
    first_left = _first_frame_path_from_video_root(root_left) if root_left else None
    first_right = _first_frame_path_from_video_root(root_right) if root_right else None

    im_left = _open_rgb_safe(first_left) if first_left else None
    im_right = _open_rgb_safe(first_right) if first_right else None
    likely_left, stats_left = _overlay_likely_boxnum(im_left)
    likely_right, stats_right = _overlay_likely_boxnum(im_right)
    return {
        "left_video_path": vp_left,
        "right_video_path": vp_right,
        "left_video_root": root_left,
        "right_video_root": root_right,
        "left_first_frame": first_left,
        "right_first_frame": first_right,
        "left": {"likely": likely_left, "stats": stats_left},
        "right": {"likely": likely_right, "stats": stats_right},
    }

def _save_preview_for_type(
    tname: str,
    item0: Dict[str, Any],
    save_dir: str,
    *,
    video_cell_w: int,
    video_cell_h: int,
    video_max_frames: int,
) -> Dict[str, Any]:
    """
    Save one preview image per dataset type for manual inspection.
    Video: all frames (or first video_max_frames) stitched into one grid.
    """
    from PIL import Image as PILImage

    os.makedirs(save_dir, exist_ok=True)
    safe_name = tname.replace("/", "_")
    out: Dict[str, Any] = {"type": tname, "saved": None, "error": None}

    if tname == "vip_image_oe_qa":
        return _save_vip_source_vs_bbox(
            tname, save_dir, video_cell_w=video_cell_w, video_cell_h=video_cell_h
        )

    if tname == "gar_image_detail_oe_qa":
        from behaviors import get_gar_caption_detailed_data

        return _save_dual_image_path(
            tname=tname,
            save_dir=save_dir,
            getter_noref_true=get_gar_caption_detailed_data,
            getter_noref_false=get_gar_caption_detailed_data,
            left_label="noref",
            right_label="refer",
            video_cell_w=video_cell_w,
            video_cell_h=video_cell_h,
        )

    if tname in ("inst_it_image_mc_qa", "inst_it_image_oe_qa"):
        from behaviors import get_inst_it_image_mc_data, get_inst_it_image_oe_data

        getter = get_inst_it_image_mc_data if tname == "inst_it_image_mc_qa" else get_inst_it_image_oe_data
        return _save_dual_image_path(
            tname=tname,
            save_dir=save_dir,
            getter_noref_true=getter,
            getter_noref_false=getter,
            left_label="raw",
            right_label="vpt",
            video_cell_w=video_cell_w,
            video_cell_h=video_cell_h,
        )

    if tname in ("inst_it_video_mc_qa", "inst_it_video_oe_qa"):
        from behaviors import get_inst_it_video_mc_data, get_inst_it_video_oe_data

        getter = get_inst_it_video_mc_data if tname == "inst_it_video_mc_qa" else get_inst_it_video_oe_data
        return _save_dual_video_mosaic(
            tname=tname,
            save_dir=save_dir,
            getter_noref_true=getter,
            getter_noref_false=getter,
            left_label="raw",
            right_label="vpt",
            video_cell_w=video_cell_w,
            video_cell_h=video_cell_h,
            video_max_frames=video_max_frames,
        )

    if tname == "blink_image_mc_qa":
        # Only check selected BLINK subsets: RRF, RDP, FCO.
        out_multi: Dict[str, Any] = {"type": tname, "saved": {}, "error": None}
        try:
            # item0 is ignored; we use the already loaded items from caller via a hack:
            # caller passes item0, but we can re-import getter and load once here.
            from behaviors import get_blink_image_mc_data

                items = get_blink_image_mc_data(subsets=list(BLINK_CHECK_SUBSETS.values()))
            for abbr, subset_name in BLINK_CHECK_SUBSETS.items():
                match = next((it for it in items if it.get("type") == subset_name), None)
                if match is None:
                    out_multi["saved"][abbr] = None
                    continue
                ims = match.get("pil_images")
                pil_list = [x for x in ims] if isinstance(ims, list) else []
                pil_list = [x for x in pil_list if _is_pil_image(x)]
                if not pil_list:
                    out_multi["saved"][abbr] = None
                    continue
                grid, err = _stitch_pil_grid(
                    pil_list,
                    cell_w=video_cell_w,
                    cell_h=video_cell_h,
                    draw_borders=True,
                )
                if err or grid is None:
                    out_multi["saved"][abbr] = None
                    continue
                path = os.path.join(save_dir, f"{tname}_{abbr}_pil_images_grid.png")
                grid.save(path)
                out_multi["saved"][abbr] = path
        except Exception as ex:
            out_multi["error"] = repr(ex)
        return out_multi

    src = _infer_image_source(item0)

    try:
        if src == "pil_image":
            im = item0.get("pil_image")
            if not _is_pil_image(im):
                out["error"] = "pil_image missing or not PIL"
                return out
            path = os.path.join(save_dir, f"{safe_name}.png")
            im.convert("RGB").save(path)
            out["saved"] = path
            return out

        if src == "pil_images":
            ims = item0.get("pil_images")
            if not isinstance(ims, list) or not ims:
                out["error"] = "pil_images empty"
                return out
            pil_list = [x for x in ims if _is_pil_image(x)]
            if not pil_list:
                out["error"] = "no PIL in pil_images"
                return out
            grid, err = _stitch_pil_grid(pil_list, cell_w=video_cell_w, cell_h=video_cell_h)
            if err or grid is None:
                out["error"] = err or "stitch failed"
                return out
            path = os.path.join(save_dir, f"{safe_name}_pil_images_grid.png")
            grid.save(path)
            out["saved"] = path
            out["num_subimages"] = len(pil_list)
            return out

        if src == "image_path":
            p = item0.get("image_path")
            if not isinstance(p, str) or not os.path.exists(p):
                out["error"] = f"bad image_path: {p!r}"
                return out
            im = PILImage.open(p).convert("RGB")
            path = os.path.join(save_dir, f"{safe_name}.png")
            im.save(path)
            out["saved"] = path
            out["source_path"] = p
            return out

        if src == "video_path":
            vp = item0.get("video_path")
            if not isinstance(vp, str) or not vp:
                out["error"] = "bad video_path"
                return out
            video_root = os.path.join("..", "DATA", "Inst-It-Bench", vp)
            frames, err = _load_all_video_frames_as_pil(video_root)
            if err:
                out["error"] = err
                out["video_root"] = video_root
                return out
            if video_max_frames > 0 and len(frames) > video_max_frames:
                frames = frames[:video_max_frames]
            grid, err2 = _stitch_pil_grid(frames, cell_w=video_cell_w, cell_h=video_cell_h)
            if err2 or grid is None:
                out["error"] = err2 or "stitch failed"
                return out
            path = os.path.join(save_dir, f"{safe_name}_all_frames_grid.png")
            grid.save(path)
            out["saved"] = path
            out["video_root"] = video_root
            out["num_frames_used"] = len(frames)
            return out

        out["error"] = f"unknown source: {src}"
    except Exception as ex:
        out["error"] = repr(ex)
    return out


def _overlay_likely_boxnum(im) -> Tuple[bool, Dict[str, Any]]:
    """
    Heuristic: detect presence of highly-saturated red/blue/green pixels consistent with
    overlay_visual_prompt(..., format="boxnum") which draws:
      - rectangle outlines in red/blue/green
      - filled background boxes in same colors
      - white digits

    Returns (likely, stats).
    """
    stats: Dict[str, Any] = {}
    if im is None or not _is_pil_image(im):
        return False, {"error": f"not a PIL.Image: {type(im)}"}

    try:
        import numpy as np
    except Exception as ex:
        return False, {"error": f"numpy import failed: {ex!r}"}

    arr = np.asarray(im.convert("RGB"))
    h, w = arr.shape[:2]
    stats["width"] = int(w)
    stats["height"] = int(h)
    if h == 0 or w == 0:
        return False, {"error": "empty image"}

    # Sample a central crop to reduce false positives from colorful backgrounds at borders.
    y0, y1 = int(h * 0.10), int(h * 0.90)
    x0, x1 = int(w * 0.10), int(w * 0.90)
    crop = arr[y0:y1, x0:x1, :]
    if crop.size == 0:
        crop = arr

    r = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    b = crop[:, :, 2].astype(np.int16)

    # Near-pure-ish saturated colors (outline/background)
    red = (r >= 220) & (g <= 80) & (b <= 80)
    blue = (b >= 220) & (r <= 80) & (g <= 120)
    green = (g >= 190) & (r <= 120) & (b <= 120)

    total = crop.shape[0] * crop.shape[1]
    red_ratio = float(red.sum()) / float(total)
    blue_ratio = float(blue.sum()) / float(total)
    green_ratio = float(green.sum()) / float(total)

    stats["red_ratio"] = red_ratio
    stats["blue_ratio"] = blue_ratio
    stats["green_ratio"] = green_ratio
    stats["saturated_sum_ratio"] = red_ratio + blue_ratio + green_ratio

    # Decision rule:
    # - Require at least two of the three colors present above tiny threshold OR
    # - One color very prominent (large filled label box or thick outlines).
    present = [
        ("red", red_ratio),
        ("blue", blue_ratio),
        ("green", green_ratio),
    ]
    present_count = sum(1 for _, rr in present if rr >= 0.0005)  # 0.05%
    stats["present_color_count"] = int(present_count)

    likely = (present_count >= 2 and stats["saturated_sum_ratio"] >= 0.0012) or (
        max(red_ratio, blue_ratio, green_ratio) >= 0.003
    )
    stats["likely_rule"] = ">=2 colors & sum>=0.12% OR max>=0.3%"
    return bool(likely), stats


def _check_item_schema(eval_type: str, item: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    src = _infer_image_source(item)

    # Common sanity checks
    if not isinstance(item, dict):
        return [f"item is not a dict: {type(item)}"]

    # Image / video checks by source
    if src == "pil_image":
        if not _is_pil_image(item.get("pil_image")):
            errs.append(f"pil_image is not a PIL.Image: {type(item.get('pil_image'))}")
    elif src == "pil_images":
        images = item.get("pil_images")
        if not isinstance(images, list) or len(images) == 0:
            errs.append(f"pil_images must be a non-empty list, got: {_short(images)}")
        else:
            for i, im in enumerate(images[:4]):
                if not _is_pil_image(im):
                    errs.append(f"pil_images[{i}] is not a PIL.Image: {type(im)}")
    elif src == "image_path":
        e = _check_path_exists(item.get("image_path"), "image_path")
        if e:
            errs.append(e)
    elif src == "video_path":
        # wrapper expands this under ../DATA/Inst-It-Bench/<video_path>
        video_path = item.get("video_path")
        if not isinstance(video_path, str) or not video_path:
            errs.append(f"video_path missing or not a string: {video_path!r}")
        else:
            video_root = os.path.join("..", "DATA", "Inst-It-Bench", video_path)
            if not os.path.isdir(video_root):
                errs.append(f"video frames dir does not exist: {video_root}")
            else:
                try:
                    frames = sorted(os.listdir(video_root))
                except Exception as ex:
                    errs.append(f"failed to list frames in {video_root}: {ex}")
                else:
                    if len(frames) == 0:
                        errs.append(f"no frames found in {video_root}")
    else:
        errs.append("could not infer image source (expected pil_image/pil_images/image_path/video_path)")

    # Task-specific fields (minimal / non-exhaustive)
    if eval_type in ("gar_image_mc_qa",):
        errs += _require_keys(item, ["question", "choices", "answer", "mask_rles"])
        if "choices" in item and not isinstance(item["choices"], list):
            errs.append(f"choices must be list, got: {type(item['choices'])}")
        if "mask_rles" in item and not isinstance(item["mask_rles"], list):
            errs.append(f"mask_rles must be list, got: {type(item['mask_rles'])}")
    elif eval_type in ("gar_image_simple_oe_qa", "gar_image_detail_oe_qa"):
        errs += _require_keys(item, ["question", "answer", "mask_rles"])
        if "mask_rles" in item and not isinstance(item["mask_rles"], list):
            errs.append(f"mask_rles must be list, got: {type(item['mask_rles'])}")
    elif eval_type == "vip_image_oe_qa":
        errs += _require_keys(item, ["question", "answer", "capability"])
    elif eval_type == "blink_image_mc_qa":
        errs += _require_keys(item, ["question", "choices", "answer", "type"])
        if "choices" in item and not isinstance(item["choices"], list):
            errs.append(f"choices must be list, got: {type(item['choices'])}")
    elif eval_type == "inst_it_image_mc_qa":
        errs += _require_keys(
            item,
            ["question", "choice_a", "choice_b", "choice_c", "choice_d", "answer"],
        )
    elif eval_type == "inst_it_image_oe_qa":
        errs += _require_keys(item, ["question", "answer"])
    elif eval_type in ("inst_it_video_mc_qa", "inst_it_video_oe_qa"):
        errs += _require_keys(item, ["question", "video_path"])
    else:
        errs.append(f"unknown eval_type: {eval_type}")

    return errs


def _load_getters() -> Dict[str, Callable[[], List[Dict[str, Any]]]]:
    # Import lazily so users can run this script without importing heavy deps elsewhere.
    from behaviors import (
        get_blink_image_mc_data,
        get_gar_caption_detailed_data,
        get_gar_caption_simple_data,
        get_gar_image_mc_data,
        get_inst_it_image_mc_data,
        get_inst_it_image_oe_data,
        get_inst_it_video_mc_data,
        get_inst_it_video_oe_data,
        get_vip_bench_data,
    )

    return {
        "gar_image_mc_qa": get_gar_image_mc_data,
        "gar_image_simple_oe_qa": get_gar_caption_simple_data,
        "gar_image_detail_oe_qa": get_gar_caption_detailed_data,
        "vip_image_oe_qa": get_vip_bench_data,
        "blink_image_mc_qa": get_blink_image_mc_data,
        "inst_it_image_mc_qa": get_inst_it_image_mc_data,
        "inst_it_image_oe_qa": get_inst_it_image_oe_data,
        "inst_it_video_mc_qa": get_inst_it_video_mc_data,
        "inst_it_video_oe_qa": get_inst_it_video_oe_data,
    }


def _summarize_items(eval_type: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    src_counts: Dict[str, int] = {}
    keys_seen: Dict[str, int] = {}
    for it in items[:200]:
        src = _infer_image_source(it)
        src_counts[src] = src_counts.get(src, 0) + 1
        for k in it.keys():
            keys_seen[k] = keys_seen.get(k, 0) + 1
    top_keys = sorted(keys_seen.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    return {
        "type": eval_type,
        "num_items": len(items),
        "image_source_counts_first200": src_counts,
        "top_keys_first200": top_keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check get_*_data outputs. Includes a fast overlay detection mode.")
    parser.add_argument(
        "--mode",
        type=str,
        default="schema",
        choices=["schema", "overlay", "save"],
        help="schema: validate fields/paths; overlay: heuristic overlay detection; save: save one preview image per type",
    )
    parser.add_argument("--types", nargs="*", default=None, help="subset of types to check (default: all supported)")
    parser.add_argument("--max_items", type=int, default=50, help="max items to validate per type (schema mode)")
    parser.add_argument("--fail_fast", action="store_true", help="stop on first error")
    parser.add_argument("--report_out", type=str, default=None, help="optional JSON report output path")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="required for --mode save: directory to write PNG previews (manual inspection)",
    )
    parser.add_argument(
        "--video_cell_w",
        type=int,
        default=320,
        help="thumbnail width per cell when stitching video frames / pil_images grid",
    )
    parser.add_argument(
        "--video_cell_h",
        type=int,
        default=320,
        help="thumbnail height per cell when stitching video frames / pil_images grid",
    )
    parser.add_argument(
        "--video_max_frames",
        type=int,
        default=0,
        help="if >0, only use first N frames for video mosaic (0 = all frames)",
    )
    args = parser.parse_args()

    if args.mode == "save" and not args.save_dir:
        parser.error("--mode save requires --save_dir")

    getters = _load_getters()
    types = args.types if args.types else list(getters.keys())

    report: Dict[str, Any] = {"summary": [], "errors": {}, "overlay": {}, "save": {}}
    any_fail = False

    for tname in types:
        if tname not in getters:
            print(f"[SKIP] unknown type: {tname}")
            continue

        print(f"\n=== {tname} ===")
        getter = getters[tname]
        try:
            items = getter()
        except Exception as ex:
            any_fail = True
            report["errors"][tname] = {"_loader_exception": repr(ex)}
            print(f"[FAIL] loader threw exception: {ex!r}")
            if args.fail_fast:
                break
            continue

        report["summary"].append(_summarize_items(tname, items))
        print(f"[OK] loaded {len(items)} items")

        if args.mode == "save":
            if not items:
                report["save"][tname] = {"error": "no items"}
                print("[FAIL] no items")
                continue
            item0 = items[0]
            save_res = _save_preview_for_type(
                tname,
                item0,
                args.save_dir,
                video_cell_w=args.video_cell_w,
                video_cell_h=args.video_cell_h,
                video_max_frames=args.video_max_frames,
            )
            report["save"][tname] = save_res
            if tname == "blink_image_mc_qa":
                saved_map = (save_res or {}).get("saved", {}) if isinstance(save_res, dict) else {}
                for abbr, p in saved_map.items():
                    if p:
                        print(f"[SAVED] {p}")
                    else:
                        print(f"[FAIL] blink subset {abbr} not saved")
                if (save_res or {}).get("error"):
                    print(f"[FAIL] {save_res.get('error')}")
            else:
                if save_res.get("saved"):
                    print(f"[SAVED] {save_res['saved']}")
                    if save_res.get("num_frames_used") is not None:
                        print(f"        frames in mosaic: {save_res['num_frames_used']}")
                    if save_res.get("num_subimages") is not None:
                        print(f"        sub-images in grid: {save_res['num_subimages']}")
                else:
                    print(f"[FAIL] {save_res.get('error', 'unknown')}")
            continue

        if args.mode == "overlay":
            if not items:
                any_fail = True
                report["overlay"][tname] = {"error": "no items"}
                print("[FAIL] no items")
                if args.fail_fast:
                    break
                continue

            if tname == "vip_image_oe_qa":
                dual = _overlay_check_vip_dual()
                report["overlay"][tname] = dual
                ls = dual["source"]["likely"]
                lb = dual["bbox"]["likely"]
                print(
                    f"[ViP] source={dual.get('source_path')} OVERLAY_LIKELY={ls} | "
                    f"bbox={dual.get('bbox_path')} OVERLAY_LIKELY={lb}"
                )
                continue

            if tname == "gar_image_detail_oe_qa":
                from behaviors import get_gar_caption_detailed_data

                dual = _overlay_check_dual_image_path(
                    getter_noref_true=get_gar_caption_detailed_data,
                    getter_noref_false=get_gar_caption_detailed_data,
                )
                report["overlay"][tname] = dual
                ln = dual["left"]["likely"]
                lr = dual["right"]["likely"]
                print(
                    f"[GAR-Detailed] noref={dual.get('left_path')} OVERLAY_LIKELY={ln} | "
                    f"refer={dual.get('right_path')} OVERLAY_LIKELY={lr}"
                )
                continue

            if tname in ("inst_it_image_mc_qa", "inst_it_image_oe_qa"):
                from behaviors import get_inst_it_image_mc_data, get_inst_it_image_oe_data

                getter = get_inst_it_image_mc_data if tname == "inst_it_image_mc_qa" else get_inst_it_image_oe_data
                dual = _overlay_check_dual_image_path(
                    getter_noref_true=getter,
                    getter_noref_false=getter,
                )
                report["overlay"][tname] = dual
                ln = dual["left"]["likely"]
                lr = dual["right"]["likely"]
                print(
                    f"[Inst-It {tname}] raw={dual.get('left_path')} OVERLAY_LIKELY={ln} | "
                    f"vpt={dual.get('right_path')} OVERLAY_LIKELY={lr}"
                )
                continue

            if tname in ("inst_it_video_mc_qa", "inst_it_video_oe_qa"):
                from behaviors import get_inst_it_video_mc_data, get_inst_it_video_oe_data

                getter = get_inst_it_video_mc_data if tname == "inst_it_video_mc_qa" else get_inst_it_video_oe_data
                dual = _overlay_check_dual_video(
                    getter_noref_true=getter,
                    getter_noref_false=getter,
                )
                report["overlay"][tname] = dual
                ln = dual["left"]["likely"]
                lr = dual["right"]["likely"]
                print(
                    f"[Inst-It {tname}] raw_first={dual.get('left_first_frame')} OVERLAY_LIKELY={ln} | "
                    f"vpt_first={dual.get('right_first_frame')} OVERLAY_LIKELY={lr}"
                )
                continue

            if tname == "blink_image_mc_qa":
                # Only check selected BLINK subsets: RRF, RDP, FCO.
                results = {}
                for abbr, subset_name in BLINK_CHECK_SUBSETS.items():
                    match = next((it for it in items if it.get("type") == subset_name), None)
                    if match is None:
                        results[abbr] = {"error": "no match"}
                        continue
                    ims = match.get("pil_images")
                    first = None
                    if isinstance(ims, list) and ims:
                        first = ims[0]
                    likely, stats = _overlay_likely_boxnum(first)
                    results[abbr] = {"likely": likely, "stats": stats}
                    print(f"[BLINK {abbr}] subset={subset_name} OVERLAY_LIKELY={likely}")
                report["overlay"][tname] = results
                continue

            item0 = items[0]
            im, meta = _load_first_image_for_overlay_check(tname, item0)
            likely, stats = _overlay_likely_boxnum(im)
            out = {"likely": likely, "meta": meta, "stats": stats}
            report["overlay"][tname] = out

            tag = "OVERLAY_LIKELY" if likely else "OVERLAY_UNLIKELY"
            origin = meta.get("origin", "unknown")
            extra = ""
            if "image_path" in meta:
                extra = f" image_path={meta['image_path']}"
            if "first_frame_path" in meta:
                extra = f" first_frame={meta['first_frame_path']}"
            print(f"[{tag}] source={meta.get('image_source')} origin={origin}{extra}")

            # In overlay-only mode we don't treat OVERLAY_LIKELY as a failure; it's just a signal.
            continue

        errs_for_type: Dict[str, Any] = {}
        n_check = min(len(items), max(0, args.max_items))
        for i in range(n_check):
            it = items[i]
            if not isinstance(it, dict):
                any_fail = True
                errs_for_type[str(i)] = [f"item not dict: {type(it)}"]
                print(f"[FAIL] item[{i}] not dict: {type(it)}")
                if args.fail_fast:
                    break
                continue

            errs = _check_item_schema(tname, it)
            if errs:
                any_fail = True
                errs_for_type[str(i)] = errs
                brief = "; ".join(errs[:3])
                print(f"[FAIL] item[{i}] {brief}")
                if args.fail_fast:
                    break

        if errs_for_type:
            report["errors"][tname] = errs_for_type
        else:
            print(f"[OK] first {n_check} items passed basic checks")

        if args.fail_fast and any_fail:
            break

    if args.report_out:
        import json

        os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[REPORT] wrote {args.report_out}")

    if args.mode == "schema" and any_fail:
        raise SystemExit(2)
    print("\nDone.")


if __name__ == "__main__":
    main()

