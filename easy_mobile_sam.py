import os
import urllib.request
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

import folder_paths

try:
    from groundingdino.util.inference import Model as GroundingDinoModel
except ImportError as exc:
    GroundingDinoModel = None
    GROUNDING_DINO_IMPORT_ERROR = exc
else:
    GROUNDING_DINO_IMPORT_ERROR = None

try:
    from .mobile_sam.build_sam import sam_model_registry
    from .mobile_sam.predictor import SamPredictor
except ImportError as exc:
    raise ImportError(
        "Failed to import bundled MobileSAM code. Ensure the `mobile_sam` folder lives next to this node."
    ) from exc

GROUNDING_DINO_DIR = os.path.join(folder_paths.models_dir, "grounding-dino")
DETECTION_DIR = os.path.join(folder_paths.models_dir, "detection")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_file(base_dir: str, keywords: Tuple[str, ...], extensions: Tuple[str, ...]) -> Optional[str]:
    if not os.path.isdir(base_dir):
        return None
    normalized_keywords = [keyword.lower() for keyword in keywords]
    for entry in sorted(os.listdir(base_dir)):
        lower_entry = entry.lower()
        if normalized_keywords and not any(keyword in lower_entry for keyword in normalized_keywords):
            continue
        if extensions and not any(lower_entry.endswith(ext) for ext in extensions):
            continue
        return os.path.join(base_dir, entry)
    return None


def _download_file(url: str, dest_path: str) -> None:
    """Download a file from URL to destination path."""
    try:
        print(f"Downloading {url} to {dest_path}")
        # Set up request with user agent to avoid issues
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(dest_path, 'wb') as f:
                f.write(response.read())
        print(f"Successfully downloaded {dest_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def _ensure_grounding_dino_dir() -> None:
    """Ensure the GroundingDINO directory exists."""
    os.makedirs(GROUNDING_DINO_DIR, exist_ok=True)


def _ensure_detection_dir() -> None:
    """Ensure the detection directory exists."""
    os.makedirs(DETECTION_DIR, exist_ok=True)


def _grounding_config_path() -> str:
    """Get or download GroundingDINO config file."""
    _ensure_grounding_dino_dir()
    config_file = _find_file(GROUNDING_DINO_DIR, ("groundingdino",), (".cfg.py", ".cfg", ".py"))

    if config_file is None:
        # Download the config file
        config_url = "https://huggingface.co/pengxian/grounding-dino/resolve/main/GroundingDINO_SwinT_OGC.cfg.py"
        config_path = os.path.join(GROUNDING_DINO_DIR, "GroundingDINO_SwinT_OGC.cfg.py")
        _download_file(config_url, config_path)
        return config_path

    return config_file


def _grounding_checkpoint_path() -> str:
    """Get or download GroundingDINO checkpoint file."""
    _ensure_grounding_dino_dir()
    checkpoint_file = _find_file(GROUNDING_DINO_DIR, ("groundingdino",), (".pth", ".pt"))

    if checkpoint_file is None:
        # Download the checkpoint file
        checkpoint_url = "https://huggingface.co/pengxian/grounding-dino/resolve/main/groundingdino_swint_ogc.pth"
        checkpoint_path = os.path.join(GROUNDING_DINO_DIR, "groundingdino_swint_ogc.pth")
        _download_file(checkpoint_url, checkpoint_path)
        return checkpoint_path

    return checkpoint_file


def _mobile_sam_checkpoint_path() -> str:
    """Get or download MobileSAM checkpoint file."""
    _ensure_detection_dir()
    checkpoint_file = _find_file(DETECTION_DIR, ("mobile_sam", "mobilesam", "sam"), (".pt", ".pth", ".safetensors"))

    if checkpoint_file is None:
        # Download the MobileSAM checkpoint
        checkpoint_url = "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt?raw=true"
        checkpoint_path = os.path.join(DETECTION_DIR, "mobile_sam.pt")
        _download_file(checkpoint_url, checkpoint_path)
        return checkpoint_path

    return checkpoint_file


class EasyMobileSAM:
    _grounding_model: Optional["GroundingDinoModel"] = None
    _sam_predictor: Optional["SamPredictor"] = None
    _mobile_sam_checkpoint: Optional[str] = None
    _device = DEVICE

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam_prompt": ("STRING", {"default": "object"}),
                "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_pixels_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "min_pixels_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "JSON", "STRING")
    RETURN_NAMES = ("preview", "masks", "mask_combined", "detections", "mobile_sam_checkpoint")
    FUNCTION = "segment"
    CATEGORY = "EasyFilePaths"
    DESCRIPTION = (
        "Detect text-guided regions using GroundingDINO, then run MobileSAM to "
        "return previews, masks, combined mask, and the selected checkpoint path."
    )

    def segment(self, image, sam_prompt, threshold, min_pixels_width, min_pixels_height):
        if not sam_prompt or not sam_prompt.strip():
            raise ValueError("sam_prompt cannot be empty.")

        threshold = float(max(0.0, min(1.0, threshold)))
        min_pixels_width = max(0, int(min_pixels_width))
        min_pixels_height = max(0, int(min_pixels_height))

        image_rgb = self._tensor_to_rgb(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        grounding_model = self._ensure_grounding_model()
        detections, phrases = grounding_model.predict_with_caption(
            image=image_bgr,
            caption=sam_prompt,
            box_threshold=threshold,
            text_threshold=threshold,
        )

        records, boxes_array = self._prepare_detections(
            detections, phrases, threshold, min_pixels_width, min_pixels_height
        )

        predictor = self._ensure_sam_predictor()
        predictor.set_image(image_rgb, image_format="RGB")

        mask_tensor, combined_mask = self._predict_masks(
            predictor, boxes_array, image_rgb.shape[:2]
        )

        preview_rgb = self._render_preview(image_rgb, records)
        preview_tensor = torch.tensor(preview_rgb.astype(np.float32) / 255.0)
        preview_tensor = preview_tensor.unsqueeze(0)

        return (
            preview_tensor,
            mask_tensor,
            combined_mask,
            {"detections": records},
            self._mobile_sam_checkpoint or "",
        )

    @classmethod
    def _ensure_grounding_model(cls):
        if cls._grounding_model is not None:
            return cls._grounding_model
        if GroundingDinoModel is None:
            hint = f" ({GROUNDING_DINO_IMPORT_ERROR})" if GROUNDING_DINO_IMPORT_ERROR else ""
            raise RuntimeError(
                "GroundingDINO package is unavailable. "
                "Install it (e.g., `pip install groundingdino`) and restart ComfyUI."
                + hint
            )
        config_path = _grounding_config_path()
        checkpoint_path = _grounding_checkpoint_path()
        cls._grounding_model = GroundingDinoModel(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=cls._device.type,
        )
        return cls._grounding_model

    @classmethod
    def _ensure_sam_predictor(cls):
        if cls._sam_predictor is not None:
            return cls._sam_predictor
        if not sam_model_registry:
            raise RuntimeError("MobileSAM registry is not available.")
        builder = sam_model_registry.get("vit_t") or sam_model_registry.get("default")
        if builder is None:
            raise RuntimeError("SAM builder is missing from the MobileSAM registry.")
        checkpoint_path = _mobile_sam_checkpoint_path()
        cls._mobile_sam_checkpoint = checkpoint_path
        sam_model = builder(checkpoint=checkpoint_path)
        sam_model.to(cls._device)
        cls._sam_predictor = SamPredictor(sam_model)
        return cls._sam_predictor

    @staticmethod
    def _tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
        array = image_tensor.detach().cpu().numpy()
        if array.ndim == 4:
            array = array[0]
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).astype(np.uint8)
        if array.shape[-1] > 3:
            array = array[..., :3]
        return array

    @staticmethod
    def _prepare_detections(
        detections,
        phrases: List[str],
        threshold: float,
        min_w: int,
        min_h: int,
    ) -> Tuple[List[dict], np.ndarray]:
        boxes = detections.xyxy
        confidences = (
            detections.confidence
            if detections.confidence is not None
            else np.ones(len(boxes), dtype=np.float32)
        )
        records: List[dict] = []
        for idx, box in enumerate(boxes):
            width = box[2] - box[0]
            height = box[3] - box[1]
            score = float(confidences[idx]) if len(confidences) > idx else 0.0
            if width < min_w or height < min_h or score < threshold:
                continue
            phrase = phrases[idx] if idx < len(phrases) else ""
            records.append(
                {
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "confidence": score,
                    "phrase": phrase,
                }
            )
        if records:
            boxes_array = np.array([record["bbox"] for record in records], dtype=np.float32)
        else:
            boxes_array = np.zeros((0, 4), dtype=np.float32)
        return records, boxes_array

    @staticmethod
    def _predict_masks(predictor, boxes: np.ndarray, image_size: Tuple[int, int]):
        height, width = image_size
        if boxes.shape[0] == 0:
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return empty_mask, torch.zeros((height, width), dtype=torch.float32)
        mask_tensors = []
        combined = torch.zeros((height, width), dtype=torch.float32)
        for index in range(boxes.shape[0]):
            box_np = boxes[index : index + 1]
            masks_np, _, _ = predictor.predict(
                box=box_np,
                multimask_output=False,
                return_logits=False,
            )
            if masks_np.size == 0:
                continue
            mask_tensor = torch.from_numpy(masks_np[0]).float().cpu()
            mask_tensors.append(mask_tensor)
            combined = torch.maximum(combined, mask_tensor)
        if not mask_tensors:
            empty_mask = torch.zeros((1, height, width), dtype=torch.float32)
            return empty_mask, torch.zeros((height, width), dtype=torch.float32)
        masks_tensor = torch.stack(mask_tensors, dim=0)
        combined_mask = combined.clamp(0.0, 1.0)
        return masks_tensor, combined_mask

    @staticmethod
    def _render_preview(image_rgb: np.ndarray, records: List[dict]) -> np.ndarray:
        preview = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
        for record in records:
            x1, y1, x2, y2 = [int(round(value)) for value in record["bbox"]]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if record["phrase"]:
                cv2.putText(
                    preview,
                    record["phrase"],
                    (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 255, 0),
                    1,
                    lineType=cv2.LINE_AA,
                )
        return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


NODE_CLASS_MAPPINGS = {"EasyMobileSAM": EasyMobileSAM}
NODE_DISPLAY_NAME_MAPPINGS = {"EasyMobileSAM": "Easy Mobile SAM"}
