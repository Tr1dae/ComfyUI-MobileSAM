# ComfyUI-MobileSAM

A ComfyUI custom node for text-guided image segmentation using GroundingDINO and MobileSAM.

## Features

- **Text-guided segmentation**: Use natural language prompts like "face", "car", or "building" to segment objects in images
- **High-quality masks**: Leverages MobileSAM for precise segmentation masks
- **Multiple outputs**: Returns preview images, per-instance masks, combined masks, and detection metadata
- **Easy integration**: Self-contained node pack with bundled MobileSAM code

## Installation

### 1. Install Python dependencies

```bash
cd custom_nodes/ComfyUI-MobileSAM
pip install -r requirements.txt
```

### 2. Download Required Models (Optional)

The node will automatically download the required models on first use. However, if you prefer to download them manually:

#### GroundingDINO Models
Download the GroundingDINO config and weights:
- Config: [GroundingDINO_SwinT_OGC.cfg.py](https://huggingface.co/pengxian/grounding-dino/resolve/main/GroundingDINO_SwinT_OGC.cfg.py)
- Weights: [groundingdino_swint_ogc.pth](https://huggingface.co/pengxian/grounding-dino/resolve/main/groundingdino_swint_ogc.pth)

Place them in your ComfyUI `models/grounding-dino/` directory:
```
ComfyUI/
├── models/
│   └── grounding-dino/
│       ├── GroundingDINO_SwinT_OGC.cfg.py
│       └── groundingdino_swint_ogc.pth
```

#### MobileSAM Model
Download the MobileSAM checkpoint:
- [mobile_sam.pt](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt?raw=true)

Place it in your ComfyUI `models/detection/` directory:
```
ComfyUI/
├── models/
│   └── detection/
│       └── mobile_sam.pt
```

**Note**: If you don't download the models manually, they will be automatically downloaded to the correct directories when you first use the node.

### Optional model selection

Each Easy Mobile SAM node now exposes three **optional** dropdown inputs:

- `grounding_config_choice` (GroundingDINO config)
- `grounding_checkpoint_choice` (GroundingDINO weights)
- `mobile_sam_checkpoint_choice` (MobileSAM checkpoint)

All three default to `AUTO`, which scans `models/grounding-dino`, `models/groundingdino`, `models/grounding_dino`, `models/detection`, `models/detections`, `models/sam`, and `models/mobile-sam` for matching files before triggering any download. If you pre-place a file inside one of those folders, select it from the dropdown to skip the download entirely—ComfyUI will load the path you chose and raise an error if the file later disappears.

Setting any input to a specific file path also works; the node will validate the file exists before using it and will never call the downloader when a file is explicitly provided.

### 3. Install the Custom Node

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Tr1dae/ComfyUI-MobileSAM.git
```

Restart ComfyUI to load the node.

## Usage

### Node Inputs

- **image**: Input image (IMAGE tensor)
- **sam_prompt**: Text prompt describing what to segment (e.g., "face", "car", "person")
- **threshold**: Detection confidence threshold (0.0-1.0, default: 0.35)
- **min_pixels_width**: Minimum width filter for detected boxes (default: 0)
- **min_pixels_height**: Minimum height filter for detected boxes (default: 0)

### Node Outputs

- **preview**: Annotated image showing detected bounding boxes and labels
- **masks**: Individual segmentation masks for each detected object (MASK tensor)
- **mask_combined**: Combined mask of all detected objects (MASK tensor)
- **detections**: JSON object containing detection metadata including bounding boxes, confidence scores, and phrases
- **mobile_sam_checkpoint**: Path to the loaded MobileSAM checkpoint

### Workflow Example

1. Load an image using a Load Image node
2. Connect the image to the Easy Mobile SAM node
3. Set the `sam_prompt` to describe what you want to segment (e.g., "person")
4. Adjust threshold and minimum size filters as needed
5. Connect the outputs to your desired processing nodes:
   - Use `preview` for visualization
   - Use `masks` for per-object processing
   - Use `mask_combined` for binary masking operations

## Dependencies

- torch
- torchvision
- numpy
- opencv-python
- groundingdino (external pip install)
- supervision (included with groundingdino)

## Architecture

The node works in a pipeline:

1. **Text Detection**: GroundingDINO processes the text prompt and input image to detect relevant objects and their bounding boxes
2. **Box Filtering**: Detected boxes are filtered by confidence threshold and minimum size requirements
3. **Mask Generation**: MobileSAM generates high-quality segmentation masks for each filtered bounding box
4. **Output Assembly**: Individual masks are combined and returned along with preview and metadata

## Troubleshooting

### Import Errors
- Ensure GroundingDINO is installed: `pip install groundingdino`
- The required model files will be automatically downloaded on first use
- Restart ComfyUI after installing dependencies

### Download Issues
- If automatic downloads fail, download the models manually as described in the installation section
- Check your internet connection and ensure you have sufficient disk space

### No Detections
- Try lowering the `threshold` value
- Check your `sam_prompt` - use simple, descriptive terms
- Ensure the objects you're trying to detect are clearly visible in the image

### Performance Issues
- The node requires CUDA-compatible GPU for best performance
- Large images may take longer to process
- Consider using minimum size filters to reduce processing of small/irrelevant detections
- First run may take longer due to model downloads

## License

This project bundles MobileSAM code (licensed under Apache 2.0) and creates a ComfyUI interface for it. See individual component licenses for details.

## Acknowledgments

- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) - Efficient segmentation model
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) - Text-guided object detection
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Node-based interface framework