# pip install -U torch torchvision
# pip install -U transformers pillow opencv-python
# pip install git+https://github.com/facebookresearch/segment-anything.git

import numpy as np
import torch
from PIL import Image
import cv2

# --- Grounding DINO (HF) ---
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# --- SAM (official) ---
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image_bgr(path: str):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr

@torch.inference_mode()
def grounding_dino_boxes(image_pil: Image.Image, text_prompt: str,
                         box_threshold=0.30, text_threshold=0.25):
    model_id = "IDEA-Research/grounding-dino-base"  # 他のvariantでもOK
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE).eval()

    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    # post-process: 画像サイズに合わせてxyxy boxを返す
    target_sizes = torch.tensor([image_pil.size[::-1]]).to(DEVICE)  # (h,w)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]

    # results["boxes"]: (N,4) xyxy in pixels
    boxes = results["boxes"].detach().cpu().numpy()
    labels = results["labels"]
    scores = results["scores"].detach().cpu().numpy()
    return boxes, labels, scores

@torch.inference_mode()
def sam_masks_from_boxes(image_bgr: np.ndarray, boxes_xyxy: np.ndarray,
                         sam_ckpt_path: str, sam_type="vit_h"):
    # sam_type: "vit_h" / "vit_l" / "vit_b" など（チェックポイントに合わせる）
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt_path).to(DEVICE).eval()
    predictor = SamPredictor(sam)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    masks = []
    for box in boxes_xyxy:
        # SAMは numpy box (x0,y0,x1,y1) を受け取れる
        mask, _, _ = predictor.predict(box=box, multimask_output=False)
        masks.append(mask[0].astype(bool))
    return masks

def main():
    img_path = r"D:\V2X\pair_V2X\infra\pair_000000.jpg"
    sam_ckpt = "sam_vit_h_4b8939.pth"  # 事前にDLしてパスを指定

    image_bgr = load_image_bgr(img_path)
    image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # Grounding DINOはプロンプト末尾に '.' を付ける流儀が多い
    text_prompt = "car . pedestrian . traffic light ."

    boxes, labels, scores = grounding_dino_boxes(
        image_pil, text_prompt, box_threshold=0.30, text_threshold=0.25
    )

    print(f"detected: {len(boxes)}")
    for l, s, b in zip(labels, scores, boxes):
        print(l, float(s), b.tolist())

    masks = sam_masks_from_boxes(image_bgr, boxes, sam_ckpt, sam_type="vit_h")

    # 例：マスクを重ねて保存（簡易）
    overlay = image_bgr.copy()
    for m in masks:
        overlay[m] = (0.5 * overlay[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite("overlay.png", overlay)

if __name__ == "__main__":
    main()