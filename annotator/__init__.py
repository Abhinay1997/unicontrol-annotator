import torch
import einops
import cv2
import numpy as np
import cvlib as cv

from .util import resize_image, HWC3

from .canny import CannyDetector
from .midas import MidasDetector
from .hed import HEDdetector
from .openpose import OpenposeDetector
from .grayscale import GrayscaleConverter
from .blur import Blurrer
from .outpainting import Outpainter
from .inpainting import Inpainter
from .uniformer import UniformerDetector


apply_uniformer = UniformerDetector()
apply_midas = MidasDetector()
apply_canny = CannyDetector()
apply_hed = HEDdetector()
model_outpainting = Outpainter()
apply_openpose = OpenposeDetector()
model_grayscale = GrayscaleConverter()
model_blur = Blurrer()
model_inpainting = Inpainter()

#All functions take img (h,w,c3) as input and return (c,h,w) with values in 0-1.
color_dict = {
    'background': (0, 0, 100),
    'person': (255, 0, 0),
    'bicycle': (0, 255, 0),
    'car': (0, 0, 255),
    'motorcycle': (255, 255, 0),
    'airplane': (255, 0, 255),
    'bus': (0, 255, 255),
    'train': (128, 128, 0),
    'truck': (128, 0, 128),
    'boat': (0, 128, 128),
    'traffic light': (128, 128, 128),
    'fire hydrant': (64, 0, 0),
    'stop sign': (0, 64, 0),
    'parking meter': (0, 0, 64),
    'bench': (64, 64, 0),
    'bird': (64, 0, 64),
    'cat': (0, 64, 64),
    'dog': (192, 192, 192),
    'horse': (32, 32, 32),
    'sheep': (96, 96, 96),
    'cow': (160, 160, 160),
    'elephant': (224, 224, 224),
    'bear': (32, 0, 0),
    'zebra': (0, 32, 0),
    'giraffe': (0, 0, 32),
    'backpack': (32, 32, 0),
    'umbrella': (32, 0, 32),
    'handbag': (0, 32, 32),
    'tie': (96, 0, 0),
    'suitcase': (0, 96, 0),
    'frisbee': (0, 0, 96),
    'skis': (96, 96, 0),
    'snowboard': (96, 0, 96),
    'sports ball': (0, 96, 96),
    'kite': (160, 0, 0),
    'baseball bat': (0, 160, 0),
    'baseball glove': (0, 0, 160),
    'skateboard': (160, 160, 0),
    'surfboard': (160, 0, 160),
    'tennis racket': (0, 160, 160),
    'bottle': (224, 0, 0),
    'wine glass': (0, 224, 0),
    'cup': (0, 0, 224),
    'fork': (224, 224, 0),
    'knife': (224, 0, 224),
    'spoon': (0, 224, 224),
    'bowl': (64, 64, 64),
    'banana': (128, 64, 64),
    'apple': (64, 128, 64),
    'sandwich': (64, 64, 128),
    'orange': (128, 128, 64),
    'broccoli': (128, 64, 128),
    'carrot': (64, 128, 128),
    'hot dog': (192, 64, 64),
    'pizza': (64, 192, 64),
    'donut': (64, 64, 192),
    'cake': (192, 192, 64),
    'chair': (192, 64, 192),
    'couch': (64, 192, 192),
    'potted plant': (96, 32, 32),
    'bed': (32, 96, 32),
    'dining table': (32, 32, 96),
    'toilet': (96, 96, 32),
    'tv': (96, 32, 96),
    'laptop': (32, 96, 96),
    'mouse': (160, 32, 32),
    'remote': (32, 160, 32),
    'keyboard': (32, 32, 160),
    'cell phone': (160, 160, 32),
    'microwave': (160, 32, 160),
    'oven': (32, 160, 160),
    'toaster': (224, 32, 32),
    'sink': (32, 224, 32),
    'refrigerator': (32, 32, 224),
    'book': (224, 224, 32),
    'clock': (224, 32, 224),
    'vase': (32, 224, 224),
    'scissors': (64, 96, 96),
    'teddy bear': (96, 64, 96),
    'hair drier': (96, 96, 64),
    'toothbrush': (160, 96, 96)
}

# task = 'canny'
def process_canny(img, resolution = 512, low_threshold = 40, high_threshold = 200, num_images_per_prompt = 1):
    img = resize_image(HWC3(img), resolution)
    H, W, C = img.shape

    detected_map = apply_canny(img, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)

    control = torch.from_numpy(detected_map.copy()).float()/ 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return control

# task = 'hed'
def process_hed(input_image, img_resolution = 512, hed_resolution = 512, num_images_per_prompt = 1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    detected_map = apply_hed(resize_image(input_image, hed_resolution))
    detected_map = HWC3(detected_map)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return control

#task = 'hedsketch'
def process_sketch(input_image, img_resolution = 512, detect_resolution = 512, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    detected_map = apply_hed(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)

    # sketch the hed image
    retry = 0
    cnt = 0
    while retry == 0:
        threshold_value = np.random.randint(110, 160)
        kernel_size = 3
        alpha = 1.5
        beta = 50
        binary_image = cv2.threshold(detected_map, threshold_value, 255, cv2.THRESH_BINARY)[1]
        inverted_image = cv2.bitwise_not(binary_image)
        smoothed_image = cv2.GaussianBlur(inverted_image, (kernel_size, kernel_size), 0)
        sketch_image = cv2.convertScaleAbs(smoothed_image, alpha=alpha, beta=beta)
        if np.sum(sketch_image < 5) > 0.005 * sketch_image.shape[0] * sketch_image.shape[1] or cnt == 5:
            retry = 1
        else:
            cnt += 1
    detected_map = sketch_image

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    return control

# task = 'depth'
def process_depth(input_image, img_resolution = 512, detect_resolution = 384, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return control

# task = 'normal'
def process_normal(input_image, img_resolution = 512, detect_resolution = 384, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape
    
    _, detected_map = apply_midas(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return control

# task = 'openpose'
def process_pose(input_image, img_resolution = 512, detect_resolution = 512, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return control

# task = 'seg'
def process_segmentation(input_image, img_resolution = 512, detect_resolution = 512, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    detected_map = apply_uniformer(resize_image(input_image, detect_resolution))

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

# task = 'bbox'
def process_bbox(input_image, img_resolution = 512, confidence = 0.4, nms_thresh = 0.5, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    bbox, label, conf = cv.detect_common_objects(input_image, confidence=confidence, nms_thresh=nms_thresh)
    mask = np.zeros((input_image.shape), np.uint8)
    if len(bbox) > 0:
        order_area = np.zeros(len(bbox))

        area_all = 0
        for idx_mask, box in enumerate(bbox):
            x_1, y_1, x_2, y_2 = box

            x_1 = 0 if x_1 < 0 else x_1
            y_1 = 0 if y_1 < 0 else y_1
            x_2 = input_image.shape[1] if x_2 < 0 else x_2
            y_2 = input_image.shape[0] if y_2 < 0 else y_2

            area = (x_2 - x_1) * (y_2 - y_1)
            order_area[idx_mask] = area
            area_all += area
        ordered_area = np.argsort(-order_area)

        for idx_mask in ordered_area:
            box = bbox[idx_mask]
            x_1, y_1, x_2, y_2 = box
            x_1 = 0 if x_1 < 0 else x_1
            y_1 = 0 if y_1 < 0 else y_1
            x_2 = input_image.shape[1] if x_2 < 0 else x_2
            y_2 = input_image.shape[0] if y_2 < 0 else y_2

            mask[y_1:y_2, x_1:x_2, :] = color_dict[label[idx_mask]]
    
    detected_map = mask
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    return control

# task = 'outpainting'
def process_outpainting(input_image, img_resolution = 512, height_top_extended = 50, height_down_extended = 50, width_left_extended = 50, width_right_extended = 50, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    detected_map = outpainting(input_image, img_resolution, height_top_extended, height_down_extended, width_left_extended, width_right_extended)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    
    return control

#task = 'inpainting'
def process_inpainting(input_image, img_resolution = 512, h_ratio_t = 30, h_ratio_d = 60, w_ratio_l = 30, w_ratio_r = 60, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    detected_map = inpainting(input_image, img_resolution, h_ratio_t, h_ratio_d, w_ratio_l, w_ratio_r)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

#task = 'grayscale'
def process_colorization(input_image, img_resolution = 512, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    detected_map = grayscale(input_image, img_resolution)
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    detected_map = detected_map[:, :, np.newaxis]
    detected_map = detected_map.repeat(3, axis=2)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    return control

#task = 'blur'
def process_deblur(input_image, img_resolution = 512, ksize = 51, num_images_per_prompt=1):
    input_image = HWC3(input_image)
    img = resize_image(input_image, img_resolution)
    H, W, C = img.shape

    detected_map = blur(input_image, img_resolution, ksize)

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = torch.stack([control for _ in range(num_images_per_prompt)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    return control



