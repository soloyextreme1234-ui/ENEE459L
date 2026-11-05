import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3 

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, 4]
    boxes = box_scores[:, :4]
    indices = np.argsort(scores)[-candidate_size:]
    picked = []

    while indices.size > 0:
        current = indices[-1]
        picked.append(current)
        if (top_k > 0 and len(picked) >= top_k) or indices.size == 1:
            break
        indices = indices[:-1]
        remaining = boxes[indices]
        lower = np.maximum(boxes[current, :2], remaining[:, :2])
        upper = np.minimum(boxes[current, 2:], remaining[:, 2:])
        intersection = np.prod(np.maximum(upper - lower, 0), axis=1)
        area_current = np.prod(np.maximum(boxes[current, 2:] - boxes[current, :2], 0))
        area_remaining = np.prod(np.maximum(remaining[:, 2:] - remaining[:, :2], 0), axis=1)
        union = area_current + area_remaining - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
        indices = indices[iou <= iou_threshold]

    return box_scores[picked]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized = image.resize((320, 240))
    tensor = np.asarray(resized, dtype=np.float32)
    tensor = (tensor - 127.0) / 128.0
    tensor = np.expand_dims(tensor.transpose(2, 0, 1), axis=0)
    tensor = np.ascontiguousarray(tensor)

    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=available_providers)
    scores, boxes = session.run(None, {session.get_inputs()[0].name: tensor})
    scores = scores[0]
    boxes = boxes[0]
    face_scores = scores[:, 1]
    mask = face_scores >= conf_threshold
    box_scores = np.column_stack((boxes[mask], face_scores[mask]))
    selected = hard_nms(box_scores, iou_threshold=0.3)

    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, score in selected:
        x1, x2 = np.clip([x1 * width, x2 * width], 0, width - 1).astype(int)
        y1, y2 = np.clip([y1 * height, y2 * height], 0, height - 1).astype(int)
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        draw.text((x1, max(0, y1 - 12)), f"Face: {score:.2f}", fill="lime")
    image.save(output_path)

# Edge detection
def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    output = np.zeros_like(magnitude, dtype=np.float32)
    degrees = np.degrees(angle) % 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            direction = degrees[i, j]
            if direction < 22.5 or direction >= 157.5:
                first, second = magnitude[i, j - 1], magnitude[i, j + 1]
            elif direction < 67.5:
                first, second = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif direction < 112.5:
                first, second = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                first, second = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            if magnitude[i, j] >= first and magnitude[i, j] >= second:
                output[i, j] = magnitude[i, j]
    return output

def hysteresis_thresholding(img, low_thresh, high_thresh):
    strong = 255
    weak = 50

    output = np.zeros(img.shape, dtype=np.uint8)

    output[img >= high_thresh] = strong
    output[(img >= low_thresh) & (img < high_thresh)] = weak

    height, width = output.shape
    weak_rows, weak_cols = np.where(output == weak)

    for i, j in zip(weak_rows, weak_cols):
        neighborhood = output[
            max(0, i - 1):min(height, i + 2),
            max(0, j - 1):min(width, j + 2)
        ]

        if np.any(neighborhood == strong):
            output[i, j] = strong
        else:
            output[i, j] = 0

    return output

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    image = Image.open(image_path).convert("L")
    pixels = np.asarray(image, dtype=np.float32)
    smoothed = gaussian_filter(pixels, sigma=1.4)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve(smoothed, kx)
    gy = convolve(smoothed, ky)
    magnitude = np.hypot(gx, gy)
    maximum = magnitude.max()
    if maximum > 0:
        magnitude = magnitude / maximum * 255.0
    angle = np.arctan2(gy, gx)
    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)
    Image.fromarray(edges).save(output_path)

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")