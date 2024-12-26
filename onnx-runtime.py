# %% 
import matplotlib.pyplot as plt 
import cv2 
import onnxruntime as ort
import numpy as np 
from softnms_rotate import softnms_rotate_cpu
from shapely import Polygon

# %%
# Loading Image
image_path = "datasets/dota8/images/val/P1470__1024__3296___1648.jpg"

def img_loader(img_path, imgsz=640, scale_up=True, center=True):     
    image = cv2.imread(img_path) # BGR # (1024, 1024, 3) # HWC
    shape = image.shape[:2] # (height, width)

    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    


    r = min(imgsz[0] / shape[0], imgsz[1] / shape[1])
    
    if not scale_up: 
        r = min(r, 1.0)
    
    
    ratio = r, r 
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = imgsz[1] - new_unpad[1], imgsz[0] - new_unpad[0]

    if center: 
        dw /= 2
        dh /= 2

    if shape != new_unpad: 
        image = cv2.resize(image, imgsz, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))

    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    ) # add border 

    im = np.stack(image)

    im = im[np.newaxis, ...]
    im = im[..., ::-1].transpose((0, 3, 1, 2)) 

    im = np.ascontiguousarray(im, dtype=np.float32)
    im /= 255

    return im 

img = img_loader(image_path)

nc = 15
# %% 
# Loading our model/Starting our Session
model_path = "./runs/obb/train2/weights/best.onnx"
session = ort.InferenceSession(model_path)
# %% 
# Results from Image
raw_output = session.run(None, {"images": img})[0]
raw_output = raw_output.squeeze().T

# %% 
# 0, 1, 2, 3 -> xywh
# [4, 18] -> classification scores 
# 19 -> r

class_scores = raw_output[:, 4:4 + nc]# (4, num_boxes)

# confidence scores
confidence_scores = np.max(class_scores, axis=1)

# Sort the indices based on our confidence score
sorted_indices = np.argsort(confidence_scores)
sorted_confidence_scores = confidence_scores[sorted_indices]

# classes
classes = np.argmax(class_scores, axis=1)
sorted_classes = classes[sorted_indices]

xywh = raw_output[:, :4]
r = (raw_output[:, -1][..., np.newaxis] * 180) / np.pi
boxes = np.concatenate([xywh, r], axis=1)
boxes = boxes[sorted_indices]

offset_classes = sorted_classes[:, np.newaxis]

offset_classes = np.concatenate((offset_classes, np.zeros((8400, 4))), axis=1) * img.shape[2]
boxes = boxes + offset_classes

to_keep = softnms_rotate_cpu(boxes, sorted_confidence_scores, 0.1, 0.45)

print(to_keep)

# %% 
final_res = boxes[to_keep]
final_classes = []

for i in range(final_res.shape[0]): 
    final_classes.append(final_res[i, 0]//img.shape[2])
    final_res[i, 0] %= img.shape[2]

final_classes = np.array(final_classes)
def xywhr_xyxyxyxy(res): 
    boxes = []
    for box in res: 
        points = []
        x, y, w, h, r = box
        theta = np.pi * r / 180
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        points.append([int(x - w/2), int(y - h/2)])
        points.append([int(x + w/2), int(y - h/2)])
        points.append([int(x + w/2), int(y + h/2)])
        points.append([int(x - w/2), int(y + h/2)])
        boxes.append((np.array(points - np.array([[x, y]])) @ rotation_matrix.T) + np.array([[x, y]]))
    return boxes

def xywhr_xyxyxyxyn(res): 
    boxes = []
    for box in res: 
        points = []
        x, y, w, h, r = box
        theta = np.pi * r / 180
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        points.append([int(x - w/2), int(y - h/2)])
        points.append([int(x + w/2), int(y - h/2)])
        points.append([int(x + w/2), int(y + h/2)])
        points.append([int(x - w/2), int(y + h/2)])
        boxes.append(((np.array(points - np.array([[x, y]])) @ rotation_matrix.T) + np.array([[x, y]]))/np.array([[img.shape[2], img.shape[3]]]))
    return boxes

results_obb = xywhr_xyxyxyxyn(final_res)

# %%
# Plotting Image 
image = cv2.imread("datasets/dota8/images/val/P1470__1024__3296___1648.jpg")
plt.imshow(image)
for i in results_obb: 
    polygon = Polygon(i * np.array([[image.shape[0], image.shape[1]]]))
    print(polygon)
    x, y = polygon.exterior.xy 
    plt.plot(x, y)
plt.show()
# %%
