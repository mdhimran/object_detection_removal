from imports import *

# Function to get a pre-trained model
def get_model():
    model = models.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to transform the image
def transform_image(image):
    transform = T.Compose([
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to perform object detection
def detect_objects(model, image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions

# Function to visualize the results
def visualize_detection(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    scores = predictions[0]['scores'].numpy()
    boxes = predictions[0]['boxes'].detach().numpy()
    labels = predictions[0]['labels'].numpy()

    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
        'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    for score, box, label in zip(scores, boxes, labels):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)
            draw.text((x1, y1), COCO_INSTANCE_CATEGORY_NAMES[label], fill="yellow", font=font)

    return image

# Function to inpaint detected objects
def inpaint_objects(image, predictions, threshold=0.5):
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)

    scores = predictions[0]['scores'].numpy()
    boxes = predictions[0]['boxes'].detach().numpy()

    for score, box in zip(scores, boxes):
        if score > threshold:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 255

    inpainted_image = cv2.inpaint(image_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    inpainted_image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    return inpainted_image_pil