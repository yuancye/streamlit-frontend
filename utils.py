
import requests
import json

FASTAPI_URL = "http://127.0.0.1:8000/infer"

def run_inference(uploaded_file, nms, min_score, bboxes=None):

    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

    data = {
        "nms": nms,
        "min_score": min_score,
        "bboxes": json.dumps(bboxes) if bboxes is not None else None
    }

    try:
        response = requests.post(FASTAPI_URL, files=files, data=data)
        response.raise_for_status()   
        return response.json() 
    except requests.exceptions.RequestException as e:
        print(f"Error calling FastAPI: {e}")
        return None


def post_process_bbox(bboxes):
    bboxes_xyxy = []
    for bbox in bboxes:
        x1 = bbox["x"]
        y1 = bbox["y"]
        x2 = bbox["x"] + bbox["width"]
        y2 = bbox["y"] + bbox["height"]
        bboxes_xyxy.append([x1, y1, x2, y2])
    
    return bboxes_xyxy

def convert_to_original_coordinates(bboxes, scale_x, scale_y):
    original_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        original_x1 = round(float(x1 * scale_x), 4)
        original_y1 = round(float(y1 * scale_y), 4)
        original_x2 = round(float(x2 * scale_x), 4)
        original_y2 = round(float(y2 * scale_y), 4)
        original_bboxes.append([original_x1, original_y1, original_x2, original_y2])
    
    return original_bboxes
