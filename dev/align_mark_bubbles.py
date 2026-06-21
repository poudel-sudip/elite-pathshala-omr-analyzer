import cv2
import numpy as np
import json


def normalize_image(img):

    img = cv2.resize(img,(2409, 3437))

    template_corners = {}
    page_corners = locate_page_corner_markers(img)

    with open("template_corner_markers.json") as f:
        template_corners = json.load(f)

    src = np.array(
        [
            [page_corners["tl"]["x"], page_corners["tl"]["y"]],
            [page_corners["tr"]["x"], page_corners["tr"]["y"]],
            [page_corners["br"]["x"], page_corners["br"]["y"]],
            [page_corners["bl"]["x"], page_corners["bl"]["y"]],
        ],
        dtype=np.float32,
    )

    dst = np.array(
        [
            [template_corners["tl"]["x"], template_corners["tl"]["y"]],
            [template_corners["tr"]["x"], template_corners["tr"]["y"]],
            [template_corners["br"]["x"], template_corners["br"]["y"]],
            [template_corners["bl"]["x"], template_corners["bl"]["y"]],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src, dst)
    h, w = img.shape[:2]
    aligned_img = cv2.warpPerspective(img, H, (w, h))

    return aligned_img

def locate_page_corner_markers(img):
                
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 6000:
            continue

        x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = float(w_box) / h_box

        if 0.2 <= aspect_ratio <= 4.5:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                candidate_centers.append((cX, cY))


    if len(candidate_centers) < 4:
        raise ValueError(f"Found only {len(candidate_centers)} candidate markers. Need at least 4.")

    pts = np.array(candidate_centers)
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1).flatten()
    
    tl = pts[np.argmin(sum_pts)]
    br = pts[np.argmax(sum_pts)]
    tr = pts[np.argmin(diff_pts)]
    bl = pts[np.argmax(diff_pts)]

    corner_coordinates = {
        "tl":     {"x": int(tl[0]), "y": int(tl[1])},
        "tr":    {"x": int(tr[0]), "y": int(tr[1])},
        "br": {"x": int(br[0]), "y": int(br[1])},
        "bl":  {"x": int(bl[0]), "y": int(bl[1])}
    }

    return corner_coordinates

def generate_merged_debug_image(vis_img):
    page_corners = {}
    student_id_bubbles = {}
    questions_bubbles = {}

    with open("template_corner_markers.json") as f:
        page_corners = json.load(f)
    
    with open("template_student_id_bubbles.json") as f:
        student_id_bubbles = json.load(f)
    
    with open("template_question_bubbles.json") as f:
        questions_bubbles = json.load(f)


    # 1. Read base image once
    # vis_img = cv2.imread(image_path)
    # if vis_img is None:
    #     raise FileNotFoundError(f"Could not open or find the image for debugging: {image_path}")

    # 2. Draw Page Corner Markers
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 125, 255)] # BGR
    sorted_keys = ["tl", "tr", "br", "bl"]
    
    for idx, key in enumerate(sorted_keys):
        pt = (page_corners[key]["x"], page_corners[key]["y"])
        cv2.circle(vis_img, pt, 25, colors[idx], -1)
        cv2.putText(vis_img, labels[idx], (pt[0] + 30, pt[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[idx], 4)

    # 3. Draw Student ID Bubbles (Blue dots)
    for key, pt in student_id_bubbles.items():
        cv2.circle(vis_img, (pt["x"], pt["y"]), 8, (255, 0, 0), -1)

    # 4. Draw Question Bubbles (Blue dots with red labels)
    for q, options in questions_bubbles.items():
        for opt, pt in options.items():
            cv2.circle(vis_img, (pt["x"], pt["y"]), 8, (255, 0, 0), -1)
            cv2.putText(vis_img, f"{q}{opt}", (pt["x"] - 10, pt["y"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # 5. Save the combined master view
    # cv2.imwrite("template_marked_image.jpg", vis_img)
    
    return vis_img



# Run detection on your sample file
try:
    image_path = "samples-1/CCI_000059.jpg.jpeg"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image for debugging: {image_path}")

    img = normalize_image(img)
    final_image = generate_merged_debug_image(img)

    cv2.imwrite("final-debug-image.jpg", final_image)
   
except Exception as e:
    print(f"Error processing image: {e}")

