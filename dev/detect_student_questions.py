import cv2
import numpy as np
import json
import requests

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

def detect_student_id(img):
    COLS = 8
    ROWS = 10
    BUBBLE_RADIUS = 14

    # Detection thresholds
    CONTRAST_THRESHOLD_PERCENT = 0.20
    DISTINCTNESS_RATIO = 1.8
    MIN_FILLED_PIXEL_COUNT = 120
    SOLID_FILL_PERCENT = 0.60

    with open("template_student_id_bubbles.json", "r") as f:
        coords = json.load(f)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    detected_id_digits = {}

    for col in range(COLS):
        pixel_counts = []
        for row in range(ROWS):
            key = f"{col}_{row}"
            pt = coords[key]
            x, y = pt["x"], pt["y"]

            x1 = max(0, x - BUBBLE_RADIUS)
            y1 = max(0, y - BUBBLE_RADIUS)
            x2 = min(thresh.shape[1], x + BUBBLE_RADIUS)
            y2 = min(thresh.shape[0], y + BUBBLE_RADIUS)

            bubble_roi = thresh[y1:y2, x1:x2]
            dark_pixel_count = cv2.countNonZero(bubble_roi)
            pixel_counts.append(dark_pixel_count)

        max_dark_pixels = max(pixel_counts)
        min_dark_pixels = min(pixel_counts)
        selected_row = pixel_counts.index(max_dark_pixels)
        other_pixels = [p for i, p in enumerate(pixel_counts) if i != selected_row]
        avg_baseline = sum(other_pixels) / len(other_pixels) if other_pixels else 0
        total_box_area = (BUBBLE_RADIUS * 2) ** 2

        has_high_contrast = (max_dark_pixels - min_dark_pixels) > (total_box_area * CONTRAST_THRESHOLD_PERCENT)
        is_distinct = max_dark_pixels > (avg_baseline * DISTINCTNESS_RATIO) and max_dark_pixels > MIN_FILLED_PIXEL_COUNT
        is_solid_fill = max_dark_pixels > (total_box_area * SOLID_FILL_PERCENT)

        if (has_high_contrast and is_distinct) or is_solid_fill:
            detected_id_digits[col] = str(selected_row)
        else:
            detected_id_digits[col] = "?"

    student_id = "".join([detected_id_digits[col] for col in range(COLS)])

    return student_id

def detect_question_answers(img):
    TOTAL_QUESTIONS = 50
    BUBBLE_RADIUS = 14
    OPTIONS = ["A", "B", "C", "D"]

    # Detection tuning
    CONTRAST_THRESHOLD_PERCENT = 0.20
    DISTINCTNESS_RATIO = 1.8
    MIN_FILLED_PIXEL_COUNT = 120
    SOLID_FILL_PERCENT = 0.60


    with open("template_question_bubbles.json", "r") as f:
        coords = json.load(f)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    answers = {}

    total_box_area = (BUBBLE_RADIUS * 2) ** 2

    for q in range(TOTAL_QUESTIONS):
        q_key = str(q+1)
        pixel_counts = {}
        for option in OPTIONS:
            pt = coords[q_key][option]
            x, y = pt["x"], pt["y"]

            x1 = max(0, x - BUBBLE_RADIUS)
            y1 = max(0, y - BUBBLE_RADIUS)
            x2 = min(thresh.shape[1], x + BUBBLE_RADIUS)
            y2 = min(thresh.shape[0], y + BUBBLE_RADIUS)

            bubble_roi = thresh[y1:y2, x1:x2]
            dark_pixel_count = cv2.countNonZero(bubble_roi)
            pixel_counts[option] = dark_pixel_count

        sorted_options = sorted(
            pixel_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_option = sorted_options[0][0]
        best_pixels = sorted_options[0][1]
        second_pixels = sorted_options[1][1]
        min_pixels = min(pixel_counts.values())
        avg_baseline = (sum(pixel_counts.values()) - best_pixels) / 3
        has_high_contrast = (best_pixels - min_pixels) > (total_box_area * CONTRAST_THRESHOLD_PERCENT)
        is_distinct = (best_pixels > avg_baseline * DISTINCTNESS_RATIO and best_pixels > MIN_FILLED_PIXEL_COUNT)
        is_solid_fill = (best_pixels > total_box_area * SOLID_FILL_PERCENT)

        multi_marked = False
        for option, count in pixel_counts.items():
            if option == best_option:
                continue

            if count > best_pixels * 0.75:
                multi_marked = True
                break          

        if multi_marked:
            answers[q_key] = ""
        elif ((has_high_contrast and is_distinct) or is_solid_fill):
            answers[q_key] = best_option
        else:
            answers[q_key] = ""


    return answers

def update_omr_sheet(img, user_answers, correct_answers):
    with open("template_question_bubbles.json", "r") as f:
        coords = json.load(f)
    
    output = img.copy()
    BUBBLE_RADIUS = 14

    for q_no, options in coords.items():
        user_ans = user_answers.get(q_no)
        correct_ans = correct_answers.get(q_no)
        if user_ans == "" or user_ans is None or correct_ans is None:
            continue

        is_correct = (user_ans == correct_ans)

        for option, pt in options.items():
            x, y = pt["x"], pt["y"]

            if option == user_ans:
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.circle(
                    output,
                    (x, y),
                    BUBBLE_RADIUS,
                    color,
                    -1
                )

    output_image = upload_image_to_api(output)
    
    return output_image

def upload_image_to_api(img):
    _, buffer = cv2.imencode(".jpg", img)
    file_bytes = buffer.tobytes()

    files = {
        "upload_file": ("analysed_sheet.jpg", file_bytes, "image/jpeg")
    }

    url = "http://127.0.0.1:8000/api/upload-file"

    response = requests.post(url, files=files);
    data = response.json()
    img_url = ''

    try:
        img_url = data.get("url")
    except Exception as e:
        img_url = ''

    return img_url

def analyse_student_data(img, correct_answers = {}):
    correct_answers = {
        '1': 'A',
        '2': 'B',
        '3' : 'C',
        '4' : 'D',
        '5' : 'B',
        '6' : 'B',
    }

    student_id = detect_student_id(img)
    student_answers = detect_question_answers(img)
    updated_sheet = update_omr_sheet(img, student_answers, correct_answers)

    return {
        "success" : True,
        "message" : "Data Analysed Successflly",
        "data" : {
            "student_id" : student_id,
            "answers" : student_answers,
            "final_sheet" : updated_sheet,
        }
    }





# Run detection on your sample file
try:
    image_path = "samples/filled.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image for debugging: {image_path}")

    img = normalize_image(img)
    final_data = analyse_student_data(img)
    print(final_data)
   
except Exception as e:
    print(f"Error processing image: {e}")

