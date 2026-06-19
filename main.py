import cv2
import numpy as np
import requests
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import gc

from config import Config

app = Flask(__name__)
CORS(app)

class OMRAnalyzer:

    def download_image(self, image_url):
        try:
            response = requests.get(image_url, timeout=Config.IMAGE_DOWNLOAD_TIMEOUT)
            response.raise_for_status()

            image = cv2.imdecode(
                np.frombuffer(response.content, np.uint8),
                cv2.IMREAD_COLOR
            )

            return image
        except Exception as e:
            raise Exception(f"Image download failed: {str(e)}")

    def upload_image(self, img):
        _, buffer = cv2.imencode(".jpg", img)
        file_bytes = buffer.tobytes()

        files = {
            "upload_file": ("analysed_sheet.jpg", file_bytes, "image/jpeg")
        }

        url = str(Config.FILE_UPLOAD_URL)

        response = requests.post(url, files=files);
        data = response.json()
        img_url = ''

        try:
            img_url = data.get("url")
        except Exception as e:
            img_url = ''

        return img_url

    def normalize_image(self, img):
        img = cv2.resize(img,(Config.TARGET_IMAGE_WIDTH, Config.TARGET_IMAGE_HEIGHT))
        template_corners = {}
        page_corners = self.locate_page_corner_markers(img)

        with open(Config.TEMPLATE_CORNER_MARKERS) as f:
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

    def locate_page_corner_markers(self, img):
                    
        h, w, _ = img.shape

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, Config.CORNER_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_centers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < Config.CORNER_MIN_AREA or area > Config.CORNER_MAX_AREA:
                continue

            x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = float(w_box) / h_box

            if Config.CORNER_MIN_ASPECT_RATIO <= aspect_ratio <= Config.CORNER_MAX_ASPECT_RATIO:
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

    def detect_student_id(self, img):
       
        with open(Config.TEMPLATE_STUDENT_ID_BUBBLES, "r") as f:
            coords = json.load(f)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        detected_id_digits = {}

        for col in range(Config.STUDENT_ID_COLS):
            pixel_counts = []
            for row in range(Config.STUDENT_ID_ROWS):
                key = f"{col}_{row}"
                pt = coords[key]
                x, y = pt["x"], pt["y"]

                x1 = max(0, x - Config.BUBBLE_RADIUS)
                y1 = max(0, y - Config.BUBBLE_RADIUS)
                x2 = min(thresh.shape[1], x + Config.BUBBLE_RADIUS)
                y2 = min(thresh.shape[0], y + Config.BUBBLE_RADIUS)

                bubble_roi = thresh[y1:y2, x1:x2]
                dark_pixel_count = cv2.countNonZero(bubble_roi)
                pixel_counts.append(dark_pixel_count)

            max_dark_pixels = max(pixel_counts)
            min_dark_pixels = min(pixel_counts)
            selected_row = pixel_counts.index(max_dark_pixels)
            other_pixels = [p for i, p in enumerate(pixel_counts) if i != selected_row]
            avg_baseline = sum(other_pixels) / len(other_pixels) if other_pixels else 0
            total_box_area = (Config.BUBBLE_RADIUS * 2) ** 2

            has_high_contrast = (max_dark_pixels - min_dark_pixels) > (total_box_area * Config.CONTRAST_THRESHOLD_PERCENT)
            is_distinct = max_dark_pixels > (avg_baseline * Config.DISTINCTNESS_RATIO) and max_dark_pixels > Config.MIN_FILLED_PIXEL_COUNT
            is_solid_fill = max_dark_pixels > (total_box_area * Config.SOLID_FILL_PERCENT)

            if (has_high_contrast and is_distinct) or is_solid_fill:
                detected_id_digits[col] = str(selected_row)
            else:
                detected_id_digits[col] = "?"

        student_id = "".join([detected_id_digits[col] for col in range(Config.STUDENT_ID_COLS)])

        return student_id

    def detect_question_answers(self, img):
        
        with open(Config.TEMPLATE_QUESTION_BUBBLES, "r") as f:
            coords = json.load(f)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        answers = {}

        total_box_area = (Config.BUBBLE_RADIUS * 2) ** 2

        for q in range(Config.TOTAL_QUESTIONS):
            q_key = str(q+1)
            pixel_counts = {}
            for option in Config.OPTIONS:
                pt = coords[q_key][option]
                x, y = pt["x"], pt["y"]

                x1 = max(0, x - Config.BUBBLE_RADIUS)
                y1 = max(0, y - Config.BUBBLE_RADIUS)
                x2 = min(thresh.shape[1], x + Config.BUBBLE_RADIUS)
                y2 = min(thresh.shape[0], y + Config.BUBBLE_RADIUS)

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
            has_high_contrast = (best_pixels - min_pixels) > (total_box_area * Config.CONTRAST_THRESHOLD_PERCENT)
            is_distinct = (best_pixels > avg_baseline * Config.DISTINCTNESS_RATIO and best_pixels > Config.MIN_FILLED_PIXEL_COUNT)
            is_solid_fill = (best_pixels > total_box_area * Config.SOLID_FILL_PERCENT)

            multi_marked = False
            for option, count in pixel_counts.items():
                if option == best_option:
                    continue

                if count > best_pixels * Config.MULTI_MARK_RATIO:
                    multi_marked = True
                    break          

            if multi_marked:
                answers[q_key] = ""
            elif ((has_high_contrast and is_distinct) or is_solid_fill):
                answers[q_key] = best_option
            else:
                answers[q_key] = ""

        return answers

    def update_omr_sheet(self, img, user_answers, correct_answers):
        with open(Config.TEMPLATE_QUESTION_BUBBLES, "r") as f:
            coords = json.load(f)
        
        output = img.copy()

        for q_no, options in coords.items():
            user_ans = user_answers.get(q_no)
            correct_ans = correct_answers.get(q_no)
            if user_ans == "" or user_ans is None or correct_ans is None:
                continue

            is_correct = (user_ans == correct_ans)

            for option, pt in options.items():
                x, y = pt["x"], pt["y"]

                if option == user_ans:
                    color = Config.GREEN_COLOR if is_correct else Config.RED_COLOR
                    cv2.circle(
                        output,
                        (x, y),
                        Config.BUBBLE_RADIUS,
                        color,
                        -1
                    )

        output_image = self.upload_image(output)
        
        return output_image

    def analyze(self, omr_sheet, correct_answers = {}):
        try:
            img = self.download_image(omr_sheet)
            if img is None:
                return {"success": False, "error": "Invalid OMR image"}

            img = self.normalize_image(img)

            student_id = self.detect_student_id(img)
            student_answers = self.detect_question_answers(img)
            # updated_sheet = self.update_omr_sheet(img, student_answers, correct_answers)

            return {
                "success" : True,
                "message" : "Data Analysed Successflly",
                "data" : {
                    "student_id" : student_id,
                    "answers" : student_answers,
                    # "final_sheet" : updated_sheet,
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
        
        finally:
            gc.collect()



analyzer = OMRAnalyzer()

@app.route("/api/analyze-omr", methods=["POST"])
def analyze_omr():
    data = request.get_json(silent=True)
    omr_sheet = None
    correct_answers = None

    if data and "omr_sheet" in data:
        omr_sheet = data["omr_sheet"]
    elif request.form.get("omr_sheet"):
        omr_sheet = request.form.get("omr_sheet")
    else:
        omr_sheet = request.args.get("omr_sheet")

    if not omr_sheet:
        return jsonify({"success": False, "error": "omr_sheet required"}), 400


    if data and "correct_answers" in data:
        correct_answers = data["correct_answers"]
    elif request.form.get("correct_answers"):
        correct_answers = request.form.get("correct_answers")
    else:
        correct_answers = request.args.get("correct_answers")
    
    if not correct_answers:
        return jsonify({"success": False, "error": "correct_answers required"}), 400
    

    result = analyzer.analyze(omr_sheet, correct_answers)

    return jsonify(result), (200 if result["success"] else 400)


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == "__main__":
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)