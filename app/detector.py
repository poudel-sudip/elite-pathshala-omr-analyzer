import logging

import cv2
import numpy as np

from app.config import Config
from app.templates import templates

logger = logging.getLogger(__name__)

def normalize_image(img):
    """
    Normalize image using template corner markers.
    """

    img = cv2.resize(
        img,
        (
            Config.TARGET_IMAGE_WIDTH,
            Config.TARGET_IMAGE_HEIGHT
        )
    )

    page_corners = locate_page_corner_markers(img)

    template_corners = templates.corner_markers

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

    transform_matrix = cv2.getPerspectiveTransform(
        src,
        dst
    )

    h, w = img.shape[:2]

    aligned_img = cv2.warpPerspective(
        img,
        transform_matrix,
        (w, h)
    )

    return aligned_img


def locate_page_corner_markers(img):
    """
    Detect four page corner markers.
    """

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        Config.CORNER_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidate_centers = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if (
            area < Config.CORNER_MIN_AREA
            or area > Config.CORNER_MAX_AREA
        ):
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = float(w) / h

        if (
            Config.CORNER_MIN_ASPECT_RATIO
            <= aspect_ratio
            <= Config.CORNER_MAX_ASPECT_RATIO
        ):

            moments = cv2.moments(contour)

            if moments["m00"] != 0:

                center_x = int(
                    moments["m10"] / moments["m00"]
                )

                center_y = int(
                    moments["m01"] / moments["m00"]
                )

                candidate_centers.append(
                    (center_x, center_y)
                )

    if len(candidate_centers) < 4:
        raise ValueError(
            f"Found only {len(candidate_centers)} corner markers"
        )

    pts = np.array(candidate_centers)

    sum_pts = pts.sum(axis=1)

    diff_pts = np.diff(
        pts,
        axis=1
    ).flatten()

    tl = pts[np.argmin(sum_pts)]
    br = pts[np.argmax(sum_pts)]
    tr = pts[np.argmin(diff_pts)]
    bl = pts[np.argmax(diff_pts)]

    return {
        "tl": {"x": int(tl[0]), "y": int(tl[1])},
        "tr": {"x": int(tr[0]), "y": int(tr[1])},
        "br": {"x": int(br[0]), "y": int(br[1])},
        "bl": {"x": int(bl[0]), "y": int(bl[1])},
    }


def _count_dark_pixels(thresh,x,y):
    """
    Count dark pixels inside bubble.
    """

    radius = Config.BUBBLE_RADIUS

    x1 = max(
        0,
        x - radius
    )

    y1 = max(
        0,
        y - radius
    )

    x2 = min(
        thresh.shape[1],
        x + radius
    )

    y2 = min(
        thresh.shape[0],
        y + radius
    )

    roi = thresh[
        y1:y2,
        x1:x2
    ]

    return cv2.countNonZero(roi)


def detect_student_id(img):
    """
    Detect student ID.
    """

    coords = templates.student_id_bubbles

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    detected_digits = {}

    total_box_area = (Config.BUBBLE_RADIUS * 2) ** 2

    for col in range(Config.STUDENT_ID_COLS):

        pixel_counts = []

        for row in range(Config.STUDENT_ID_ROWS):

            point = coords[f"{col}_{row}"]

            count = _count_dark_pixels(
                thresh,
                point["x"],
                point["y"]
            )

            pixel_counts.append(count)

        max_pixels = max(pixel_counts)
        min_pixels = min(pixel_counts)

        selected_row = pixel_counts.index(max_pixels)

        other_pixels = [
            p
            for i, p in enumerate(pixel_counts)
            if i != selected_row
        ]

        avg_baseline = (
            sum(other_pixels) / len(other_pixels)
            if other_pixels
            else 0
        )

        has_high_contrast = (max_pixels - min_pixels) > (total_box_area * Config.CONTRAST_THRESHOLD_PERCENT)

        is_distinct = (max_pixels > avg_baseline * Config.DISTINCTNESS_RATIO and max_pixels > Config.MIN_FILLED_PIXEL_COUNT)

        is_solid_fill = (max_pixels > total_box_area * Config.SOLID_FILL_PERCENT)

        if ((has_high_contrast and is_distinct) or is_solid_fill):
            detected_digits[col] = str(selected_row)
        else:
            detected_digits[col] = "?"

    return "".join(
        detected_digits[col]
        for col in range(
            Config.STUDENT_ID_COLS
        )
    )

def detect_set_key(img):
    """
    Detect question set key.
    """

    coords = templates.set_bubbles

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    set_key = ""
    pixel_counts = {}

    for option in Config.OPTIONS:
        point = coords[option]
        pixel_counts[option] = (
            _count_dark_pixels(
                thresh,
                point["x"],
                point["y"]
            )
        )

    sorted_options = sorted(
        pixel_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_option, best_pixels = sorted_options[0]
    second_pixels = sorted_options[1][1]

    ratio = second_pixels / best_pixels

    set_key = best_option  # ATTEMPTED

    if (second_pixels >= Config.MULTI_MARK_MIN_PIXELS and ratio >= Config.MULTI_MARK_RATIO):
        set_key = best_option+"M"  # MULTI_MARKED

    if best_pixels < Config.ATTEMPTED_MIN_PIXELS:
        set_key = ""  # NOT_ATTEMPTED

    return set_key


def detect_question_answers(img):
    """
    Detect answers for all questions.
    """

    coords = templates.question_bubbles

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    answers = {}

    for question in range(Config.TOTAL_QUESTIONS):

        q_key = str(question + 1)

        pixel_counts = {}

        for option in Config.OPTIONS:

            point = coords[q_key][option]

            pixel_counts[option] = (
                _count_dark_pixels(
                    thresh,
                    point["x"],
                    point["y"]
                )
            )

        sorted_options = sorted(
            pixel_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )

        best_option, best_pixels = sorted_options[0]
        second_pixels = sorted_options[1][1]

        ratio = second_pixels / best_pixels

        if best_pixels < Config.ATTEMPTED_MIN_PIXELS:
            answers[q_key] = ""  # NOT_ATTEMPTED
            continue

        if (second_pixels >= Config.MULTI_MARK_MIN_PIXELS and ratio >= Config.MULTI_MARK_RATIO):
            answers[q_key] = best_option+"M"  # MULTI_MARKED
            continue

        answers[q_key] = best_option  # ATTEMPTED
        
    return answers


def update_omr_sheet(img,user_answers,correct_answers = {}, user_set = None, correct_set = None):
    """
    Draw evaluation result on OMR image.

    ```
    Returns:
        OpenCV image
    """


    output = img.copy()
    if len(correct_answers):
        coords = templates.question_bubbles
        for q_no, options in coords.items():
            user_answer = user_answers.get(q_no)
            correct_answer = correct_answers.get(q_no)
            if (user_answer == "" or user_answer is None or correct_answer is None):
                continue

            is_correct = (user_answer == correct_answer)
            for option, point in options.items():

                # if option != user_answer:
                if option != (user_answer[0] if user_answer else ""):
                    continue

                color = (Config.GREEN_COLOR if is_correct else Config.RED_COLOR)

                cv2.circle(
                    output,
                    (
                        point["x"],
                        point["y"]
                    ),
                    Config.BUBBLE_RADIUS,
                    color,
                    -1
                )


    if len(user_set) and len(correct_set):
        coords = templates.set_bubbles
        is_correct = (user_set == correct_set)
        color = (Config.GREEN_COLOR if is_correct else Config.RED_COLOR)
        point = coords.get(user_set)
        if point:
            cv2.circle(
                output,
                (
                    point["x"],
                    point["y"]
                ),
                Config.BUBBLE_RADIUS,
                color,
                -1
            )


    return output

