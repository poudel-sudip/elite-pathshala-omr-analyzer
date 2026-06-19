import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()


class Config:
    """
    Central configuration for the OMR Analyzer.
    All constants should live here.
    """

    # =====================================
    # Application
    # =====================================

    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))

    # =====================================
    # External Services
    # =====================================

    FILE_UPLOAD_URL = os.getenv(
        "FILE_UPLOAD_URL",
        "http://127.0.0.1:8000/api/upload-file"
    )

    IMAGE_DOWNLOAD_TIMEOUT = int(
        os.getenv("IMAGE_DOWNLOAD_TIMEOUT", 10)
    )

    IMAGE_UPLOAD_TIMEOUT = int(
        os.getenv("IMAGE_UPLOAD_TIMEOUT", 30)
    )

    # =====================================
    # Image Processing
    # =====================================

    TARGET_IMAGE_WIDTH = 2409
    TARGET_IMAGE_HEIGHT = 3437

    # =====================================
    # Template Files
    # =====================================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    STATIC_DIR = os.path.join(BASE_DIR, "static")

    TEMPLATE_CORNER_MARKERS = os.path.join(
        STATIC_DIR,
        "template_corner_markers.json"
    )

    TEMPLATE_STUDENT_ID_BUBBLES = os.path.join(
        STATIC_DIR,
        "template_student_id_bubbles.json"
    )

    TEMPLATE_QUESTION_BUBBLES = os.path.join(
        STATIC_DIR,
        "template_question_bubbles.json"
    )

    # =====================================
    # OMR General Settings
    # =====================================

    TOTAL_QUESTIONS = 50

    OPTIONS = (
        "A",
        "B",
        "C",
        "D"
    )

    # =====================================
    # Bubble Detection
    # =====================================

    BUBBLE_RADIUS = 14

    CONTRAST_THRESHOLD_PERCENT = 0.20

    DISTINCTNESS_RATIO = 1.8

    MIN_FILLED_PIXEL_COUNT = 120

    SOLID_FILL_PERCENT = 0.60

    MULTI_MARK_RATIO = 0.75

    # =====================================
    # Student ID Detection
    # =====================================

    STUDENT_ID_COLS = 8

    STUDENT_ID_ROWS = 10

    # =====================================
    # Corner Marker Detection
    # =====================================

    CORNER_THRESHOLD = 180

    CORNER_MIN_AREA = 100

    CORNER_MAX_AREA = 6000

    CORNER_MIN_ASPECT_RATIO = 0.2

    CORNER_MAX_ASPECT_RATIO = 4.5

    # =====================================
    # API Responses
    # =====================================

    SUCCESS_MESSAGE = "Data Analysed Successfully"

    HEALTH_STATUS = "healthy"

    # =====================================
    # Logging
    # =====================================

    LOG_LEVEL = os.getenv(
        "LOG_LEVEL",
        "INFO"
    )

    # =====================================
    # Colors
    # =====================================

    GREEN_COLOR = (0, 255, 0)    
    RED_COLOR = (0, 0, 255)