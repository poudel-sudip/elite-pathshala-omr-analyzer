import json
import logging

from flask import (Blueprint,jsonify,render_template,request)

from app.analyzer import OMRAnalyzer
from app.config import Config

logger = logging.getLogger(__name__)

routes = Blueprint("routes",__name__)

analyzer = OMRAnalyzer()

@routes.route("/", methods=["GET"])
def home():
    """
    Home page.
    """
    return render_template("home.html")

@routes.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    """

    return jsonify(
        {
            "status": Config.HEALTH_STATUS
        }
    ), 200


@routes.route("/api/analyze-omr",methods=["POST"])
def analyze_omr():
    """
    Analyze OMR sheet.

    ```
    Accepts:
    {
        "omr_sheet": "https://...",
        "upload_url": "https://...",
        "question_set": "A|B|C|D",
        "correct_answers": {
            "1": "A",
            "2": "C"
        }
    }
    """

    try:

        payload = request.get_json(silent=True) or {}

        omr_sheet = payload.get("omr_sheet")

        upload_url = payload.get("upload_url")

        correct_answers = payload.get("correct_answers")

        correct_set = payload.get("question_set")

        # ----------------------------------
        # Validate OMR Sheet
        # ----------------------------------

        if not omr_sheet:

            return jsonify(
                {
                    "success": False,
                    "error": "omr_sheet is required"
                }
            ), 400

        # ----------------------------------
        # Validate Correct Answers
        # ----------------------------------

        if not correct_answers:
            correct_answers = {}


        if correct_answers is None:

            return jsonify(
                {
                    "success": False,
                    "error": "correct_answers is required"
                }
            ), 400

        # ----------------------------------
        # Support stringified JSON
        # ----------------------------------

        if isinstance(correct_answers, str):
            try:
                correct_answers = json.loads(correct_answers)

            except json.JSONDecodeError:

                return jsonify(
                    {
                        "success": False,
                        "error": "correct_answers must be valid JSON "
                    }
                ), 400

        # ----------------------------------
        # Validate dict
        # ----------------------------------

        if not isinstance(correct_answers, dict):
            return jsonify(
                {
                    "success": False,
                    "error": "correct_answers must be an object"
                }
            ), 400

        # logger.info("Received OMR analysis request")

        result = analyzer.analyze(
            omr_sheet_url=omr_sheet,
            correct_answers=correct_answers,
            upload_url=upload_url,
            correct_set=correct_set
        )

        status_code = (200 if result.get("success") else 400)

        return jsonify(result), status_code

    except Exception as e:

        logger.exception(
            "API request failed"
        )

        return jsonify(
            {
                "success": False,
                "error": str(e)
            }
        ), 500
    
