import logging

from app.config import Config
from app.services import (download_image, upload_image)
from app.detector import (normalize_image, detect_student_id, detect_question_answers, update_omr_sheet)

logger = logging.getLogger(__name__)

class OMRAnalyzer:
    """
    Main OMR Analysis Workflow

    ```
    Flow:
        Download Image
            ↓
        Normalize Image
            ↓
        Detect Student ID
            ↓
        Detect Answers
            ↓
        Generate Evaluated Sheet
            ↓
        Upload Evaluated Sheet
            ↓
        Return Result
    """

    def analyze(self, omr_sheet_url: str, correct_answers: dict, upload_url: str = None) -> dict:

        try:

            # logger.info("Starting analysis for %s", omr_sheet_url)

            # ----------------------------------------
            # Download OMR Sheet
            # ----------------------------------------

            original_image = download_image(omr_sheet_url)

            if original_image is None:
                raise ValueError(
                    "Unable to load image"
                )

            # ----------------------------------------
            # Normalize Sheet
            # ----------------------------------------

            normalized_image = normalize_image(original_image)

            # logger.info("Image normalization completed")

            # ----------------------------------------
            # Detect Student ID
            # ----------------------------------------

            student_id = detect_student_id(normalized_image)

            # logger.info("Student ID detected: %s", student_id)

            # ----------------------------------------
            # Detect Answers
            # ----------------------------------------

            detected_answers = (detect_question_answers(normalized_image))

            # logger.info("Detected %s answers", len(detected_answers))

            

            # ----------------------------------------
            # Upload Evaluated Sheet
            # ----------------------------------------

            final_sheet_url = ""
            if upload_url is not None and len(correct_answers):
                # ----------------------------------------
                # Create Evaluated Sheet
                # ----------------------------------------

                evaluated_sheet = update_omr_sheet(normalized_image, detected_answers, correct_answers)

                if evaluated_sheet is not None:
                    final_sheet_url = upload_image(evaluated_sheet, upload_url)

                    # logger.info("Evaluated sheet uploaded")

            # ----------------------------------------
            # Calculate Score
            # ----------------------------------------

            # total_questions = len(correct_answers)

            # attempted = 0
            # correct = 0
            # incorrect = 0

            # for question_no, answer in (detected_answers.items()):

            #     if not answer:
            #         continue

            #     attempted += 1

            #     if (answer == correct_answers.get(question_no)):
            #         correct += 1
            #     else:
            #         incorrect += 1

            # score_percentage = 0.0

            # if total_questions > 0:
            #     score_percentage = round((correct / total_questions) * 100, 2)

            # ----------------------------------------
            # Success Response
            # ----------------------------------------

            return {
                "success": True,
                "message": Config.SUCCESS_MESSAGE,
                "data": {
                    "student_id": student_id,
                    "answers": detected_answers,
                    "final_sheet": final_sheet_url,
                    # "summary": {
                    #     "total_questions": total_questions,
                    #     "attempted": attempted,
                    #     "correct": correct,
                    #     "incorrect": incorrect,
                    #     "score_percentage": score_percentage
                    # }
                }
            }

        except Exception as e:

            logger.exception("OMR analysis failed")

            return {
                "success": False,
                "error": str(e)
            }
    
