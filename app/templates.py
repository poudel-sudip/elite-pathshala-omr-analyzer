import json
import logging

from app.config import Config

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Loads and caches all OMR template files.

    Templates are loaded only once during application startup
    and then reused throughout the application lifetime.
    """

    def __init__(self):
        self.corner_markers = self._load_json(
            Config.TEMPLATE_CORNER_MARKERS
        )

        self.student_id_bubbles = self._load_json(
            Config.TEMPLATE_STUDENT_ID_BUBBLES
        )

        self.question_bubbles = self._load_json(
            Config.TEMPLATE_QUESTION_BUBBLES
        )

        # logger.info("OMR templates loaded successfully")

    @staticmethod
    def _load_json(file_path: str) -> dict:
        """
        Load JSON file safely.
        """

        try:
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except FileNotFoundError:
            logger.exception(
                "Template file not found: %s",
                file_path
            )
            raise

        except json.JSONDecodeError:
            logger.exception(
                "Invalid JSON file: %s",
                file_path
            )
            raise

        except Exception:
            logger.exception(
                "Failed to load template: %s",
                file_path
            )
            raise


# =====================================================
# Singleton Instance
# =====================================================

templates = TemplateManager()