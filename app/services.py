import logging

import cv2
import numpy as np
import requests

from app.config import Config

logger = logging.getLogger(__name__)

"""
downloads and uploads image files.

"""

def download_image(image_url: str):
    """
    Download image from URL and convert to OpenCV image.
    """

    try:

        response = requests.get(
            image_url,
            timeout=Config.IMAGE_DOWNLOAD_TIMEOUT
        )

        response.raise_for_status()

        image = cv2.imdecode(
            np.frombuffer(response.content, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                "Downloaded file is not a valid image"
            )

        # logger.info("Image downloaded successfully")

        return image

    except requests.Timeout:
        logger.exception(
            "Image download timed out"
        )
        raise Exception(
            "Image download timed out"
        )

    except requests.RequestException as e:
        logger.exception(
            "Image download failed"
        )
        raise Exception(
            f"Image download failed: {str(e)}"
        )

    except Exception:
        logger.exception(
            "Unable to process downloaded image"
        )
        raise

def upload_image(image, upload_url = Config.FILE_UPLOAD_URL):
    """
    Upload OpenCV image and return uploaded URL.
    """

    try:

        success, buffer = cv2.imencode(
            ".jpg",
            image
        )

        if not success:
            raise Exception(
                "Failed to encode image"
            )

        files = {
            "upload_file": (
                "analysed_sheet.jpg",
                buffer.tobytes(),
                "image/jpeg"
            )
        }

        response = requests.post(
            upload_url,
            files=files,
            timeout=Config.IMAGE_UPLOAD_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        image_url = data.get("url")

        if not image_url:
            raise Exception(
                "Upload service did not return URL"
            )

        # logger.info("Image uploaded successfully")

        return image_url

    except requests.Timeout:
        # return ""
        logger.exception(
            "Image upload timed out"
        )
        raise Exception(
            "Image upload timed out"
        )

    except requests.RequestException as e:
        # return ""
        logger.exception(
            "Image upload failed"
        )
        raise Exception(
            f"Image upload failed: {str(e)}"
        )

    except Exception:
        # return ""
        logger.exception(
            "Failed to upload image"
        )
        raise

