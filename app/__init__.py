import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.routes import routes

def configure_logging():
    """
    Configure application logging.
    """
    
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(
        getattr(
            logging,
            Config.LOG_LEVEL.upper(),
            logging.INFO
        )
    )

    if root_logger.handlers:
        root_logger.handlers.clear()

    app_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5
    )

    app_handler.setFormatter(formatter)

    error_handler = RotatingFileHandler(
        "logs/error.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5
    )

    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)



    # logging.basicConfig(
    #     level=getattr(
    #         logging,
    #         Config.LOG_LEVEL.upper(),
    #         logging.INFO
    #     ),
    #     format=(
    #         "%(asctime)s | "
    #         "%(levelname)s | "
    #         "%(name)s | "
    #         "%(message)s"
    #     )
    # )



def create_app():
    """
    Flask Application Factory.
    """

    configure_logging()

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )

    app.config["JSON_SORT_KEYS"] = False

    CORS(app)

    app.register_blueprint(routes)

    return app

