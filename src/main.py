from models.train import train_model
import os
from dotenv import load_dotenv
import logging

if __name__ == "__main__":
    try:
        load_dotenv(dotenv_path="../.env")
        os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    except FileNotFoundError:
        logging.error('Could not find WANDB_API_KEY')
        raise FileNotFoundError

    train_model()