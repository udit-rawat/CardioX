import os
import requests
import zipfile
from src.mlpro import logger
from src.mlpro.utils.common import get_size
from src.mlpro.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            try:
                response = requests.get(self.config.source_URL)
                response.raise_for_status()

                with open(self.config.local_data_file, 'wb') as file:
                    file.write(response.content)

                logger.info(
                    f"File downloaded to {self.config.local_data_file}")
                logger.info(f"Response headers: {response.headers}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error downloading file: {str(e)}")
                raise
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
