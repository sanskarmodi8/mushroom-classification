import abc
import os
import zipfile

import gdown

from MushroomClassification import logger
from MushroomClassification.entity.config_entity import DataIngestionConfig


# Abstract Base Class
class AbstractDataIngestion(abc.ABC):
    def __init__(self, config: DataIngestionConfig):
        """
        Constructor for AbstractDataIngestion class
        :param config: DataIngestionConfig object that contains the dataset source URL and the path where the data is downloaded
        """
        self.config = config

    @abc.abstractmethod
    def download_file(self) -> str:
        """Download the file from the given source URL"""
        pass

    @abc.abstractmethod
    def extract_zip_file(self):
        """Extract the downloaded zip file"""
        pass


# Concrete Class that implements the abstract methods
class GDriveDataIngestion(AbstractDataIngestion):
    def download_file(self):
        """
        Fetch data from Google Drive URL
        """
        if os.path.exists(self.config.file_path):
            return
        else:
            try:
                dataset_url = self.config.source_url
                zip_download_path = self.config.file_path
                logger.info(
                    f"Downloading data from {dataset_url} into file {zip_download_path}"
                )

                file_id = dataset_url.split("/")[-2]
                prefix = "https://drive.google.com/uc?/export=download&id="

                gdown.download(prefix + file_id, zip_download_path)
                logger.info(
                    f"Downloaded data from {dataset_url} into file {zip_download_path}"
                )

            except Exception as e:
                logger.error(f"Error occurred while downloading data: {e}")
                raise e

    def extract_zip_file(self):
        """
        Extract the zip file into the configured data directory
        """
        try:
            unzip_path = self.config.unzip_dir
            with zipfile.ZipFile(self.config.file_path, "r") as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted zip file to {unzip_path}")
        except Exception as e:
            logger.error(f"Error occurred while extracting zip file: {e}")
            raise e


# Data Ingestion Factory Class
class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def create_data_ingestion(self) -> AbstractDataIngestion:
        """
        Factory method to return the appropriate DataIngestion class
        """
        if self.config.source_url.startswith("https://drive.google.com"):
            return GDriveDataIngestion(self.config)
        else:
            # Here, we can extend this logic for other sources like AWS S3, HTTP URLs, etc.
            raise ValueError("Unsupported data source URL")
