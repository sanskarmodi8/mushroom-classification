import gdown
import zipfile
from MushroomClassification import logger
from MushroomClassification.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_url
            zip_download_path = self.config.file_path
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_path}")
            
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            gdown.download(prefix+file_id,zip_download_path)
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_path}")
            
        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        with zipfile.ZipFile(self.config.file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)