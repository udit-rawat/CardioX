from src.mlpro.config.configuration import ConfigurationManager
from src.mlpro.components.data_ingestion import DataIngestion
from src.mlpro import logger

StageName = "Data Ingestion Pipeline"


class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {StageName} Initiated <<<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>>>> stage {StageName} Commenced <<<<<<<<<\n\nx=================x")
    except Exception as e:
        logger.exception(e)
        raise e
