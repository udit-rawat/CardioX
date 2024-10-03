from src.mlpro.pipeline.data_ingestion_pipe import DataIngestionTrainingPipeline
from src.mlpro import logger


StageName = "Data Ingestion Stage"
try:
    logger.info(f">>>>>>>> stage {StageName} Initiated <<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(
        f">>>>>>>> stage {StageName} Commenced <<<<<<<<<\n\nx=================x")
except Exception as e:
    logger.exception(e)
    raise e
