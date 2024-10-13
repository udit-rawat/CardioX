from src.mlpro.pipeline.data_ingestion_pipe import DataIngestionTrainingPipeline
from src.mlpro.pipeline.data_validation_pipe import DataValidationTrainingPipeline
from src.mlpro.pipeline.data_transformation_pipe import DataTransformationTrainingPipeline
from src.mlpro.pipeline.model_trainer_pipe import ModelTrainerTrainingPipeline
from src.mlpro.pipeline.model_eval_pipe import ModelEvaluationTrainingPipeline
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
StageName = "Data Validation Stage"
try:
    logger.info(f">>>>>>>> stage {StageName} Initiated <<<<<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(
        f">>>>>>>> stage {StageName} Commenced <<<<<<<<<\n\nx=================x")
except Exception as e:
    logger.exception(e)
    raise e
StageName = "Data Transformation Stage"
try:
    logger.info(f">>>>>>>> stage {StageName} Initiated <<<<<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(
        f">>>>>>>> stage {StageName} Commenced <<<<<<<<<\n\nx=================x")
except Exception as e:
    logger.exception(e)
    raise e
STAGE_NAME = "Model Trainer stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = ModelTrainerTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} Commenced <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
STAGE_NAME = "Model evaluation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = ModelEvaluationTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
