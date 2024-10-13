from src.mlpro.config.configuration import ConfigurationManager
from src.mlpro.components.model_train import ModelTrainer
from src.mlpro import logger

StageName = "Model Training Stage"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer_config = ModelTrainer(config=model_trainer_config)
            model_trainer_config.train()
        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {StageName} Initiated <<<<<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>>>> stage {StageName} Commenced <<<<<<<<<\n\nx=================x")
    except Exception as e:
        logger.exception(e)
        raise e
