from src.mlpro.config.configuration import ConfigurationManager
from src.mlpro.components.data_validation import DataValidation
from src.mlpro import logger

StageName = "Data Validation Stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_columns()

        except Exception as e:
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {StageName} Initiated <<<<<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>>>> stage {StageName} Commenced <<<<<<<<<\n\nx=================x")
    except Exception as e:
        logger.exception(e)
        raise e
