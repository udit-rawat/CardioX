from src.mlpro.config.configuration import ConfigurationManager
from src.mlpro.components.data_transformation import DataTransformation
from src.mlpro import logger
StageName = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Initialize configuration and data transformation process
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(
                config=data_transformation_config)

            # Perform data transformation steps (preprocessing handles all necessary steps)
            X_train, X_test, y_train, y_test = data_transformation.train_test_splitting()

            logger.info(
                "Data transformation and splitting completed successfully.")

        except Exception as e:
            logger.exception("An error occurred during data transformation.")
            raise


if __name__ == '__main__':
    try:
        logger.info(f">>>>>>>> stage {StageName} Initiated <<<<<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>>>> stage {StageName} Commenced <<<<<<<<<\n\nx=================x")
    except Exception as e:
        logger.exception(e)
        raise e
