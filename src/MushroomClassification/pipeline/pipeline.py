from zenml.pipelines import pipeline


@pipeline
def classification_pipeline(ingestion, transformation, training, evaluation):
    """
    The classification pipeline is the main entry point for the ZenML pipeline.
    It sequences together the ingestion, transformation, training, evaluation, and
    deployment steps.

    Each step is passed in as a separate function, and the outputs of each step
    are passed as arguments to the next step in the sequence.

    :param ingestion: Function that ingests the data
    :param transformation: Function that performs basic EDA and required preprocessing the data
    :param training: Function that trains the model
    :param evaluation: Function that evaluates the model
    :param deploy: Function that deploys the model
    """
    success = ingestion()
    success_2 = transformation(success)
    success_3 = training(success_2)
    success_4 = evaluation(success_3)
