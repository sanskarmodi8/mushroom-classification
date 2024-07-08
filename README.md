# Mushroom Classifier

This project uses [Mushroom Classification Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification) from Kaggle and aims to classify a Mushroom as edible or poisonous.

Deployment is done on Azure Portal, [click here](https://classifymushroom.azurewebsites.net/) to visit the deployed api.
The response '0' suggests that the mushroom is edible and response '1' suggests that the mushroom is poisonous.

This project mainly utilizes following tools and libraries :

- Numpy, Pandas, Scikit Learn (for data handling, pipelines, models, etc)
- MLFLOW and Dagshub (for experiment tracking and model registry)
- DVC (for pipeline versioning)
- FastAPI (for server)
- Streamlit application for user interface (alternative to fastapi app)
- Docker (for containerization)

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project follows a modular structure for better organization and maintainability. Here's an overview of the directory structure:

- `.github/workflows`: GitHub Actions workflows for CI/CD.
- `src/`: Source code directory.
  - `MushroomClassification/`
    - `components/`: Modules for different stages of the pipeline.
    - `utils/`: Utility functions.
    - `config/`: Configuration for each of the components.
    - `pipeline/`: Scripts for pipeline stages.
    - `entity/`: Data entity classes.
    - `constants/`: Constants used throughout the project.
- `config/`: Base Configuration for each stage of the project.
- `notebook/`: Directory for trials, experiments and prototype code in jupyter notebook.
- `fastapiApp.py`: FastAPI server.
- `streamlitApp.py`: Streamlit application.
- `Dockerfile`: Docker configuration for containerization.
- `requirements.txt`: Project dependencies.
- `setup.py`: Setup script for installing the project.
- `main.py`: Main script for execution of the complete pipeline.
- `params.yaml`: All the parameters used in the complete pipeline.
- `dvc.yaml`: Configuration file for DVC Pipeline Versioning.

## Setup

To set up the project environment, follow these steps:

1. Clone this repository.
2. Install Python 3.8 and ensure pip is installed.
3. Install project dependencies using `pip install -r requirements.txt`.
4. Ensure Docker is installed if you intend to use containerization.

## Usage

### To directly run the complete Data ingestion, Data cleaning, Model preparation and training and Model evaluation pipeline

run the command

```bash
dvc init
dvc repro
```

### To explicitly run each pipeline follow following commands-

#### Data Ingestion

To download and save the dataset, run:

```bash
python src/MushroomClassification/pipeline/stage_01_data_ingestion.py
```

#### Preprocessing the Data

To preprocess and save the cleaned data, run:

```bash
python src/MushroomClassification/pipeline/stage_02_data_transformation.py
```

#### Model Preparation and Training

To train the model, execute:

```bash
python src/MushroomClassification/pipeline/stage_03_model_training.py
```

#### Model Evaluation

To evaluate the trained model, run:

```bash
python src/MushroomClassification/pipeline/stage_04_model_evaluation.py
```

### To start the FastAPI server for making prediction :

Change the port to 8080 in app.py file and then,

```bash
python fastapiApp.py
```

### To start the Streamlit application :

```bash
streamlit run streamlitApp.py
```

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.

Please ensure that your contributions adhere to the project's coding standards.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
