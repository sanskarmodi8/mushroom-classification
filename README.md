# 🍄 Mushroom Classifier

## 📊 Project Overview

This project utilizes the [Mushroom Classification Dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification) from Kaggle to classify mushrooms as edible or poisonous. The deployed application is available on Streamlit Cloud.

🚀 [**Visit the Deployed App**](https://mushroom-classification.streamlit.app/)

![Mushroom Image](https://imgs.search.brave.com/FU44oOE0mC2dOCkKK6kFrPbH85ZCEmAkAVHUHcl9Pzk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzLzA2LzAy/LzU4LzA2MDI1ODQ5/NWNmMTNjM2JhMGJk/Y2E1NDBlY2Y5Mzhm/LmpwZw)


### 🛠️ Key Technologies

- **Data Handling & Machine Learning**: NumPy, Pandas, Scikit-learn
- **Experiment Tracking**: MLflow, DagsHub
- **Orchestration**: ZenML
- **User Interface**: Streamlit
- **Containerization**: Docker

## 📁 Project Structure

```
.
├── .github/workflows/      # GitHub Actions workflows
├── src/
│   └── MushroomClassification/
│       ├── components/     # Pipeline stage modules
│       ├── utils/          # Utility functions
│       ├── config/         # Component configurations
│       ├── pipeline/       # Pipeline stage scripts
│       ├── entity/         # Data entity classes
│       └── constants/      # Project constants
├── config/                 # Base configurations
├── notebook/               # Jupyter notebooks for experiments
├── streamlitApp.py         # Streamlit application
├── Dockerfile              # Docker configuration
├── requirements.txt        # Project dependencies
├── setup.py                # Project setup script
├── main.py                 # Main execution script
├── params.yaml             # Pipeline parameters
└── format.sh               # Formatting script
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Docker (optional, for containerization)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mushroom-classifier.git
   cd mushroom-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

On Ubuntu you can use `bash format.sh` to format the python files in the current directory.

### Running the Complete Pipeline

```bash
zenml init
python main.py
```

If you want to use MLFLOW then follow below commands :
```bash
zenml init
zenml experiment-tracker register mlflow_tracker --flavor=mlflow \
  --tracking_uri=<your_mlflow_uri> \
  --tracking_username=<your_mlflow_username> \
  --tracking_password=<your_mlflow_password>
zenml artifact-store register my_artifact_store --flavor=local
zenml orchestrator register my_orchestrator --flavor=local
zenml stack register mushroom_classification_stack \
  --orchestrator=my_orchestrator \
  --artifact-store=my_artifact_store
zenml stack set mushroom_classification_stack
python main.py
```

Also set `enable_mlflow_logging = True` in `src/MushroomClassification/pipeline/stage_03_model_training.py` and `src/MushroomClassification/pipeline/stage_04_model_evaluation.py` if you intend to use MLFLOW.

### Launching the Streamlit App

```bash
streamlit run streamlitApp.py
```

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code adheres to the project's coding standards.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Built with 💙 by Sanskar
