from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rosenblatt_lab.experiments import run_experiment
from rosenblatt_lab.model import PerceptronConfig
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "ML-Perceptron-Experiment"))
from ML_Perceptron_Implementation import run_ml_pipeline

app = FastAPI(title="Perceptron Reconstructed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExperimentRequest(BaseModel):
    seed: int = 7
    epochs: int = 20
    samples: int = 100
    test_samples: int = 50
    n_projection: int = 96
    n_association: int = 256
    n_responses: int = 2
    threshold: int = 1
    learning_rate: float = 0.05
    noise: float = 0.02
    jitter: int = 0
    rule: str = "bivalent_gamma"
    locality: str = "local"
    disjoint: bool = True

@app.post("/api/run")
def run_lab_experiment(req: ExperimentRequest):
    config = PerceptronConfig(
        retina_size=16,
        n_projection_units=req.n_projection,
        n_association_units=req.n_association,
        n_responses=req.n_responses,
        projection_threshold=req.threshold,
        association_threshold=1,
        projection_locality=req.locality,  # type: ignore
        learning_rule=req.rule,            # type: ignore
        disjoint_response_sources=req.disjoint,
        learning_rate=req.learning_rate,
        seed=req.seed,
    )
    result = run_experiment(
        config=config,
        epochs=req.epochs,
        train_samples_per_class=req.samples,
        test_samples_per_class=req.test_samples,
        stimulus_noise=req.noise,
        stimulus_jitter=req.jitter,
        seed=req.seed,
    )
    
    # Calculate P_a and P_e
    probs = result.model.calculate_probabilities(result.X_test)
    
    return {
        "initial_test_accuracy": result.initial_test_accuracy,
        "final_train_accuracy": result.final_train_accuracy,
        "final_test_accuracy": result.final_test_accuracy,
        "history": result.history,
        "test_confusion": result.test_confusion.tolist(),
        "probabilities": probs
    }

class MLRequest(BaseModel):
    seed: int = 42
    samples: int = 1000
    features: int = 20
    select_k: int = 5
    noise_level: float = 0.01
    use_pca: bool = False
    pca_components: int = 3
    use_rbf: bool = False
    rbf_components: int = 50
    model_type: str = "perceptron"

@app.post("/api/run-ml")
def run_ml_experiment(req: MLRequest):
    config = req.dict()
    metrics = run_ml_pipeline(config)
    return metrics

@app.post("/api/run-ml-upload")
async def run_ml_experiment_upload(
    file: UploadFile = File(...),
    seed: int = Form(42),
    select_k: int = Form(5),
    noise_level: float = Form(0.01),
    use_pca: bool = Form(False),
    pca_components: int = Form(3),
    use_rbf: bool = Form(False),
    rbf_components: int = Form(50),
    model_type: str = Form("perceptron")
):
    # Parse CSV file
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Assume last column is target, remaining are features
        y_col = df.columns[-1]
        X_df = df.drop(columns=[y_col])
        y_series = df[y_col]
        
        # Convert categorical features using one-hot encoding
        X_df = pd.get_dummies(X_df, drop_first=True)
        # Fill missing values if any
        X_df = X_df.fillna(0)
        
        X_ext = X_df.astype(float).values
        
        # Label encode target
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_ext = le.fit_transform(y_series)
        
        config = {
            "seed": seed,
            "select_k": select_k,
            "noise_level": noise_level,
            "use_pca": use_pca,
            "pca_components": pca_components,
            "use_rbf": use_rbf,
            "rbf_components": rbf_components,
            "model_type": model_type
        }
        
        metrics = run_ml_pipeline(config, X_ext=X_ext, y_ext=y_ext)
        return metrics
    except Exception as e:
        return {"error": str(e)}
