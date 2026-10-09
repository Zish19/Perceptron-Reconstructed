from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rosenblatt_lab.experiments import run_experiment, run_experiment_custom
from rosenblatt_lab.model import PerceptronConfig
import pandas as pd
import numpy as np
import io
import math
from rosenblatt_lab.model import PerceptronConfig
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
    assoc_exc = min(16, max(1, int(req.n_projection * 0.8)))
    assoc_inh = min(4, max(0, req.n_projection - assoc_exc))

    config = PerceptronConfig(
        retina_size=16,
        n_projection_units=req.n_projection,
        n_association_units=req.n_association,
        n_responses=req.n_responses,
        projection_threshold=req.threshold,
        association_threshold=1,
        association_excitatory_origins=assoc_exc,
        association_inhibitory_origins=assoc_inh,
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
    
    probs = result.model.calculate_probabilities(result.X_test)
    
    return {
        "initial_test_accuracy": result.initial_test_accuracy,
        "final_train_accuracy": result.final_train_accuracy,
        "final_test_accuracy": result.final_test_accuracy,
        "history": result.history,
        "test_confusion": result.test_confusion.tolist(),
        "probabilities": probs
    }

@app.post("/api/run-upload")
async def run_lab_experiment_upload(
    file: UploadFile = File(...),
    n_projection: int = Form(96),
    n_association: int = Form(256),
    n_responses: int = Form(2),
    threshold: int = Form(1),
    epochs: int = Form(20),
    learning_rate: float = Form(0.05)
):
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        y_col = df.columns[-1]
        X_df = df.drop(columns=[y_col])
        y_series = df[y_col]
        
        X_df = pd.get_dummies(X_df, drop_first=True)
        X_df = X_df.fillna(0)
        
        X_mat = X_df.astype(float).values
        means = np.mean(X_mat, axis=0)
        X_bin = (X_mat > means).astype(np.uint8)
        
        y_mat = pd.factorize(y_series)[0].astype(np.int64)
        
        num_features = X_bin.shape[1]
        retina_size = math.ceil(math.sqrt(num_features))
        required_features = retina_size ** 2
        
        if required_features > num_features:
            padding = np.zeros((X_bin.shape[0], required_features - num_features), dtype=np.uint8)
            X_bin = np.hstack((X_bin, padding))
            
        actual_n_projection = max(n_projection, 3)
        assoc_exc = min(16, max(1, int(actual_n_projection * 0.8)))
        assoc_inh = min(4, max(0, actual_n_projection - assoc_exc))

        config = PerceptronConfig(
            retina_size=retina_size,
            n_projection_units=actual_n_projection,
            n_association_units=n_association,
            n_responses=2,
            projection_threshold=threshold,
            association_threshold=1,
            association_excitatory_origins=assoc_exc,
            association_inhibitory_origins=assoc_inh,
            projection_locality="random",
            learning_rule="bivalent_gamma",
            disjoint_response_sources=True,
            learning_rate=learning_rate,
            seed=42,
        )
        
        result = run_experiment_custom(X=X_bin, y=y_mat, config=config, epochs=epochs)
        probs = result.model.calculate_probabilities(result.X_test)
        
        return {
            "initial_test_accuracy": result.initial_test_accuracy,
            "final_train_accuracy": result.final_train_accuracy,
            "final_test_accuracy": result.final_test_accuracy,
            "history": result.history,
            "test_confusion": result.test_confusion.tolist(),
            "probabilities": probs,
            "retina_size": retina_size
        }
    except Exception as e:
        return {"error": str(e)}


