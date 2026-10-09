from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rosenblatt_lab.experiments import run_experiment
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


