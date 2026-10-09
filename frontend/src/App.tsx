import React, { useState } from 'react';
import axios from 'axios';
import './index.css';

interface Config {
  seed: number;
  epochs: number;
  samples: number;
  test_samples: number;
  n_projection: number;
  n_association: number;
  n_responses: number;
  threshold: number;
  learning_rate: number;
  noise: number;
  jitter: number;
  rule: string;
  locality: string;
  disjoint: boolean;
}

interface Probabilities {
  P_a_mean: number;
  P_a_std: number;
  P_e_mean: number;
  P_e_std: number;
}

interface Result {
  initial_test_accuracy: number;
  final_train_accuracy: number;
  final_test_accuracy: number;
  test_confusion: number[][];
  probabilities: Probabilities;
}

interface MLConfig {
  seed: number;
  select_k: number;
  noise_level: number;
  use_pca: boolean;
  pca_components: number;
  use_rbf: boolean;
  rbf_components: number;
  model_type: string;
}

interface MLResult {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion: number[][];
  final_features: number;
  best_params: Record<string, any>;
  error?: string;
}

function App() {
  const [loading, setLoading] = useState(false);
  const [loadingML, setLoadingML] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const [mlResult, setMLResult] = useState<MLResult | null>(null);
  const [file, setFile] = useState<File | null>(null);

  const [config, setConfig] = useState<Config>({
    seed: 7,
    epochs: 20,
    samples: 100,
    test_samples: 50,
    n_projection: 3,
    n_association: 4,
    n_responses: 2,
    threshold: 1,
    learning_rate: 0.05,
    noise: 0.02,
    jitter: 0,
    rule: 'bivalent_gamma',
    locality: 'local',
    disjoint: true
  });

  const [mlConfig, setMLConfig] = useState<MLConfig>({
    seed: 42,
    select_k: 5,
    noise_level: 0.01,
    use_pca: true,
    pca_components: 3,
    use_rbf: true,
    rbf_components: 50,
    model_type: "perceptron"
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    let finalValue: any = value;
    
    if (type === 'checkbox') {
      finalValue = (e.target as HTMLInputElement).checked;
    } else if (type === 'number' || type === 'range') {
      finalValue = Number(value);
    }

    setConfig(prev => ({
      ...prev,
      [name]: finalValue
    }));
  };

  const handleMLChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    let finalValue: any = value;
    
    if (type === 'checkbox') {
      finalValue = (e.target as HTMLInputElement).checked;
    } else if (type === 'number' || type === 'range') {
      finalValue = Number(value);
    }

    setMLConfig(prev => ({
      ...prev,
      [name]: finalValue
    }));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const runExperiment = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/api/run', config);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert('Simulation error. Ensure API runs on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const runMLExperiment = async () => {
    if (!file) {
      alert("Please upload a CSV dataset first.");
      return;
    }

    setLoadingML(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('seed', String(mlConfig.seed));
    formData.append('select_k', String(mlConfig.select_k));
    formData.append('noise_level', String(mlConfig.noise_level));
    formData.append('use_pca', String(mlConfig.use_pca));
    formData.append('pca_components', String(mlConfig.pca_components));
    formData.append('use_rbf', String(mlConfig.use_rbf));
    formData.append('rbf_components', String(mlConfig.rbf_components));
    formData.append('model_type', mlConfig.model_type);

    try {
      const res = await axios.post('http://localhost:8000/api/run-ml-upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setMLResult(res.data);
    } catch (err) {
      console.error(err);
      alert('ML Pipeline error. Ensure API runs on port 8000 and dataset is valid CSV.');
    } finally {
      setLoadingML(false);
    }
  };

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2 style={{ marginTop: 0 }}>Contents</h2>
        <ul>
          <li><a href="#article-top">(Top)</a></li>
          <li><a href="#theory">1 Theory (According to The Perceptron Research Paper by F. Rosenblatt)</a></li>
          <li><a href="#maths">2 Maths Calculations</a></li>
          <li><a href="#simulation">3 Simulation Part</a></li>
          <li><a href="#ipynb">4 The ipynb file result</a></li>
        </ul>
      </aside>

      <main className="main-content">
        <div>
          <h1 id="article-top">Perceptron Reconstructed</h1>
          <hr style={{ borderTop: '1px solid var(--border)', borderBottom: 'none' }}/>

          <h2 id="theory">Theory (According to The Perceptron Research Paper by F. Rosenblatt)</h2>
          <p>
            In his foundational 1958 research paper, Frank Rosenblatt introduced the <b>Photoperceptron</b>, a probabilistic model for information storage and organization in the brain. It consists of three discrete layers:
          </p>
          <ul>
            <li><b>Sensory units (S-points):</b> The input layer forming a retina.</li>
            <li><b>Association units (A-units):</b> Feature extractors connected to S-points with fixed, randomly distributed weights.</li>
            <li><b>Response units (R-units):</b> The final classification layer which adapts weights based on reinforcement rules.</li>
          </ul>
          <p>
            A key theoretical finding by Rosenblatt is the state space expansion. With <i>N</i> sensory points, there are 2<sup>N</sup> possible subsets of stimuli. The association layer randomly samples this immense phase space, making the learning problem linearly separable for the response layer.
          </p>
          <div className="diagram-container">
            <svg width="100%" height="200" viewBox="0 0 600 200" style={{ maxWidth: '600px', margin: '0 auto' }}>
              <rect x="50" y="50" width="80" height="100" rx="0" fill="#fff" stroke="#000" strokeWidth="2" />
              <text x="90" y="90" fill="#000" textAnchor="middle" style={{fontFamily: 'var(--font-mono)'}}>S-Units</text>
              <text x="90" y="120" fill="#000" textAnchor="middle" style={{fontFamily: 'var(--font-serif)', fontSize: '24px'}}>N</text>
              
              <rect x="250" y="20" width="80" height="160" rx="0" fill="#fff" stroke="#000" strokeWidth="2" />
              <text x="290" y="90" fill="#000" textAnchor="middle" style={{fontFamily: 'var(--font-mono)'}}>A-Units</text>
              <text x="290" y="120" fill="#000" textAnchor="middle" style={{fontFamily: 'var(--font-serif)', fontSize: '24px'}}>2^N</text>
              
              <rect x="450" y="70" width="80" height="60" rx="0" fill="#fff" stroke="#000" strokeWidth="2" />
              <text x="490" y="105" fill="#000" textAnchor="middle" style={{fontFamily: 'var(--font-mono)'}}>R-Units</text>

              <path d="M 130 100 Q 190 60 250 80" fill="none" stroke="#000" strokeWidth="1" strokeDasharray="4 4" />
              <path d="M 130 100 Q 190 100 250 100" fill="none" stroke="#000" strokeWidth="1" strokeDasharray="4 4" />
              <path d="M 130 100 Q 190 140 250 120" fill="none" stroke="#000" strokeWidth="1" strokeDasharray="4 4" />
              
              <path d="M 330 80 Q 390 90 450 100" fill="none" stroke="#000" strokeWidth="2" />
              <path d="M 330 100 Q 390 100 450 100" fill="none" stroke="#000" strokeWidth="2" />
              <path d="M 330 120 Q 390 110 450 100" fill="none" stroke="#000" strokeWidth="2" />
            </svg>
          </div>

          <h2 id="maths">Maths Calculations</h2>
          <p>
            The activation probability of an A-unit (<i>P<sub>a</sub></i>) and the expected active proportion (<i>P<sub>e</sub></i>) are crucial to understanding the network's capacity.
          </p>
          <p>
            <b>1. A-Unit Activation (Threshold Rule):</b>
            An A-unit fires if the sum of its inputs exceeds threshold <i>θ</i>:
            <br />
            <code style={{fontFamily: 'var(--font-mono)'}}>a<sub>j</sub> = 1 if Σ(w<sub>ij</sub> * s<sub>i</sub>) &ge; θ else 0</code>
          </p>
          <p>
            <b>2. Probability of Activation (<i>P<sub>a</sub></i>):</b>
            If sensory connections are drawn binomially, the probability an A-unit is active is defined by the cumulative binomial distribution intersecting the threshold <i>θ</i>.
          </p>
          <p>
            <b>3. Weight Update Rule (Reinforcement):</b>
            The bivalent gamma rule strictly updates weights connecting A-units to R-units when an error occurs:
            <br />
            <code style={{fontFamily: 'var(--font-mono)'}}>Δv<sub>jk</sub> = η * (R<sub>true</sub> - R<sub>pred</sub>) * a<sub>j</sub></code>
          </p>
          <p>
            The results of these probabilistic calculations are displayed dynamically in the Simulation Part.
          </p>

          <h2 id="simulation">Simulation Part</h2>
          <div className="sim-card" style={{borderLeft: '4px solid #ef4444'}}>
            <h1 style={{marginTop: 0, borderBottom: 'none'}}>Interactive Neural Network</h1>
            <p>Visualize how neural networks process information through layers of interconnected neurons.</p>
            
            <div className="sim-card" style={{background: '#fafafa', marginTop: '1rem'}}>
              <h3>Network Configuration</h3>
              <div className="sim-controls">
                <div className="slider-group">
                  <label>Input Neurons: {config.n_projection}</label>
                  <input type="range" name="n_projection" min="1" max="6" value={config.n_projection} onChange={handleChange} />
                </div>
                <div className="slider-group">
                  <label>Hidden Neurons: {config.n_association}</label>
                  <input type="range" name="n_association" min="1" max="6" value={config.n_association} onChange={handleChange} />
                </div>
                <div className="slider-group">
                  <label>Output Neurons: {config.n_responses}</label>
                  <input type="range" name="n_responses" min="1" max="4" value={config.n_responses} onChange={handleChange} />
                </div>
              </div>
            </div>

            <div className="diagram-container" style={{padding: '2rem 1rem'}}>
              <div style={{display: 'flex', justifyContent: 'space-around', marginBottom: '1rem', fontWeight: 'bold'}}>
                <span>Input Layer</span>
                <span>Hidden Layer</span>
                <span>Output Layer</span>
              </div>
              <svg width="100%" height="300" viewBox="0 0 600 300">
                {/* Draw edges */}
                {Array.from({ length: config.n_projection }).map((_, i) =>
                  Array.from({ length: config.n_association }).map((_, j) => (
                    <line 
                      key={`e1-${i}-${j}`} 
                      x1="100" 
                      y1={150 + (i - (config.n_projection - 1) / 2) * 50} 
                      x2="300" 
                      y2={150 + (j - (config.n_association - 1) / 2) * 50} 
                      stroke="#cbd5e1" 
                      strokeWidth="2" 
                    />
                  ))
                )}
                {Array.from({ length: config.n_association }).map((_, i) =>
                  Array.from({ length: config.n_responses }).map((_, j) => (
                    <line 
                      key={`e2-${i}-${j}`} 
                      x1="300" 
                      y1={150 + (i - (config.n_association - 1) / 2) * 50} 
                      x2="500" 
                      y2={150 + (j - (config.n_responses - 1) / 2) * 50} 
                      stroke="#cbd5e1" 
                      strokeWidth="2" 
                    />
                  ))
                )}
                {/* Draw Input nodes */}
                {Array.from({ length: config.n_projection }).map((_, i) => (
                  <g key={`in-${i}`}>
                    <circle cx="100" cy={150 + (i - (config.n_projection - 1) / 2) * 50} r="16" fill="#fb7185" stroke="#000" />
                    <text x="100" y={154 + (i - (config.n_projection - 1) / 2) * 50} textAnchor="middle" fill="#fff" fontSize="12">{i+1}</text>
                  </g>
                ))}
                {/* Draw Hidden nodes */}
                {Array.from({ length: config.n_association }).map((_, i) => (
                  <g key={`hid-${i}`}>
                    <circle cx="300" cy={150 + (i - (config.n_association - 1) / 2) * 50} r="16" fill="#fb7185" stroke="#000" />
                    <text x="300" y={154 + (i - (config.n_association - 1) / 2) * 50} textAnchor="middle" fill="#fff" fontSize="12">{i+1}</text>
                  </g>
                ))}
                {/* Draw Output nodes */}
                {Array.from({ length: config.n_responses }).map((_, i) => (
                  <g key={`out-${i}`}>
                    <circle cx="500" cy={150 + (i - (config.n_responses - 1) / 2) * 50} r="16" fill="#fb7185" stroke="#000" />
                    <text x="500" y={154 + (i - (config.n_responses - 1) / 2) * 50} textAnchor="middle" fill="#fff" fontSize="12">{i+1}</text>
                  </g>
                ))}
              </svg>
            </div>

            <div className="info-box">
              <strong><span style={{marginRight: '0.5rem'}}>🛈</span> Network Weights</strong>
              <p style={{margin: '0.5rem 0 0 0'}}>Weights determine the strength of connections between neurons. During training, these weights are adjusted to minimize error.</p>
              <p style={{margin: '0.5rem 0 0 0', fontSize: '0.9rem'}}>Click on connections to see weight values</p>
            </div>
          </div>

          <div>
            {result && !loading ? (
              <div>
                <p>The simulation yielded the following metrics over <b>{config.epochs} epochs</b>:</p>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <h4>Test Accuracy</h4>
                    <p style={{fontSize: '2rem', margin: 0}}>{(result.final_test_accuracy * 100).toFixed(1)}%</p>
                  </div>
                </div>

                <h3>Probability Models</h3>
                <ul>
                  <li><b>Probability A-Unit Active (<i>P<sub>a</sub></i>):</b> {result.probabilities.P_a_mean.toFixed(4)} (&plusmn;{result.probabilities.P_a_std.toFixed(4)})</li>
                  <li><b>Expected Active Proportion (<i>P<sub>e</sub></i>):</b> {result.probabilities.P_e_mean.toFixed(4)} (&plusmn;{result.probabilities.P_e_std.toFixed(4)})</li>
                </ul>
              </div>
            ) : (
              <p><i>Execute the simulation from the infobox to generate empirical results.</i></p>
            )}
          </div>
          <div style={{ clear: 'both' }}></div>


          <h2 id="ipynb">The ipynb file result</h2>
          <p>
            Upload a dataset (CSV) to run the modern Scikit-Learn Perceptron pipeline (incorporating feature selection, dimensionality reduction via PCA, noise augmentation, and RBF kernel approximations) adapted directly from the exploratory ipynb file.
          </p>
          
          <div className="infobox" style={{ float: 'left', marginRight: '2rem', marginLeft: 0 }}>
            <h3 style={{ marginTop: 0, textAlign: 'center' }}>ML Pipeline Config</h3>
            <table>
              <tbody>
                <tr>
                  <td colSpan={2}>
                    <b>Upload Dataset (CSV)</b><br />
                    <small><i>Note: The last column must be the target label.</i></small>
                    <input type="file" accept=".csv" onChange={handleFileChange} style={{marginTop: '0.5rem'}} />
                  </td>
                </tr>
                <tr>
                  <td>Model Type</td>
                  <td>
                    <select name="model_type" value={mlConfig.model_type} onChange={handleMLChange}>
                      <option value="perceptron">Perceptron (GridSearchCV)</option>
                    </select>
                  </td>
                </tr>
                <tr>
                  <td>Select K Best</td>
                  <td><input type="number" name="select_k" value={mlConfig.select_k} onChange={handleMLChange} /></td>
                </tr>
                <tr>
                  <td>Noise Level</td>
                  <td><input type="number" name="noise_level" step="0.01" value={mlConfig.noise_level} onChange={handleMLChange} /></td>
                </tr>
                <tr>
                  <td>Use PCA</td>
                  <td><input type="checkbox" name="use_pca" checked={mlConfig.use_pca} onChange={handleMLChange} /></td>
                </tr>
                <tr>
                  <td>Use RBF Kernel</td>
                  <td><input type="checkbox" name="use_rbf" checked={mlConfig.use_rbf} onChange={handleMLChange} /></td>
                </tr>
              </tbody>
            </table>
            <div style={{ padding: '1rem' }}>
              <button 
                className="btn" 
                onClick={runMLExperiment} 
                disabled={loadingML}
              >
                {loadingML ? 'Running Pipeline...' : 'Run Pipeline'}
              </button>
            </div>
          </div>

          <div style={{ overflow: 'hidden' }}>
            {mlResult && !loadingML ? (
              mlResult.error ? (
                <div style={{ color: 'red' }}><b>Error:</b> {mlResult.error}</div>
              ) : (
                <div>
                  <p>The modern pipeline on your uploaded dataset yielded the following metrics:</p>
                  <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
                    <div className="metric-card">
                      <h4>Accuracy</h4>
                      <p style={{fontSize: '2rem', margin: 0}}>{(mlResult.accuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div className="metric-card">
                      <h4>F1 Score</h4>
                      <p style={{fontSize: '2rem', margin: 0}}>{(mlResult.f1 * 100).toFixed(1)}%</p>
                    </div>
                  </div>

                  <h3>Pipeline Insights</h3>
                  <ul>
                    <li><b>Final Feature Count:</b> {mlResult.final_features} (after Selection, PCA & RBF)</li>
                    <li><b>Best Params Found:</b> {JSON.stringify(mlResult.best_params)}</li>
                    <li><b>Precision:</b> {(mlResult.precision * 100).toFixed(1)}%</li>
                    <li><b>Recall:</b> {(mlResult.recall * 100).toFixed(1)}%</li>
                  </ul>
                  
                  <h3>Logic Behind the Result</h3>
                  <p>
                    When you run the pipeline, the dataset goes through a sequence of advanced machine learning transformations:
                  </p>
                  <ol>
                    <li><b>Preprocessing:</b> Categorical text features (like 'Electronics') are automatically mapped to binary columns (one-hot encoding) and standard scaling is applied to normalize variances.</li>
                    <li><b>Feature Selection (SelectKBest):</b> The pipeline analyzes the statistical relationship between each feature and the target, selecting only the top K most informative features.</li>
                    <li><b>Data Augmentation:</b> A small degree of Gaussian noise is injected into the training samples. This forces the model to generalize rather than memorizing the training data.</li>
                    <li><b>Dimensionality Reduction (PCA):</b> If enabled, Principal Component Analysis compresses the feature space into the most dominant orthogonal components, removing collinearity.</li>
                    <li><b>Non-Linear Mapping (RBF):</b> If enabled, an RBF Kernel approximation maps the input into a high-dimensional space, allowing a linear perceptron to solve non-linear problems.</li>
                    <li><b>Hyperparameter Tuning:</b> Finally, the Perceptron explores various regularization penalties (L1, L2, ElasticNet) using Cross-Validation (`GridSearchCV`) to find the most optimal state (shown in Best Params).</li>
                  </ol>
                  
                  <div style={{ clear: 'both' }}></div>
                </div>
              )
            ) : (
              <p style={{ clear: 'both' }}><i>Upload a dataset and execute the ML pipeline to see advanced metrics.</i></p>
            )}
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
