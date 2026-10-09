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

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

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

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2 style={{ marginTop: 0 }}>Contents</h2>
        <ul>
          <li><a href="#article-top">(Top)</a></li>
          <li><a href="#theory">1 Theory (According to The Perceptron Research Paper by F. Rosenblatt)</a></li>
          <li><a href="#maths">2 Maths Calculations</a></li>
          <li><a href="#simulation">3 Simulation Part</a></li>
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
                  <div style={{ clear: 'both' }}></div>
        </div>
      </main>
    </div>
  );
}

export default App;
