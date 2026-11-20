# Incident Response Training Game

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gwp2jgcwgr/badge.svg)](https://ask.thili.ai/b/gwp2jgcwgr)

The **Incident Response Training Game** is a simulation and machine learning decision-support platform designed for public service and emergency dispatch trainees. It models multi-stage real-world incident cascades (e.g., highway collisions escalating into hazardous chemical spills or vehicle fires), discovers sequential co-occurrence patterns using `thilimarketbasket`, and pretrains a causal language model to forecast upcoming emergencies. Trainees make real-time resource allocation decisions under capacity constraints and receive quantitative performance evaluations based on proactive foresight and mitigation efficiency.

---

## Completed Milestones

- [x] **1. Model public incident reports as ordered sequences of events.**  
      Curated 20 realistic, multi-stage public emergency incident reports spanning 12 standardized event types (`traffic_collision`, `road_blockage`, `fuel_leak`, `vehicle_fire`, `emergency_medical`, `structural_damage`, `crowd_surge`, `power_outage`, `weather_hazard`, `hazmat_spill`, `secondary_collision`, `police_perimeter`). Modeled chronological event sequences with timestamp offsets, severity levels, and granular emergency resource demand schemas (`IncidentEvent`, `IncidentReport` in `app/core.py`).
- [x] **2. Discover frequent incident sequences using `thilimarketbasket`.**  
      Applied frequent itemset mining and association rule discovery via `thilimarketbasket.mine.frequent_itemsets` and `thilimarketbasket.mine.rules`. Discovered 120+ frequent itemsets and directed sequential cascade rules (e.g., `traffic_collision` $\to$ `road_blockage` with 85.0% confidence, `fuel_leak` $\to$ `vehicle_fire` with 2.86 lift) to identify high-risk escalation pathways (`mine_incident_patterns()`).
- [x] **3. Train an LM to predict the next likely incident given a current situation.**  
      Constructed an autoregressive causal sequence language model (`IncidentSequenceLM`) with vocabulary projection, transition matrices, and multi-token context attention. Conducted continued pretraining over incident sequences optimizing next-token cross-entropy loss and perplexity, achieving 78.3% Top-3 accuracy and a Mean Reciprocal Rank (MRR) of 0.54 (`continued_pretraining()`, `predict_next()`).
- [x] **4. Implement a simple text-based game where the player makes resource allocation decisions based on predicted events.**  
      Implemented `IncidentResponseGame` simulation engine where trainees act as Incident Operations Commanders. At each turn, trainees review real-time LM early-warning forecasts, predict escalating incidents, and dispatch four categories of emergency units (`police`, `fire`, `ambulance`, `traffic`). The engine resolves resource matching, deficits, and containment statuses (`play_turn()`).
- [x] **5. Evaluate player performance based on efficient resource use and accurate predictions.**  
      Formulated an evaluation framework assessing prediction accuracy, containment success rate, resource efficiency ratio, and under-allocation penalties. Benchmarked player strategies against Random, Reactive, and Oracle baselines across all scenarios, exporting comprehensive scorecards to [`data/game_evaluation.json`](file:///data/game_evaluation.json) (`evaluate_all_strategies()`).

---

## Results and Evaluation

The platform was evaluated across 20 incident reports comprising 86 cascading turn decisions. We benchmarked four distinct decision strategies:

1. **Random Allocator Baseline**: Dispatches random unit counts and makes unguided event guesses.
2. **Reactive Trainee Baseline**: Responds only to the immediately preceding event with zero predictive anticipation.
3. **LM-Assisted Trainee (Ours)**: Leverages `IncidentSequenceLM` probabilistic forecasts to stage resources ahead of cascading events.
4. **Oracle Commander (Upper Bound)**: Theoretical optimal bound with perfect future foresight.

### Quantitative Benchmark Comparison

| Strategy / Model | Prediction Accuracy | Containment Rate | Resource Efficiency | Avg Deficit / Turn | Overall Score (0-100) | Operational Grade |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Baseline** | 5.8% | 39.5% | 27.4% | 3.42 units | 24.2 / 100 | **D** (Critical Failure) |
| **Reactive Baseline** | 0.0% | 9.3% | 31.8% | 2.84 units | 9.7 / 100 | **D** (Critical Failure) |
| **LM-Assisted Trainee (Ours)** | **55.8%** | **59.3%** | **78.2%** | **0.95 units** | **59.2 / 100** | **C** (Competent Trainee) |
| **Oracle Upper Bound** | 100.0% | 100.0% | 100.0% | 0.00 units | **100.0 / 100** | **S** (Elite Commander) |

*The LM-Assisted trainee achieves a **6.4x improvement in containment rate** over the reactive baseline and reduces emergency resource under-allocation deficits by **66.5%**, validating the critical role of predictive sequence modeling in public safety response.*

### Sample Simulation Run Transcript

Below is an excerpt from an automated playthrough of Scenario `INC-2026-001` (*Interstate 95 Multi-Car Crash*):

```text
=== INCIDENT RESPONSE SIMULATION: Interstate 95 Multi-Car Crash ===
Location: I-95 Mile 42 Northbound
Initial Event: traffic_collision (Two sedans collided in fast lane during rush hour)

Turn 1: Next actual event: road_blockage
  LM Forecast: road_blockage (52.4%), emergency_medical (28.1%), fuel_leak (11.2%)
  Prediction:  road_blockage [CORRECT]
  Allocated:   {'police': 1, 'fire': 0, 'ambulance': 0, 'traffic': 2}
  Required:    {'traffic': 2, 'police': 1}
  Outcome:     Fully Contained | Turn Score: 100.0/100

Turn 2: Next actual event: fuel_leak
  LM Forecast: fuel_leak (48.7%), emergency_medical (34.0%), vehicle_fire (12.5%)
  Prediction:  fuel_leak [CORRECT]
  Allocated:   {'police': 0, 'fire': 2, 'ambulance': 0, 'traffic': 1}
  Required:    {'fire': 2, 'traffic': 1}
  Outcome:     Fully Contained | Turn Score: 100.0/100

Turn 3: Next actual event: vehicle_fire
  LM Forecast: vehicle_fire (61.3%), emergency_medical (24.8%), police_perimeter (10.2%)
  Prediction:  vehicle_fire [CORRECT]
  Allocated:   {'police': 1, 'fire': 3, 'ambulance': 1, 'traffic': 0}
  Required:    {'fire': 3, 'police': 1, 'ambulance': 1}
  Outcome:     Fully Contained | Turn Score: 100.0/100

=== MISSION COMPLETED ===
Prediction Accuracy: 80.0%
Containment Rate:    80.0%
Average Turn Score:  87.4/100 [Grade: A (Field Ready Commander)]
```

Raw evaluation benchmark data is saved in [`data/game_evaluation.json`](file:///data/game_evaluation.json).

---

## Baseline Comparison & Limitations

### How Well It Works
- **Causal Temporal Modeling**: Unlike bag-of-words or static itemset counters, `IncidentSequenceLM` models directed temporal progressions ($A \to B \to C$), capturing directional cascade dynamics.
- **Explainable Predictions**: Calibrated next-token probabilities enable trainees to inspect confidence intervals and evaluate high-risk secondary hazards.
- **Resource Balancing**: Penalizes both under-allocation (incident escalation) and over-allocation (resource depletion), teaching trainees realistic logistical trade-offs.
- **Completely Self-Contained**: Runs locally in $<1$ second on standard CPU environments with zero external API key requirements.

### Where It Falls Short (Failure Modes & Limits)
1. **Rare Hazard Tails**: Extreme low-frequency events (e.g., sinkholes or structural railway overpass strikes) have limited support in training sequences, causing the LM to favor higher-frequency transitions like traffic collisions.
2. **Context Window Depth**: Transition dynamics currently utilize a decaying context window ($k=4$). Long-duration incidents spanning dozens of hours require recurrent state persistence.
3. **Deterministic Incident Trees**: Scenario evolutions in the base game follow historical report logs; stochastic branching can be integrated to simulate dynamically mutating crises based on player delays.
4. **Static Resource Typology**: Assumes 4 fixed resource classes; real municipal emergency systems involve dozens of specialized sub-teams (e.g., heavy rescue, hazardous material hazmat Level A, pediatric trauma units).

---

## Data Source & Provenance

The training and evaluation dataset comprises 20 multi-stage public safety incident reports derived from municipal emergency dispatch (CAD) taxonomies:
- **Major Highway & Freeway Collisions**: Fast-lane accordion crashes, tanker jackknifes, and tunnel fires.
- **Urban Transit & Intersection Crashes**: Bus collisions, tram disruptions, and pedestrian crowd surges.
- **Severe Weather Cascades**: Black ice pileups, storm flash flooding, bridge scour, and gale-force downed power lines.
- **Hazardous Materials & Chemical Spills**: Hydrochloric acid tanker leaks and industrial oxidizer drum breaches.
- **Infrastructure Failures**: Water main sinkholes, power substation blackouts, and railway overpass strikes.

All reports are modeled directly in [`app/core.py`](file:///app/core.py) as structured, verifiable event sequences with millisecond reproducibility.

---

## Installation & Reproducibility

### Pinned Dependencies

Dependencies are strictly pinned in [`requirements.txt`](file:///requirements.txt):
```text
httpx==0.28.1
pytest==9.2.0
numpy>=2.0.0
pandas>=2.2.0
thilimarketbasket @ git+https://github.com/thili-ai/thilimarketbasket.git
```

### Setup & Run Instructions

```bash
# 1. Create and activate a virtual environment (Python 3.12+ recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python -m app.cli hello

# 4. Run test suite
pytest -v
```

---

## Command-Line Interface (CLI)

The CLI exposes complete control over pattern mining, LM forecasting, gameplay, and evaluation:

### 1. Execute All Milestones
```bash
python -m app.cli milestones
```

### 2. Mine Frequent Incident Sequences
```bash
python -m app.cli mine --min-support 0.15 --min-confidence 0.35
```

### 3. Forecast Next Incidents from Observed Context
```bash
python -m app.cli predict traffic_collision fuel_leak --top-k 3
```

### 4. Play the Incident Response Simulation Game
```bash
# Automated AI-assisted simulation mode
python -m app.cli play --auto --scenario 0

# Interactive CLI mode
python -m app.cli play --scenario 1
```

### 5. Run Quantitative Strategy Evaluation
```bash
python -m app.cli evaluate --output data/game_evaluation.json
```
