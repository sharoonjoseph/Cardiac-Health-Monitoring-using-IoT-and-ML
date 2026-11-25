[README.md](https://github.com/user-attachments/files/35410322/README.md)
# Cardiac Health Monitoring using IoT and Machine Learning

An academic prototype that combines **12-lead ECG classification** with an **IoT-style MQTT communication pipeline**. The project uses the PTB-XL dataset to compare traditional machine-learning models with a 1D convolutional neural network (CNN), then demonstrates how an ECG record can be published, classified, and stored through a distributed message-processing setup.

> **Disclaimer:** This is an academic research prototype, not a medical device. It has not been clinically validated and must not be used for diagnosis, treatment decisions, or emergency triage. Predictions are experimental model outputs, not confirmed diagnoses.

## Project overview

The system addresses the gap between offline ECG classification and the software pipeline needed to receive, transport, process, and store ECG records.

The project:
- Prepares PTB-XL version 1.0.3 for a five-class, single-label classification task.
- Compares an RBF Support Vector Machine (SVM), Random Forest (RF), and 1D CNN.
- Uses patient-aware PTB-XL folds for training, validation, and testing.
- Replays recorded ECG data over MQTT between two laptops connected through Tailscale.
- Exposes CNN inference through a FastAPI service and stores the returned prediction.
- Uses Node-RED to receive and inspect MQTT messages.

## Architecture

```text
PTB-XL ECG record (replay publisher)
              |
              v
        MQTT broker
        (Mosquitto)
              |
       +------+------+
       |             |
       v             v
 Python MQTT      Node-RED
 subscriber       MQTT input
       |          JSON parsing
       v          Debug inspection
 FastAPI CNN
 inference API
       |
       v
 Prediction storage
 (SQLite)
```

The verified inference path uses the **Python MQTT subscriber** to call the FastAPI CNN endpoint. Node-RED independently subscribes to the MQTT topic for message inspection.

The prototype was distributed across two Windows laptops connected using Tailscale:
- **Laptop 1:** Replays a PTB-XL ECG record and publishes it to MQTT.
- **Laptop 2:** Hosts Mosquitto, the Python subscriber, FastAPI inference, SQLite storage, and Node-RED.

The dashboard, automated alert/decision stage, and return path to an alert receiver were not completed.

## Dataset and classification task

**Dataset:** [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)

PTB-XL contains 12-lead, 10-second ECG recordings. This project uses the 100 Hz signals, with 1,000 samples per lead.

Records carrying exactly one mapped diagnostic superclass were retained:

| Label | Meaning |
|---|---|
| CD | Conduction disturbance |
| HYP | Hypertrophy |
| MI | Myocardial infarction |
| NORM | Normal ECG |
| STTC | ST/T change |

After filtering, the dataset contained **16,244 records from 14,627 patients**.

| Split | PTB-XL folds | Records |
|---|---:|---:|
| Training | 1–8 | 12,957 |
| Validation | 9 | 1,637 |
| Test | 10 | 1,650 |

The patient-aware folds help prevent records from the same patient from appearing across different splits.

## Models

Three model families were trained and evaluated on the same task and data splits:

1. **RBF SVM** — trained using 144 engineered features.
2. **Random Forest** — an ensemble model using 300 trees.
3. **1D CNN** — trained on the raw 12-lead ECG signal.

### Test-set results

Results on the held-out test fold (fold 10):

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| RBF SVM | 62.79% | 55.00% | 64.08% |
| Random Forest | 66.61% | 47.86% | 61.69% |
| 1D CNN | **74.24%** | **63.92%** | **74.42%** |

The 1D CNN achieved the strongest overall test performance among the three models. Hypertrophy (HYP) was the weakest class for the CNN, with an F1-score of 0.39.

## Technology stack

| Component | Technology |
|---|---|
| Programming language | Python 3.14.3 |
| ECG dataset and signal I/O | PTB-XL, WFDB |
| Traditional ML | scikit-learn |
| Deep learning | PyTorch |
| Numerical processing | NumPy, SciPy |
| Model persistence | joblib |
| Inference API | FastAPI |
| Messaging protocol | MQTT |
| MQTT broker | Eclipse Mosquitto 2.1.2 |
| Flow inspection | Node-RED 4.1.17 |
| Database | SQLite |
| Private network between laptops | Tailscale |

## Repository structure

The project report records the following project layout. Add or adjust paths to match the actual repository contents.

```text
Cardiac_Health_Monitoring/
├── dataset/
│   ├── raw/
│   │   └── ptb-xl/             # PTB-XL v1.0.3 (download separately)
│   └── processed/              # Processed signals, metadata and features
├── src/
│   ├── preprocessing/          # Label mapping, filtering, checks,
│   │                            # normalization and feature extraction
│   └── models/
│       ├── svm_model.py
│       ├── random_forest.py
│       ├── cnn_model.py
│       └── evaluate_models.py
└── README.md
```

The report also describes publisher, subscriber, and service components used in the IoT demonstration. Their exact filenames and full repository paths were not specified in the report; include them in this tree once confirmed in the source repository.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Cardiac_Health_Monitoring
```

Replace `<YOUR_REPOSITORY_URL>` with the actual repository URL.

### 2. Prepare the dataset

Download **PTB-XL v1.0.3** from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and place the extracted dataset under:

```text
dataset/raw/ptb-xl/
```

Preserve the original PTB-XL directory structure.

### 3. Install Python dependencies

The report records the following package versions:

```bash
pip install torch==2.14.1 scikit-learn==1.8.0 numpy==2.4.4 scipy==1.17.1 wfdb==4.3.1 joblib==1.5.3 fastapi
```

The reported PyTorch installation used the CUDA 13.0 build (`2.14.1+cu130`) from the corresponding PyTorch package index. Use a PyTorch build compatible with your machine and CUDA environment.

**Dependency note:** The report does not record the FastAPI version, ASGI server, or MQTT Python client library. These must be confirmed from the source code/environment and added to a project requirements file before the repository is considered fully reproducible. The exact script arguments and output filenames were also not recorded.

### 4. Install and configure the IoT tools

Install the following separately:
- Eclipse Mosquitto 2.1.2
- Node-RED 4.1.17
- Tailscale on both laptops

Join both laptops to the same Tailscale network. Configure Mosquitto on the monitoring laptop with client authentication and a listener reachable through its Tailscale address.

Do not commit broker passwords, password files, Tailscale keys, or other credentials. Keep them in local configuration or environment variables.

In Node-RED, configure an MQTT input for the broker and use a flow equivalent to:

```text
MQTT in (cardiac/patient01/vitals) -> JSON -> Debug
```

## Running the project

### Offline model pipeline

The report describes this workflow:

1. Run the preprocessing scripts in `src/preprocessing` to apply the single-superclass label rule, integrity checks, and official fold assignment.
2. Generate the processed signal arrays and 144-feature matrices.
3. Train the SVM using `svm_model.py`.
4. Train the Random Forest using `random_forest.py`.
5. Train the 1D CNN using `cnn_model.py`.
6. Evaluate the models on fold 10 using `evaluate_models.py`.

The report does not specify command-line arguments or exact output filenames. Check the scripts in the repository for their actual invocation syntax.

### End-to-end IoT demonstration

The verified demonstration sequence is:

1. Start the Mosquitto broker on the monitoring laptop.
2. Start the FastAPI application serving the trained CNN.
3. Start the Python MQTT subscriber.
4. Optionally deploy the Node-RED flow to inspect the same MQTT topic.
5. Run the PTB-XL replay publisher on the replay laptop.
6. Confirm that the subscriber receives the record, the inference service returns a prediction, and the prediction is saved.

The MQTT topic used in the demonstration is:

```text
cardiac/patient01/vitals
```

The replay message carries a record identifier and a 12-lead ECG signal containing 1,000 samples per lead. The report does not specify the complete JSON field schema.

## Verified integration example

For the demonstration record `TEST_PATIENT_001`, the report records the following inference response:

| Field | Recorded value |
|---|---|
| Predicted class | NORM |
| Confidence | 0.6732 |
| CD probability | 0.2216 |
| HYP probability | 0.0222 |
| MI probability | 0.0746 |
| NORM probability | 0.6732 |
| STTC probability | 0.0085 |
| Alert message | No abnormality flagged by the experimental model. |
| Database status | saved |

This is one verified end-to-end record, not a clinical assessment or a system-wide reliability result.

## Current implementation status

| Feature | Status reported |
|---|---|
| PTB-XL preprocessing and patient-aware splits | Achieved |
| SVM, Random Forest and CNN training/comparison | Achieved |
| MQTT communication between two laptops | Achieved for one record |
| Node-RED message parsing and inspection | Partly achieved |
| FastAPI CNN inference endpoint | Achieved |
| Saving predictions to the database | Partly achieved |
| Dashboard for ECG, predictions and system status | Not achieved |
| Automated alert/decision and return path | Partly achieved; alert wording only |

## Limitations

- The prototype replays public PTB-XL records; it does not acquire ECG from a live wearable or physical ECG sensor.
- The task is a five-class, single-label subset. Records with more than one mapped diagnostic superclass were excluded.
- The system has not been clinically validated and is not intended for patient care.
- The dashboard and automated alert/actuation path were not completed.
- Only one end-to-end integration record was verified; latency, reconnection, and malformed-message handling were not completed.
- Performance depends on the dataset labels, filtering rule, class distribution, preprocessing, split, and model configuration.

## Future work

Potential next steps identified by the project include completing the dashboard and alert workflow, improving the reliability of message handling, adding signal-quality assessment, and validating the system with appropriate live-acquisition hardware and clinically reviewed data.

## Team

Project submitted as part of the B.Tech in Computer Science and Engineering at **Amrita School of Computing, Amritapuri Campus**.

- Ajith Ashok
- Akshaya R
- M S Kashinath
- Pardhiv J R
- Sharoon Joseph

**Project guides:** Sangeeth & Ajai Mathew

## Acknowledgements

The project acknowledges the PTB-XL dataset and PhysioNet, and the open-source communities behind Eclipse Mosquitto, Node-RED, FastAPI, scikit-learn, PyTorch, NumPy, SciPy, and WFDB, along with the Tailscale network service.

## References and resources

- [PTB-XL v1.0.3 — PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)
- [WFDB Python](https://github.com/MIT-LCP/wfdb-python)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Node-RED documentation](https://nodered.org/docs/)
- [Tailscale documentation](https://tailscale.com/kb)
