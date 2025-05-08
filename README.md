# **P-CAFE**

**P-CAFE** is a Python library designed for feature selection (FS) in electronic health record (EHR) datasets.

This package is based on the following paper-
## **Key Contributions**

## **Personalized Cost-Aware Feature Selection (FS):**  
  A novel method tailored for EHR datasets, addressing challenges such as:  
  - Multiple data modalities  
  - Missing values  
  - Cost-aware decision-making
  - Personalized and Iterative FS
  - Time series data

## **Building Benchmarks**

To generate the benchmark datasets:  
- **MIMIC-III Numeric**  
- **MIMIC-III with Costs**  
- **MIMIC-III Multi-Modal Dataset**  
- **eICU Dataset**

Navigate to the **`data`** directory for instructions.



> **Important:**  
> The MIMIC-III and eICU data are not provided. You must acquire the data independently from [MIMIC-III on PhysioNet](https://mimic.physionet.org/), [eICU on PhysioNet](https://physionet.org/content/eicu-crd/2.0/).

## **Running the Code**

1. run the requrments.txt file to install the required packages.

2. Dataset Configuration

Open `embedder_guesser.py` and choose your dataset by modifying the `--data` argument in the `FLAG` section.

Supported datasets:
- `pcafe_utils.load_time_Series()` – eICU time series data
- `pcafe_utils.load_mimic_text()` – MIMIC-III multimodal data (includes clinical text)
- `pcafe_utils.load_mimic_time_series()` – MIMIC-III numeric time series

Define the feature costs by setting `self.cost_list` in the `MultimodalGuesser` class.


3. Running the embedder_guesser Module

4. For the DDQN agent run **`main_robust.py`**, for other agent run **`main_sb3.py`** and choose the RL agent.


## **Examples**
![Qualitative Example](images/p-cafe-image-figure1.png)
 As shown in the Figure,  P-CAFE iteratively selects features in stages, personalizing the FS process to individual cases. It integrates demographic, laboratory, categorical, and textual data iteratively.



![Clinical Interpretability](images/image2.png)
We conducted an experiment on the Diabetes Prediction dataset.  
The figure shows that for patients with a high blood glucose level and patient-reported information indicating poor health (e.g., high BMI and positive hypertension status, as in Patient A), the model confidently stops and predicts the patient as diabetic. 
In contrast, when the blood glucose level is moderate (Patient B), the model continues to acquire additional features (HbA1c level) before making a prediction, reflecting the need for further confirmation. 
For patients whose reported information indicates good health and who also exhibit normal glucose levels (Patient C), the model predicts a non-diabetic outcome without requesting further tests. This behavior demonstrates the model’s ability to adaptively halt costly testing when sufficient evidence has already been gathered.