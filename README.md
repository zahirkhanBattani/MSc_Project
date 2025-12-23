AI-Powered Project Risk Assessor

MSc Project – Information Technology with Project Management

📌 Overview

This project presents an AI-powered Project Risk Assessment system developed as part of my Master’s degree in Information Technology with Project Management.
The system uses machine learning regression techniques to predict overall project risk levels based on key project characteristics and provides an interactive Streamlit dashboard for practical decision support.

The aim of the project is to support project managers and stakeholders in identifying potential risk severity early in the project lifecycle, enabling more informed planning and mitigation strategies.

🎓 Academic Context

Degree: MSc Information Technology with Project Management

Institution: University of the West of Scotland

Project Type: Final MSc Dissertation Project

Supervisor: Dr. Jas Semrl

Status: Final report submitted and viva completed

This repository contains the final, cleaned, and reproducible version of the project codebase.

🧠 Problem Statement

Project risk assessment is often subjective and experience-driven. Traditional qualitative methods may fail to consistently capture complex relationships between project attributes such as budget, team size, complexity, and timelines.

This project investigates whether machine learning regression models can:

Learn patterns from historical project data

Predict a continuous risk score

Map predictions into interpretable risk categories (Low, Medium, High, Critical)

📊 Dataset

Source: Public Kaggle project risk dataset

Size: ~4,000 project records

Features include:

Budget

Team size

Project complexity

Duration and related project characteristics

⚠️ The dataset itself is not included in this repository due to licensing and academic integrity constraints.

⚙️ Methodology

The project followed a structured data science pipeline:

Data exploration and preprocessing

Feature engineering and transformation

Baseline model experimentation

Advanced regression modelling using XGBoost Regressor

Model evaluation using standard regression metrics

Definition of global percentile-based risk thresholds

Deployment via a Streamlit web application

🤖 Model & Evaluation

The final model selected for deployment:

Model: XGBoost Regressor

Evaluation Metrics:

R²: ~0.60

MAE: ~0.13

RMSE: ~0.16

These results demonstrate a reasonable predictive capability for a real-world, noisy project management dataset.

🚦 Risk Categorisation

Continuous risk scores are mapped into four interpretable categories using global percentile thresholds derived from the training data:

Low Risk

Medium Risk

High Risk

Critical Risk

This approach ensures:

Consistent categorisation

Independence from batch-specific score distributions

Improved interpretability for non-technical users

🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application that allows users to:

Enter project parameters

Receive a predicted risk score

View the corresponding risk category

Perform batch predictions using CSV input (if enabled)

The dashboard code is located in:

AI_Risk_Assessor/dashboard/

🗂️ Repository Structure
MSc_Project/
│
├── AI_Risk_Assessor/
│   ├── dashboard/              # Streamlit application
│   ├── notebooks/              # Data analysis & modelling notebooks
│   └── figures/                # Key result visualisations
│
├── .gitignore
├── README.md


Model artifacts, datasets, and generated outputs are intentionally excluded to ensure a clean and reproducible repository.

▶️ How to Run (Local Setup)

Clone the repository:

git clone https://github.com/zahirkhanBattani/MSc_Project.git
cd MSc_Project


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run AI_Risk_Assessor/dashboard/app.py

🔍 Ethical & Academic Considerations

The project complies with academic integrity requirements

Public datasets are used responsibly

No personal or sensitive data is included

The system is intended as a decision-support tool, not a replacement for professional judgement

📌 Disclaimer

This project was developed for academic and research purposes as part of an MSc dissertation.
While the model demonstrates promising performance, results should be interpreted within the context of the dataset and methodology used.

## 📚 Author
**Zahir Khan**  
MSc IT with Project Management Candidate
Former Technical Project Manager | AI & Data-Driven Systems
📍 London, UK  
📧 zahirkhanbettani@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/zahirkhanbettani) | 💻 [GitHub](https://github.com/zahirkhan1990)


