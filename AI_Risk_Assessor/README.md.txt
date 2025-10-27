# 🧠 AI-Based Project Management Risk Assessor

This project predicts project risk levels using machine learning techniques.

## 📋 Overview
The system uses historical project data to train models that assess the likelihood of project risks.  
It includes preprocessing, model training, hyperparameter tuning, and evaluation steps.

## ⚙️ Tech Stack
- Python (pandas, NumPy, scikit-learn, matplotlib)
- Jupyter Notebooks
- Random Forest, Decision Tree, Logistic Regression
- Data preprocessing pipelines and GridSearchCV optimization

## 🧾 Project Structure
AI_Risk_Assessor/
├── data/ # Datasets used for training and testing
├── notebooks/ # Jupyter notebooks for preprocessing & model training
├── models/ # Saved ML models
├── reports/ # Interim report and project documentation
├── check_env.py # Environment check script
└── README.md # Project documentation


---

## 🎯 Current Results
- **Best Model:** Random Forest  
- **Accuracy:** ~0.51  
- **F1 Score:** ~0.50  
- **Hyperparameter Search:** 216 candidates × 5-fold cross-validation (1080 fits total)  
- **Pipeline:** Combined numerical + categorical preprocessing (scaling, encoding, imputation)

---

## 🚀 Next Steps
- Extend dataset for richer feature extraction  
- Add ensemble and deep-learning models (TensorFlow / PyTorch)  
- Develop a user interface for real-time risk prediction  
- Integrate model into a simple Flask or Streamlit app  

---

## 📚 Author
**Zahir Khan**  
MSc IT with Project Management Candidate | University of the West of Scotland  
📍 London, UK  
📧 zahirkhanbettani@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/zahirkhanbettani) | 💻 [GitHub](https://github.com/zahirkhan1990)


