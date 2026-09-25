# Tamil Nadu Government Citizen Chatbot

## 📌 Project OvervieW

The Tamil Nadu Government Citizen Chatbot is an ML-enabled RAG-based application designed to help citizens find information about Tamil Nadu government services.

The system uses Natural Language Processing, Machine Learning, TF-IDF based retrieval, and a Streamlit dashboard to classify citizen queries and retrieve relevant government service information.

## 🎯 Objectives

- Classify citizen queries using Machine Learning
- Retrieve relevant government service information
- Support different types of citizen queries
- Provide government service information through a chatbot
- Analyze chatbot queries using an interactive dashboard
- Compare ML model performance and hyperparameter tuning results

## 🔄 Project Workflow

User Query  
↓  
Text Preprocessing  
↓  
TF-IDF Vectorization  
↓  
ML Intent Classification  
↓  
RAG-based Information Retrieval  
↓  
Relevant Government Information  
↓  
Chatbot Response

## 🧠 Machine Learning

The project evaluates the following classification models:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score

### Hyperparameter Tuning

SVM:
- C values are evaluated to identify suitable model performance.

Random Forest:
- Different numbers of estimators are evaluated.

## 📚 RAG System

The Retrieval-Augmented Generation component uses the collected Tamil Nadu government service dataset as its knowledge source.

The system retrieves relevant service records based on the citizen's query and uses the retrieved information to provide the response.

## 📊 Streamlit Dashboard

The Streamlit dashboard provides:

- Dataset Overview
- Intent Distribution
- Language Distribution
- Department Distribution
- Service Distribution
- ML Model Performance
- SVM Hyperparameter Tuning
- Random Forest Hyperparameter Tuning

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Machine Learning
- RAG
- Streamlit
## 📂 Project Structure

```text
Tamil-Nadu-Government-Citizen-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
│
├── Tamil_Nadu_Combined_Dataset (3).csv
├── model_performance.csv
├── svm_tuning_results.csv
└── rf_tuning_results.csv
pip install -r requirements.txt
streamlit run app.py

Then:

```text
## 🚀 Live Demo

[Open Tamil Nadu Government Citizen Chatbot](https://tamil-nadu-government-citizen-chatbot-je8ut6wjfoo83ryt5ka4gr.streamlit.app)


## 📌 Key Features

- Government service information
- Intent-based query classification
- Document and eligibility assistance
- ML model comparison
- Hyperparameter tuning
- RAG-based information retrieval
- Interactive Streamlit dashboard
- Tamil, Tanglish and English query support

## 👩‍💻 Author

**R. Rathipriya**

M.Sc Data Analytics  
Bharathiar University
