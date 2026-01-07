📊 Problem Difficulty Prediction using NLP & Machine Learning

This project aims to predict the difficulty of programming problems based solely on their textual descriptions.
Given a problem statement, input format, and output format, the system predicts:

Difficulty class: easy, medium, or hard

Difficulty score: a continuous numerical estimate

The project follows a classical, interpretable NLP + ML pipeline, emphasizing clarity, reproducibility, and explainability.

🔍 Problem Motivation

Online judges and competitive programming platforms often label problems with difficulty levels.
However, these labels are:

subjective,

coarse-grained,

and sometimes inconsistent.

This project explores whether problem difficulty can be learned automatically from text, using:

problem descriptions,

input/output specifications,

and lightweight domain-informed features.

🧠 High-Level Workflow
Raw Data (JSONL)
        ↓
Data Loading
        ↓
Preprocessing & Feature Engineering
        ↓
TF-IDF + Numeric Feature Construction
        ↓
Model Training
  ├─ Classification (difficulty class)
  └─ Regression (difficulty score)
        ↓
Evaluation & Explainability
        ↓
Inference & Web Application

📁 Project Structure
ACM/
├── app.py                  # Streamlit web application
├── data/
│   ├── raw/                # Raw JSONL dataset
│   └── processed/          # Processed CSV dataset
├── models/                 # Saved models and transformers
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train_classifier_logreg.py
│   ├── train_classifier_svm.py
│   ├── train_regressor_ridge.py
│   ├── train_regressor_svr.py
│   ├── evaluate.py
│   ├── predict.py
│   └── main.py
└── README.md

⚙️ Detailed Workflow & Code Explanation
1️⃣ config.py — Central Configuration

Defines:

directory structure,

dataset paths,

target column names,

train–test split,

random seed for reproducibility.

This ensures consistency across all scripts and avoids hard-coded values.

2️⃣ load_data.py — Data Ingestion

Loads the raw dataset stored in JSON Lines (.jsonl) format and converts it into a pandas DataFrame.

A small sanity-check block allows quick verification of:

dataset shape,

available columns.

3️⃣ preprocess.py — Data Cleaning & Feature Engineering

Responsible for transforming raw text into structured features.

Steps performed:

Selects required text and target columns

Removes missing or empty text rows

Engineers numeric features:

text length features

keyword frequency (domain-informed)

Combines all text into a single full_text field

Output:

Processed dataset saved as CSV

Result:

Final dataset shape: (3899, 10)

4️⃣ features.py — Feature Construction

Builds the final feature matrix used for modeling.

Features used:

TF-IDF (1–2 grams) from full_text

Numeric features:

description length

input length

output length

keyword count

Both feature types are scaled and combined.

Train–test split:

80% train (3119 samples)

20% test (780 samples)

Total features: 310,762

Saved artifacts:

TF-IDF vectorizer

Numeric feature scaler

5️⃣ Model Training Scripts
🔹 Classification

train_classifier_logreg.py
Logistic Regression (final classifier)

train_classifier_svm.py
Linear SVM (comparison baseline)

🔹 Regression

train_regressor_ridge.py
Ridge Regression (baseline)

train_regressor_svr.py
Linear SVR (final regressor)

Each script:

trains the model,

evaluates on the test set,

saves the trained model to models/.

6️⃣ Model Performance Summary
🧠 Classification (Difficulty Class)
Model	Accuracy	F1-macro
Logistic Regression	0.5013	0.4894
Linear SVM	0.4859	0.4455

✔ Logistic Regression selected as final classifier.

📈 Regression (Difficulty Score)
Model	MAE	RMSE
Ridge Regression	1.7237	2.0457
Linear SVR	1.7198	2.0449

✔ Linear SVR selected as final regressor.

7️⃣ evaluate.py — Evaluation & Explainability

Performs:

classification evaluation (accuracy, F1, confusion matrix),

regression evaluation (MAE, RMSE),

regression → class calibration check,

explainability analysis by extracting top TF-IDF words per class.

This adds transparency to model behavior.

8️⃣ predict.py — Inference Module

Provides a single user-facing prediction function:

predict_difficulty(description, input_desc, output_desc)


Returns:

difficulty class,

difficulty score.

Ensures:

feature consistency,

correct use of saved models and transformers.

9️⃣ main.py — Pipeline Orchestration

Runs the entire pipeline end-to-end:

Preprocessing

Feature extraction

Classifier training

Regressor training

Final evaluation

Stops execution immediately if any step fails.

🔟 app.py — Streamlit Web Application

Provides a simple web interface where users can:

enter a problem description,

get predicted difficulty and score instantly.

Built on top of the trained ML pipeline using Streamlit.

🚀 How to Run
Run full pipeline
python src/main.py

Run Streamlit app
streamlit run app.py

🧾 Key Design Choices

Classical ML over DL for:

interpretability,

limited dataset size,

faster experimentation.

Multi-task learning:

classification + regression.

Explainability prioritized over complexity.

📌 Conclusion

This project demonstrates that problem difficulty can be reasonably estimated from textual descriptions alone using a well-designed classical ML pipeline.
The system is modular, interpretable, reproducible, and easily extensible for future deep learning approaches.
