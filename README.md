# 📊 Problem Difficulty Prediction using NLP & Machine Learning

This project predicts the difficulty of programming problems using only their textual descriptions.

Given a problem statement, input format, and output format, the system outputs:

- **Difficulty class:** `Easy`, `Medium`, or `Hard`
- **Difficulty score:** a continuous numerical value

The project follows a **classical, interpretable NLP + Machine Learning pipeline**, focusing on clarity, reproducibility, and explainability rather than unnecessary complexity.

---

## 🎯 Problem Statement

Online coding platforms assign difficulty labels to problems, but these labels are often:

- subjective,
- coarse-grained,
- inconsistent across platforms.

This project investigates whether **problem difficulty can be learned automatically from text**, using:

- problem descriptions,
- input/output specifications,
- lightweight, domain-informed features.

---

### 📌 Project Summary (Technical Description) (**IMPORTANT**)

This project predicts the difficulty of programming problems using only their textual information. The dataset consists of problem descriptions along with their input and output specifications. During preprocessing, only the relevant textual columns (`description`, `input_description`, `output_description`) and target variables (`problem_class`, `problem_score`) were retained. Rows with missing or empty text were removed to ensure data quality.

All text fields were combined into a single unified representation (`full_text`) to capture the complete semantic context of each problem. Along with text, **domain-informed numeric features** were engineered, including:

- length of the problem description,  
- length of input description,  
- length of output description,  
- frequency of algorithm-related keywords (e.g., graph, dp, recursion, bfs, dfs).

Text was represented using **TF-IDF with unigrams and bigrams**, producing a high-dimensional sparse feature space. These features were combined with scaled numeric features to form the final input matrix. The dataset was split into **80% training and 20% testing**, with stratification applied on the difficulty class.

Two learning tasks were modeled:
- **Classification** to predict the difficulty class (Easy, Medium, Hard)  
- **Regression** to predict a continuous difficulty score  

For classification, **Logistic Regression** was selected based on **macro F1-score** rather than accuracy due to class imbalance and significant overlap involving the *Medium* class. While the *Hard* class was learned reliably, ambiguity in the *Medium* class limited overall accuracy.

For regression, **Linear Support Vector Regression (SVR)** achieved slightly lower MAE and RMSE compared to Ridge Regression. Regression outputs were also mapped back to difficulty classes as a consistency check.

Overall, the project highlights the challenges of predicting problem difficulty from text, particularly due to subjective labels, class imbalance, and semantic overlap. The pipeline emphasizes **interpretability, reproducibility, and explainability**, and includes an end-to-end workflow with a Streamlit web application for real-time inference on unseen problems.

---

## 🔄 High-Level Workflow

Raw Data (JSONL)  
↓  
Data Loading  
↓  
Preprocessing & Feature Engineering  
↓  
TF-IDF + Numeric Feature Construction  
↓  
Model Training  
↓  
Evaluation & Explainability  
↓  
Inference & Web Application

## 📁 Project Structure

```
ACM/
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
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
```

## ⚙️ Detailed Workflow & Code Explanation

steps to run the code :-
config.py -> load_data.py -> preprocess.py -> festures.py -> train_classifier_logreg.py -> train_regressor_svr.py -> evaluate.py -> predict.py

or just run **'src/main.py'** 

### 1️⃣ `config.py` — Central Configuration

Defines:

- base directory paths,
- dataset locations,
- target column names,
- train–test split ratio,
- random seed for reproducibility.

This ensures **consistent configuration across all scripts**.

---

### 2️⃣ `load_data.py` — Data Loading

- Loads the raw dataset from **JSONL format**
- Converts it into a pandas **DataFrame**
- Includes a sanity check to verify:
  - dataset shape,
  - available columns

---

### 3️⃣ `preprocess.py` — Preprocessing & Feature Engineering

**Responsibilities:**

- Selects required text and target columns
- Removes missing or empty text rows
- Engineers numeric features:
  - description length
  - input description length
  - output description length
  - keyword frequency
- Combines all text fields into a single `full_text` column

**Output:** Final dataset shape: (3899, 10)

Columns:
 - description
 - input_description
 - output_description
 - problem_class
 - problem_score
 - len_description
 - len_input_description
 - len_output_description
 - full_text
 - keyword_count

---

### 4️⃣ `features.py` — Feature Construction

Builds the final feature matrix used for modeling.

**Features used:**

- **TF-IDF (1–2 grams)** from `full_text`
- **Numeric features:**
  - description length
  - input length
  - output length
  - keyword count

Both feature types are **scaled and combined**.

**Train–test split:**

- **80% training data** → 3119 samples
- **20% test data** → 780 samples

**Total features:** `310,762`

---

### 5️⃣ Model Training

#### 🔹 Classification Models

- **Logistic Regression** (final classifier)
- **Linear SVM** (comparison baseline)

#### 🔹 Regression Models

- **Ridge Regression** (baseline)
- **Linear SVR** (final regressor)

Each training script:

- trains the model,
- evaluates on the test set,
- saves the trained model to `models/`.

---

## 📊 Model Performance

### 🎯 Classification Results (Difficulty Class)

| Model               | Accuracy | F1-macro |
|---------------------|----------|----------|
| Logistic Regression | **0.5013** | **0.4894** |
| Linear SVM          | 0.4859   | 0.4455   |

✅ **Logistic Regression** selected as the final classifier.

---

### 📈 Regression Results (Difficulty Score)

| Model            | MAE     | RMSE   |
|------------------|---------|--------|
| Ridge Regression | 1.7237  | 2.0457 |
| Linear SVR       | **1.7198** | **2.0449** |

✅ **Linear SVR** selected as the final regressor.

---

### 🔍 `evaluate.py` — Evaluation & Explainability

Performs:

- classification evaluation (accuracy, F1-score, confusion matrix),
- regression evaluation (MAE, RMSE),
- regression → class calibration check,
- model explainability by identifying top TF-IDF words per class.

This improves transparency and interpretability of predictions.

---

### 🔮 `predict.py` — Inference Module

Provides a user-facing prediction function:

```python
predict_difficulty(description, input_desc, output_desc)
```

**Returns:**

- difficulty class,
- difficulty score.

**Ensures:**

- feature consistency,
- correct use of saved models and transformers.

---

### 9️⃣ `main.py` — Pipeline Orchestration

Runs the entire pipeline end-to-end:

1. Preprocessing  
2. Feature extraction  
3. Classifier training  
4. Regressor training  
5. Final evaluation  

Stops execution immediately if any step fails.

---

### 🔟 `app.py` — Streamlit Web Application

Provides a simple web interface where users can:

- enter a problem description,
- get predicted difficulty and score instantly.

Built on top of the trained ML pipeline using **Streamlit**.

---

## 🚀 How to Run

### Run full pipeline

run the src/main.py file for the code implementation
run app.py for streamlit demonstration

```bash
python src/main.py

streamlit run app.py
```

---

## 🧾 Key Design Choices

- **Classical ML chosen over deep learning** for:
  - interpretability,
  - limited dataset size,
  - faster experimentation.

- **Multi-task learning:**
  - classification + regression.

- **Explainability prioritized** alongside performance.

---

## 📌 Conclusion

This project demonstrates that **problem difficulty can be reasonably estimated from textual descriptions alone** using a well-structured classical ML pipeline.

The system is **modular, reproducible, interpretable**, and easily extensible for future deep learning approaches.

---

### link to report and video 

1. report -> https://docs.google.com/document/d/1F1t89ebRo9VksvIX6joR2EtlODbFgNlk/edit?usp=sharing&ouid=108800960855133179540&rtpof=true&sd=true
2. demo video -> https://drive.google.com/file/d/1ZDN0kpOkKAyNR6UGY2ghbHxK4JAJzmdF/view?usp=sharing






























































