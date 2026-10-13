# Student Academic Risk Prediction Using AdaBoost

A machine-learning project that predicts whether a student is **At Risk** or **On Track** using academic and student-life indicators. The project demonstrates data generation, exploratory data analysis (EDA), feature association, model training, and evaluation with an AdaBoost classifier.

> **Important:** The current script creates a synthetic dataset of 300 students. Its labels are generated from a hand-crafted risk score, so the model is a learning/demo project—not a validated tool for making real academic decisions.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Workflow](#workflow)
- [Machine Learning Model](#machine-learning-model)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [How to Run](#how-to-run)
- [Output](#output)
- [Limitations and Future Scope](#limitations-and-future-scope)

## Project Overview

Student performance can be affected by factors such as attendance, study time, internal assessment scores, assignment performance, previous CGPA, lab completion, sleep, commute time, and extracurricular commitments.

This project builds a binary classification workflow using **AdaBoost (Adaptive Boosting)** to classify a student as:

- **At Risk** — the synthetic risk score crosses the project's threshold.
- **On Track** — the synthetic risk score does not cross the threshold.

The Python script generates the dataset, explores and cleans it, visualizes patterns, trains the classifier, evaluates predictions, displays feature importance, and predicts the status of one sample student.

## Objectives

- Generate a reproducible synthetic student dataset.
- Explore student academic and lifestyle attributes.
- Clean the data and inspect missing and duplicate values.
- Visualize risk-status distributions and relationships between features.
- Examine feature association using mutual information.
- Train an AdaBoost classification model.
- Evaluate the model using accuracy, a classification report, and a confusion matrix.
- Review feature importance and demonstrate a single-student prediction.

## Dataset

The script creates **300 synthetic student records** and saves them to `Data.csv` when run.

| Column | Description |
|---|---|
| `Student_ID` | Synthetic student identifier |
| `Attendance_Percent` | Attendance percentage |
| `Study_Hours_Per_Week` | Study hours per week |
| `Internal_Score` | Internal assessment score |
| `Assignment_Score` | Assignment score |
| `Sleep_Hours` | Hours of sleep |
| `Previous_CGPA` | Previous CGPA |
| `Lab_Completion_Percent` | Lab completion percentage |
| `Commute_Minutes` | Commute time in minutes |
| `Club_Hours_Per_Week` | Club or extracurricular hours per week |
| `Risk_Status` | Target label: `At Risk` or `On Track` |

### How the target label is created

The script calculates a hand-crafted risk score using selected academic and lifestyle indicators, including lower attendance, fewer study hours, lower assessment scores, previous CGPA, longer commutes, and some threshold-based combinations. A small random component is also included. Scores above the script's threshold are labelled `At Risk`; the rest are labelled `On Track`.

Because the target is constructed from these rules, the model learns patterns in the synthetic data rather than proving how students will perform in a real institution.

## Technologies Used

- **Python** — main programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations and synthetic data generation
- **Matplotlib** — plots and charts
- **Seaborn** — statistical visualizations
- **Scikit-learn** — train/test split, AdaBoost, decision trees, mutual information, and evaluation metrics

## Workflow

1. **Generate data:** create synthetic student attributes and risk labels.
2. **Load and understand data:** inspect sample rows, column types, summary statistics, class distribution, missing values, and duplicates.
3. **Clean data:** trim column and label whitespace, remove duplicate rows, and drop rows with missing values.
4. **Perform EDA:** plot risk distribution, attendance and study-hour distributions, academic-score comparisons, a scatter plot, and a correlation heatmap.
5. **Analyze feature association:** calculate mutual information between the numeric input features and the target label.
6. **Prepare features:** exclude `Student_ID` and `Risk_Status` from the model inputs.
7. **Split the data:** use an 80% training / 20% testing split with stratification.
8. **Train AdaBoost:** use decision stumps (decision trees with `max_depth=1`) as weak learners.
9. **Evaluate:** calculate accuracy, classification report, and confusion matrix.
10. **Interpret and predict:** plot feature importance and predict a test student's risk status and class probabilities.

## Machine Learning Model

The script uses `AdaBoostClassifier` with the following configuration:

| Parameter | Value |
|---|---|
| Base estimator | `DecisionTreeClassifier(max_depth=1)` |
| Number of estimators | `50` |
| Learning rate | `0.8` |
| Random state | `42` |
| Test set size | `20%` |

AdaBoost combines multiple weak learners into a stronger ensemble. In this project, each weak learner is a shallow decision tree, also called a decision stump.

## Evaluation

The script reports:

- **Accuracy** — overall fraction of test examples classified correctly.
- **Classification report** — precision, recall, F1-score, and support for `On Track` and `At Risk`.
- **Confusion matrix** — compares actual and predicted classes.
- **Feature importance** — shows each selected feature's contribution according to the trained AdaBoost model.
- **Prediction example** — displays a test student's feature values, predicted status, predicted class probabilities, and actual status.

The exact metric values are produced when you run the script. They are intentionally not stated here because they depend on execution and should be copied from actual output rather than guessed.

## Project Structure

A simple GitHub repository layout is:

```text
Student-Academic-Risk-Prediction-AdaBoost/
├── Student_Academic_Risk_AdaBoost_PBL3.py
├── README.md
├── requirements.txt
└── .gitignore
```

`Data.csv` is generated by the script. You may commit it if you want to share the generated dataset, but it is not required to run the project because the script creates it first.

## Installation and Setup

### 1. Install Python

Install Python 3.10 or later, then check the installation:

```bash
python --version
```

### 2. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR-USERNAME/Student-Academic-Risk-Prediction-AdaBoost.git
cd Student-Academic-Risk-Prediction-AdaBoost
```

### 3. Create and activate a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

Create a `requirements.txt` file with:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Then install them:

```bash
python -m pip install -r requirements.txt
```

## How to Run

Place the Python script in the repository folder and run:

```bash
python Student_Academic_Risk_AdaBoost_PBL3.py
```

The script creates `Data.csv`, loads it, performs analysis, trains the model, prints evaluation results, and opens charts. Depending on your environment, plots may appear in separate windows.

## Output

When the script runs, you should see:

- Confirmation that libraries imported successfully.
- A message confirming that `Data.csv` was saved and loaded.
- Dataset information and data-cleaning checks.
- Exploratory data analysis charts.
- Statistical summaries and mutual-information scores.
- AdaBoost training confirmation.
- Accuracy, classification report, and confusion matrix.
- Feature-importance values and a feature-importance chart.
- A sample student's predicted status, class probabilities, and actual status.

For a more complete project page, add screenshots of selected charts and the terminal evaluation output to a `screenshots/` folder and link them here.

## Limitations and Future Scope

### Current limitations

- The dataset is synthetic and generated by rules in the script, not collected from a real institution.
- The target label is derived from a hand-crafted risk score, so evaluation measures how well the model learns that synthetic labeling process.
- The current script demonstrates prediction on a test-set row; it does not provide a web app, dashboard, or live data integration.
- Predictions should not be used to make consequential decisions about real students.

### Future improvements

- Evaluate on an appropriately consented, anonymized real-world dataset.
- Use cross-validation and compare AdaBoost with other suitable models.
- Add calibration and analyze precision/recall carefully, especially for the `At Risk` class.
- Save the trained model and add a reusable prediction function.
- Build a simple interface for entering student indicators.
- Add model monitoring, fairness checks, privacy safeguards, and clear guidance for human review.

## Author

**Dhairya Tailor**  
Student Academic Risk Prediction using AdaBoost — PBL Project

---

If you use or extend this project, please describe the dataset and evaluation honestly, especially the fact that the current data is synthetic.
