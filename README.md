# Student Performance Prediction Using Machine Learning

A machine learning mini project that predicts students' exam marks based on study hours, attendance, previous marks, assignments completed, and sleep hours. The project uses an interactive Streamlit web application to explore student data and generate predictions.

## Project Overview

The application allows users to:
- View a dashboard with student performance statistics.
- Predict estimated exam marks using input features.
- Explore and filter the student dataset.
- View model evaluation metrics.

## Technologies Used

- **Python** — Core programming language
- **Pandas** — Data handling and analysis
- **NumPy** — Numerical operations
- **Scikit-learn** — Machine learning using Linear Regression
- **Streamlit** — Interactive web application
- **Matplotlib & Seaborn** — Data visualization

## Dataset

The project includes a dataset of 500 **synthetic student records**. It contains the following features:

| Feature | Description |
|---|---|
| StudyHours | Hours spent studying |
| Attendance | Attendance percentage |
| PreviousMarks | Marks from previous assessments |
| AssignmentsCompleted | Number of completed assignments |
| SleepHours | Average hours of sleep |
| ExamMarks | Exam marks to be predicted |

*Note: This is synthetic data created for educational demonstration, not real student records.*

## Machine Learning Model

The project uses **Multiple Linear Regression** to estimate exam marks from five input features.

- Training/testing split: 80% / 20%
- Training records: 400
- Testing records: 100
- Evaluation metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score

The reported evaluation results are based on the included synthetic dataset and should not be interpreted as real-world student prediction accuracy.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vishnuhariparihar1/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the application

```bash
python -m streamlit run app.py
```

The app will open in your browser. If it does not open automatically, visit `http://localhost:8501`.

## Project Files

- `app.py` — Streamlit application and machine learning workflow
- `student_data.csv` — Synthetic dataset
- `requirements.txt` — Python dependencies
- `project_report.docx` — Project report
- `presentation.pptx` — Project presentation

## Future Improvements

- Train and evaluate the model on a real, appropriately anonymized dataset.
- Compare Linear Regression with other regression algorithms.
- Improve prediction performance through feature engineering and model tuning.

## Author

**Vishnu Hari Parihar**

## License

This project is intended for educational purposes.
