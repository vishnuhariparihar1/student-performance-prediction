# Student Performance Prediction & Analytics System

## Features
- 500-row illustrative synthetic dataset
- Dashboard KPIs
- Exam-mark distribution and study-hours scatter plot
- Performance-band chart
- Interactive marks prediction
- Dataset filtering and CSV download
- Summary statistics
- Linear Regression with MAE, RMSE, and R² evaluation

## Run on Windows
1. Extract this folder.
2. Open the folder in File Explorer.
3. Click the address bar, type `cmd`, press Enter.
4. Run `python -m pip install -r requirements.txt`
5. Run `python -m streamlit run app.py`
6. Open the local URL printed in the terminal (usually http://localhost:8501).

If `python` is not recognized but `py --version` works, use `py -m pip install -r requirements.txt` and `py -m streamlit run app.py`.

## Academic integrity and limitations
The 500-row dataset is synthetic and intended for demonstrating a workflow. Do not describe it as collected real student data. The model is a classroom demonstration and is not suitable for real educational decisions. Customize the report, understand the code, and follow your college's AI-use rules.
