import streamlit as st  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Analytics", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction & Analytics System")
st.write("A machine-learning mini-project to explore sample student data, understand performance patterns, and estimate exam marks.")

@st.cache_data
def load_sample_data():
    return pd.read_csv("student_data.csv")

@st.cache_resource
def build_model(data):
    features = ["StudyHours", "Attendance", "PreviousMarks", "AssignmentsCompleted", "SleepHours"]
    X = data[features]
    y = data["ExamMarks"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "r2": r2_score(y_test, pred),
        "features": features
    }
    return model, metrics, X_test, y_test, pred

try:
    data = load_sample_data()
except Exception:
    st.error("Keep student_data.csv in the same folder as app.py.")
    st.stop()

# Add a derived category for analysis
data["PerformanceBand"] = pd.cut(data["ExamMarks"], bins=[-1, 39, 59, 79, 100],
                                 labels=["Needs Support", "Average", "Good", "Excellent"])
model, metrics, X_test, y_test, pred = build_model(data)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predict Marks", "🗂️ Explore Data", "ℹ️ About the Model"])

with tab1:
    st.subheader("Student Performance Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in dataset", f"{len(data):,}")
    c2.metric("Average exam marks", f"{data['ExamMarks'].mean():.1f}/100")
    c3.metric("Average attendance", f"{data['Attendance'].mean():.1f}%")
    c4.metric("Average study hours", f"{data['StudyHours'].mean():.1f} hrs/day")

    left, right = st.columns(2)
    with left:
        st.markdown("**Exam marks distribution**")
        fig, ax = plt.subplots()
        ax.hist(data["ExamMarks"], bins=15, edgecolor="black")
        ax.set_xlabel("Exam marks")
        ax.set_ylabel("Number of students")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Study hours vs exam marks**")
        fig, ax = plt.subplots()
        ax.scatter(data["StudyHours"], data["ExamMarks"], alpha=0.65)
        ax.set_xlabel("Study hours per day")
        ax.set_ylabel("Exam marks")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Performance categories**")
    band_counts = data["PerformanceBand"].value_counts().reindex(
        ["Needs Support", "Average", "Good", "Excellent"], fill_value=0)
    fig, ax = plt.subplots()
    band_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Performance category")
    ax.set_ylabel("Number of students")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Estimate a student's exam marks")
    st.caption("Adjust the inputs and select Predict. The output is an estimate, not a guaranteed result.")
    left, right = st.columns(2)
    with left:
        study_hours = st.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        previous_marks = st.slider("Previous marks (%)", 0, 100, 65)
    with right:
        assignments = st.slider("Assignments completed (out of 10)", 0, 10, 7)
        sleep_hours = st.slider("Sleep hours per night", 3.0, 12.0, 7.0, 0.5)
        st.info("Class participation is included in the dataset exploration. The prediction uses five numeric features for a simple, explainable model.")
    if st.button("Predict exam marks", type="primary"):
        new = pd.DataFrame([{
            "StudyHours": study_hours, "Attendance": attendance,
            "PreviousMarks": previous_marks, "AssignmentsCompleted": assignments,
            "SleepHours": sleep_hours
        }])
        score = float(model.predict(new)[0])
        score = max(0, min(100, score))
        st.metric("Estimated exam marks", f"{score:.1f} / 100")
        if score >= 80:
            st.success("Estimated band: Excellent")
        elif score >= 60:
            st.success("Estimated band: Good")
        elif score >= 40:
            st.warning("Estimated band: Average")
        else:
            st.error("Estimated band: Needs Support")
        st.caption("Educational demonstration only. Do not use this estimate to make real decisions about students.")

with tab3:
    st.subheader("Explore the sample dataset")
    col1, col2 = st.columns(2)
    with col1:
        band_filter = st.multiselect("Filter by performance category",
            options=["Needs Support", "Average", "Good", "Excellent"],
            default=["Needs Support", "Average", "Good", "Excellent"])
    with col2:
        min_attendance = st.slider("Minimum attendance (%)", 0, 100, 0)
    filtered = data[data["PerformanceBand"].astype(str).isin(band_filter) & (data["Attendance"] >= min_attendance)]
    st.write(f"Showing **{len(filtered)}** of **{len(data)}** records.")
    st.dataframe(filtered, use_container_width=True, height=350)
    st.download_button("Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_filtered.csv", mime="text/csv")
    st.markdown("**Summary statistics**")
    st.dataframe(filtered[["StudyHours", "Attendance", "PreviousMarks", "AssignmentsCompleted", "SleepHours", "ExamMarks"]].describe().round(2),
                 use_container_width=True)

with tab4:
    st.subheader("Machine learning method")
    st.write("Algorithm: Multiple Linear Regression. It estimates a numeric outcome (ExamMarks) from five input features.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Absolute Error", f"{metrics['mae']:.2f} marks")
    c2.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f} marks")
    c3.metric("R² score", f"{metrics['r2']:.2f}")
    st.markdown("**Workflow**")
    st.markdown("""
    1. Load the CSV dataset with Pandas.
    2. Select five input features and the target column.
    3. Split records into training and test sets (80% / 20%).
    4. Train the Linear Regression model on training records.
    5. Evaluate on held-out test records using MAE, RMSE and R².
    6. Use the trained model to estimate marks for new inputs.
    """)
    st.warning("Dataset note: all 500 records are synthetic sample data generated for this college demonstration, not real student records. Metrics demonstrate evaluation methods and are not evidence of real-world predictive accuracy.")
    st.markdown("**Feature list:** " + ", ".join(metrics["features"]))
