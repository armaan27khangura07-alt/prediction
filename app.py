import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_PATH = "/content/merged_dataset.csv"
MODEL_PATH = "my_model.pkl"
TARGET = "FinalGrade"

FEATURES = [
    "StudyHours",
    "Attendance",
    "Resources",
    "Extracurricular",
    "Motivation",
    "Internet",
    "Gender",
    "Age",
    "LearningStyle",
    "OnlineCourses",
    "Discussions",
    "AssignmentCompletion",
    "ExamScore",
    "EduTech",
    "StressLevel"
]

# =====================================================
# 2. LOAD DATASET
# =====================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "merged_dataset.csv must be in the same folder as app.py"
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

missing = [
    col for col in FEATURES + [TARGET]
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

if df.empty:
    raise ValueError("No valid target values found.")

GRADE_CLASSES = sorted(df[TARGET].unique().tolist())

def class_name(value):
    return f"Class {value}"

# =====================================================
# 3. TRAIN MACHINE LEARNING MODEL
# =====================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

if y.nunique() < 2:
    raise ValueError("FinalGrade must have at least two classes.")

stratify_values = (
    y if y.value_counts().min() >= 2 else None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=stratify_values
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

joblib.dump(model, MODEL_PATH)

# =====================================================
# 4. DASHBOARD METRICS
# =====================================================

total_students = len(df)
total_features = len(FEATURES)
total_classes = df[TARGET].nunique()
average_score = df["ExamScore"].mean()
average_attendance = df["Attendance"].mean()

# =====================================================
# 5. HOME DASHBOARD
# =====================================================

def home_dashboard():
    grade_counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
        .rename_axis("FinalGrade")
        .reset_index(name="Students")
    )
    grade_counts["Class"] = grade_counts[
        "FinalGrade"
    ].apply(class_name)

    fig = px.bar(
        grade_counts,
        x="Class",
        y="Students",
        color="Class",
        text="Students",
        title="Student Grade Distribution"
    )
    fig.update_layout(
        template="plotly_dark",
        height=400
    )

    metrics = f"""
    <div class="metrics">
        <div class="metric">
            <h3>Total Students</h3>
            <h1>{total_students:,}</h1>
        </div>
        <div class="metric">
            <h3>Features</h3>
            <h1>{total_features}</h1>
        </div>
        <div class="metric">
            <h3>Grade Classes</h3>
            <h1>{total_classes}</h1>
        </div>
        <div class="metric">
            <h3>Model Accuracy</h3>
            <h1>{accuracy:.1%}</h1>
        </div>
    </div>
    """
    return metrics, fig, df.head(10)

# =====================================================
# 6. DATA ANALYSIS
# =====================================================

def analyze_data(age_range, attendance_range, study_range, selected_classes):
    filtered = df[
        df["Age"].between(age_range[0], age_range[1])
        & df["Attendance"].between(attendance_range[0], attendance_range[1])
        & df["StudyHours"].between(study_range[0], study_range[1])
    ].copy()

    if selected_classes:
        filtered = filtered[
            filtered[TARGET].isin(selected_classes)
        ]

    if filtered.empty:
        return (
            "<h3>No records found for these filters.</h3>",
            None, None, None, None, None, None
        )

    metrics = f"""
    <div class="metrics">
        <div class="metric">
            <h3>Students</h3>
            <h1>{len(filtered):,}</h1>
        </div>
        <div class="metric">
            <h3>Average Exam Score</h3>
            <h1>{filtered["ExamScore"].mean():.2f}</h1>
        </div>
        <div class="metric">
            <h3>Average Attendance</h3>
            <h1>{filtered["Attendance"].mean():.2f}</h1>
        </div>
        <div class="metric">
            <h3>Average Study Hours</h3>
            <h1>{filtered["StudyHours"].mean():.2f}</h1>
        </div>
    </div>
    """

    counts = (
        filtered[TARGET]
        .value_counts()
        .sort_index()
        .rename_axis("FinalGrade")
        .reset_index(name="Students")
    )
    counts["Class"] = counts["FinalGrade"].apply(class_name)

    fig1 = px.bar(counts, x="Class", y="Students", color="Class", title="FinalGrade Distribution")
    fig1.update_layout(template="plotly_dark")

    fig2 = px.histogram(filtered, x="ExamScore", nbins=20, title="Exam Score Distribution")
    fig2.update_layout(template="plotly_dark")

    fig3 = px.scatter(filtered, x="StudyHours", y="ExamScore", color=TARGET, title="Study Hours vs Exam Score")
    fig3.update_layout(template="plotly_dark")

    fig4 = px.scatter(filtered, x="Attendance", y="ExamScore", color=TARGET, title="Attendance vs Exam Score")
    fig4.update_layout(template="plotly_dark")

    fig5 = px.imshow(
        filtered[FEATURES + [TARGET]].corr(),
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    fig5.update_layout(template="plotly_dark", height=650)

    return metrics, fig1, fig2, fig3, fig4, fig5, filtered.head(100)

# =====================================================
# 7. STUDENT PREDICTION
# =====================================================

def predict_student(*values):
    input_data = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    classes = model.named_steps["classifier"].classes_

    probability_df = pd.DataFrame({
        "Class": [class_name(int(c)) for c in classes],
        "Probability": probabilities
    })

    fig = px.bar(
        probability_df,
        x="Class",
        y="Probability",
        text=probability_df["Probability"].apply(lambda x: f"{x:.1%}"),
        title="Prediction Probability",
        color="Probability",
        color_continuous_scale="Blues"
    )
    fig.update_layout(template="plotly_dark", yaxis=dict(range=[0, 1]))

    result = f"""
    <div class="prediction">
        <h2>Prediction Result</h2>
        <h1>{class_name(prediction)}</h1>
        <p>Predicted FinalGrade class</p>
        <p>This is a model estimate, not a guaranteed result.</p>
    </div>
    """
    return result, fig

# =====================================================
# 8. CLASSIFICATION REPORT
# =====================================================

def get_report():
    report_df = pd.DataFrame(report).transpose()
    return report_df.round(3)

# =====================================================
# 9. CUSTOM CSS
# =====================================================

css = """
.gradio-container {
    background: linear-gradient(135deg, #071426, #102747, #071426) !important;
    color: white !important;
}
#main-title {
    text-align: center;
    background: linear-gradient(90deg, #164B83, #102442);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
}
#main-title h1 { color: white; font-size: 32px; }
#main-title p { color: #D7E7F8; }
.metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 15px 0;
}
.metric {
    background: #163454;
    border: 1px solid #34577B;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
}
.metric h3 { font-size: 14px; color: #C5D9EF; }
.metric h1 { color: white; font-size: 27px; }
.prediction {
    background: #164B83;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}
.prediction h1 { font-size: 42px; color: white; }
@media(max-width: 700px) {
    .metrics { grid-template-columns: repeat(2, 1fr); }
}
"""

# =====================================================
# 10. BUILD GRADIO INTERFACE
# =====================================================

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan", neutral_hue="slate"),
    css=css,
    title="Student Performance AI Dashboard"
) as demo:

    gr.HTML("""
    <div id="main-title">
        <h1>᎐ STUDENT PERFORMANCE AI DASHBOARD</h1>
        <p>Intelligent Academic Analytics and Machine Learning Classification System</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("᎑ Home"):
            gr.Markdown("## Dashboard Overview")
            home_metrics = gr.HTML()
            home_chart = gr.Plot()
            gr.Markdown("## Dataset Preview")
            home_table = gr.Dataframe(interactive=False)
            demo.load(fn=home_dashboard, outputs=[home_metrics, home_chart, home_table])

        with gr.Tab("ᎉ Data Analysis"):
            gr.Markdown("## Explore Student Data")
            with gr.Row():
                age_slider = gr.Slider(minimum=float(df["Age"].min()), maximum=float(df["Age"].max()), value=[float(df["Age"].min()), float(df["Age"].max())], label="Age Range")
                attendance_slider = gr.Slider(minimum=float(df["Attendance"].min()), maximum=float(df["Attendance"].max()), value=[float(df["Attendance"].min()), float(df["Attendance"].max())], label="Attendance Range")
                study_slider = gr.Slider(minimum=float(df["StudyHours"].min()), maximum=float(df["StudyHours"].max()), value=[float(df["StudyHours"].min()), float(df["StudyHours"].max())], label="Study Hours Range")

            grade_filter = gr.CheckboxGroup(choices=GRADE_CLASSES, value=GRADE_CLASSES, label="Select FinalGrade Classes")
            analyze_button = gr.Button("Analyze Data", variant="primary")
            analysis_metrics = gr.HTML()

            with gr.Row():
                grade_chart = gr.Plot()
                score_chart = gr.Plot()
            with gr.Row():
                study_chart = gr.Plot()
                attendance_chart = gr.Plot()
            correlation_chart = gr.Plot()

            gr.Markdown("## Filtered Dataset")
            filtered_table = gr.Dataframe(interactive=False)

            analyze_button.click(
                fn=analyze_data,
                inputs=[age_slider, attendance_slider, study_slider, grade_filter],
                outputs=[analysis_metrics, grade_chart, score_chart, study_chart, attendance_chart, correlation_chart, filtered_table]
            )

        with gr.Tab("ᎁ Student Prediction"):
            gr.Markdown("## Predict Student Performance

Enter student information using the same numeric coding and scales as the dataset.")
            prediction_inputs = []
            with gr.Row():
                with gr.Column():
                    for feature in FEATURES[:8]:
                        values = df[feature].dropna()
                        prediction_inputs.append(gr.Number(label=feature, value=float(values.median()) if not values.empty else 0))
                with gr.Column():
                    for feature in FEATURES[8:]:
                        values = df[feature].dropna()
                        prediction_inputs.append(gr.Number(label=feature, value=float(values.median()) if not values.empty else 0))

            predict_button = gr.Button("ᎁ Predict FinalGrade", variant="primary")
            prediction_output = gr.HTML()
            probability_chart = gr.Plot()
            predict_button.click(fn=predict_student, inputs=prediction_inputs, outputs=[prediction_output, probability_chart])

        with gr.Tab("ᎀ Model Performance"):
            gr.Markdown("## Machine Learning Model")
            gr.HTML(f"""
            <div class="metrics">
                <div class="metric"><h3>Algorithm</h3><h1 style="font-size:20px">Random Forest</h1></div>
                <div class="metric"><h3>Accuracy</h3><h1>{accuracy:.2%}</h1></div>
                <div class="metric"><h3>Training Records</h3><h1>{len(X_train):,}</h1></div>
                <div class="metric"><h3>Testing Records</h3><h1>{len(X_test):,}</h1></div>
            </div>
            """)
            gr.Markdown("### Classification Report")
            report_table = gr.Dataframe(value=get_report(), interactive=False)

        with gr.Tab("᎙ About Project"):
            gr.Markdown("""
            # About Student Performance Classification
            ### Objective
To analyze student-related factors and predict the FinalGrade class using machine learning.
            ### Dataset
- Source file: merged_dataset.csv
- Target column: FinalGrade
- Input features: 15
            ### Technologies Used
- Python
- Gradio
- Pandas
- NumPy
- Plotly
- Scikit-learn
            """)

    gr.Markdown("---
<center>Student Performance AI Dashboard | Python • Gradio • Machine Learning</center>")

if __name__ == "__main__":
    demo.launch(share=False)
