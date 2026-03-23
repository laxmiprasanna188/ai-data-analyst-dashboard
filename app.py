import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional AI (uncomment if using OpenAI)
# import openai
# openai.api_key = "YOUR_API_KEY"

st.set_page_config(page_title="AI Data Analyst", layout="wide")

st.title("🤖 AI Data Analyst Dashboard")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # ---------------- DATA CLEANING ----------------
    st.subheader("🧹 Data Cleaning")
    df = df.dropna()
    st.success("Missing values removed")

    # ---------------- BASIC INFO ----------------
    st.subheader("📌 Dataset Info")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

    with col2:
        st.write("Column Names:")
        st.write(df.columns.tolist())

    # ---------------- NUMERIC COLUMNS ----------------
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # ---------------- AUTO VISUALIZATION ----------------
    st.subheader("📈 Automatic Charts")

    for col in numeric_cols:
        st.write(f"### {col} Distribution")
        fig, ax = plt.subplots()
        df[col].hist(ax=ax, bins=20)
        ax.set_title(col)
        st.pyplot(fig)

    # ---------------- CORRELATION HEATMAP ----------------
    if len(numeric_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ---------------- CUSTOM CHART ----------------
    st.subheader("🎯 Custom Visualization")

    col_x = st.selectbox("Select X-axis", df.columns)
    col_y = st.selectbox("Select Y-axis", numeric_cols)

    if st.button("Generate Chart"):
        fig, ax = plt.subplots()
        ax.scatter(df[col_x], df[col_y])
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        st.pyplot(fig)

    # ---------------- INSIGHTS (NO AI) ----------------
    st.subheader("🧠 Basic Insights")

    st.write("Summary Statistics:")
    st.write(df.describe())

    st.write("Mean Values:")
    st.write(df.mean(numeric_only=True))

    # ---------------- AI INSIGHTS (OPTIONAL) ----------------
    st.subheader("🤖 AI Insights (Optional)")

    if st.button("Generate AI Insights"):
        try:
            import openai
            openai.api_key = "YOUR_API_KEY"

            prompt = f"""
            Analyze this dataset and give key insights:
            {df.describe().to_string()}
            """

            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            st.success(response['choices'][0]['message']['content'])

        except:
            st.warning("Add your OpenAI API key to enable AI insights.")

    # ---------------- DOWNLOAD CLEAN DATA ----------------
    st.subheader("⬇️ Download Cleaned Data")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")

else:
    st.info("👆 Upload a CSV file to get started")
