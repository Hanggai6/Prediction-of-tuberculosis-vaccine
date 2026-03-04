import streamlit as st
import pandas as pd
import subprocess
import sys
import os
import uuid
import json

# 设置页面配置
st.set_page_config(
    page_title="Tuberculosis Epitope Prediction",
    page_icon="🧬",
    layout="centered"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-family: monospace;
        font-size: 14px;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        width: 200px;
        margin: 0 auto;
        display: block;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.markdown('<h1 class="main-title">🧬 Tuberculosis Epitope Prediction Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Input protein sequence in FASTA format, and AI will predict potential tuberculosis epitopes to assist vaccine design.</p>', unsafe_allow_html=True)

st.markdown("---")

# 输入区域
with st.container():
    st.subheader("📥 Input Sequence")
    fasta_text = st.text_area(
        "Paste FASTA sequence",
        height=200,
        placeholder="Example:\n>protein_name\nMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        key="fasta_text"
    )

    st.markdown("**Or upload FASTA file**")
    uploaded_file = st.file_uploader(
        "Choose file (supported: .fasta, .txt, .fa)",
        type=['fasta', 'txt', 'fa'],
        key="uploaded_file"
    )

    submit_button = st.button("🚀 Start Prediction", type="primary")

st.markdown("---")

def is_valid_fasta(content):
    """简单检查是否为有效FASTA格式（至少有一个以>开头的行，且后续有非空序列）"""
    lines = content.strip().split('\n')
    if not lines:
        return False
    if not lines[0].startswith('>'):
        return False
    has_sequence = False
    for line in lines[1:]:
        if line.strip() and not line.startswith('>'):
            has_sequence = True
            break
    return has_sequence

# 预测逻辑
if submit_button:
    if not fasta_text.strip() and uploaded_file is None:
        st.warning("Please paste FASTA sequence or upload a file.")
        st.stop()

    unique_id = str(uuid.uuid4())
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    input_path = None
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode('utf-8')
            if not file_content.strip():
                st.error("Uploaded file is empty. Please check the file content.")
                st.stop()
            if not is_valid_fasta(file_content):
                st.error("File content is not a valid FASTA format. The first line must start with '>' and contain a sequence.")
                st.stop()
            input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
            with open(input_path, "w", encoding='utf-8') as f:
                f.write(file_content)
            st.info(f"Using uploaded file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()
    else:
        text_content = fasta_text.strip()
        if not text_content:
            st.warning("Please enter FASTA sequence.")
            st.stop()
        if not is_valid_fasta(text_content):
            st.error("Input text is not a valid FASTA format. The first line must start with '>' and contain a sequence.")
            st.stop()
        if not text_content.startswith('>'):
            text_content = ">user_provided_sequence\n" + text_content
        input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
        with open(input_path, "w", encoding='utf-8') as f:
            f.write(text_content)
        st.info("Using pasted text sequence.")

    output_path = os.path.join(temp_dir, f"output_{unique_id}.csv")
    summary_path = output_path.replace('.csv', '_vaccine_summary.json')

    with st.spinner("🔬 AI model is predicting, please wait..."):
        try:
            result = subprocess.run(
                [sys.executable, '8vaccine_design_pipeline.py', '--input', input_path, '--output', output_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=300
            )

            if os.path.exists(output_path):
                df = pd.read_csv(output_path)

                st.subheader("📊 Prediction Results")
                st.dataframe(df, use_container_width=True)

                # 下载按钮
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=f,
                        file_name=f"prediction_{unique_id}.csv",
                        mime="text/csv"
                    )

                # 统计信息
                st.subheader("📈 Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Epitopes", len(df) - (1 if 'vaccine' in df['type'].values else 0))
                with col2:
                    if 'immunogenicity_score' in df.columns:
                        epitope_scores = df[df['type'] != 'vaccine']['immunogenicity_score']
                        if not epitope_scores.empty:
                            avg_score = epitope_scores.mean()
                            st.metric("Avg Immunogenicity Score", f"{avg_score:.3f}")
                with col3:
                    if 'predicted_type' in df.columns:
                        tcell_count = (df['predicted_type'] == 'T-cell').sum()
                        st.metric("T-cell Epitopes", tcell_count)
                with col4:
                    if 'predicted_type' in df.columns:
                        bcell_count = (df['predicted_type'] == 'B-cell').sum()
                        st.metric("B-cell Epitopes", bcell_count)

                # 展示前5个高免疫原性表位（排除疫苗行）
                if 'immunogenicity_score' in df.columns:
                    st.subheader("🔝 Top 5 High Immunogenicity Epitopes")
                    epitope_only = df[df['type'] != 'vaccine']
                    if not epitope_only.empty:
                        top5 = epitope_only.nlargest(5, 'immunogenicity_score')[['sequence', 'immunogenicity_score', 'predicted_type']]
                        st.table(top5)

                # 显示一条简短提示，说明疫苗序列已包含在下载文件中
                if 'vaccine' in df['type'].values:
                    st.info("The designed vaccine sequence is included as the first row in the downloaded CSV file.")

            else:
                st.error("Prediction completed but result file was not generated. Please check the backend script.")
                if result.stderr:
                    with st.expander("View error log"):
                        st.code(result.stderr)

        except subprocess.TimeoutExpired:
            st.error("⏰ Prediction timeout. Please try again later or check sequence length.")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Error running AI model: {e.stderr}")
            with st.expander("View error log"):
                st.code(e.stderr)
        except Exception as e:
            st.error(f"An unknown error occurred: {e}")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Deep learning-based tuberculosis epitope prediction tool | For research use only</p>",
    unsafe_allow_html=True
)
