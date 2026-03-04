import streamlit as st
import pandas as pd
import subprocess
import sys
import os
import uuid
import json

# 设置页面配置
st.set_page_config(
    page_title="结核杆菌表位预测",
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
st.markdown('<h1 class="main-title">🧬 结核杆菌表位预测工具</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">输入蛋白质序列（FASTA格式），AI将为你预测潜在的结核杆菌表位，辅助疫苗设计。</p>', unsafe_allow_html=True)

st.markdown("---")

# 输入区域
with st.container():
    st.subheader("📥 输入序列")
    fasta_text = st.text_area(
        "粘贴FASTA序列",
        height=200,
        placeholder="示例：\n>protein_name\nMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        key="fasta_text"
    )

    st.markdown("**或者上传FASTA文件**")
    uploaded_file = st.file_uploader(
        "选择文件 (支持 .fasta, .txt, .fa)",
        type=['fasta', 'txt', 'fa'],
        key="uploaded_file"
    )

    submit_button = st.button("🚀 开始预测", type="primary")

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
        st.warning("请先粘贴FASTA序列或上传文件。")
        st.stop()

    unique_id = str(uuid.uuid4())
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    input_path = None
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode('utf-8')
            if not file_content.strip():
                st.error("上传的文件为空。请检查文件内容。")
                st.stop()
            if not is_valid_fasta(file_content):
                st.error("文件内容不是有效的FASTA格式。第一行必须以 '>' 开头，且包含序列。")
                st.stop()
            input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
            with open(input_path, "w", encoding='utf-8') as f:
                f.write(file_content)
            st.info(f"已使用上传的文件: {uploaded_file.name}")
        except Exception as e:
            st.error(f"读取文件失败: {e}")
            st.stop()
    else:
        text_content = fasta_text.strip()
        if not text_content:
            st.warning("请输入FASTA序列。")
            st.stop()
        if not is_valid_fasta(text_content):
            st.error("输入的文本不是有效的FASTA格式。第一行必须以 '>' 开头，且包含序列。")
            st.stop()
        if not text_content.startswith('>'):
            text_content = ">user_provided_sequence\n" + text_content
        input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
        with open(input_path, "w", encoding='utf-8') as f:
            f.write(text_content)
        st.info("已使用粘贴的文本序列。")

    output_path = os.path.join(temp_dir, f"output_{unique_id}.csv")
    summary_path = output_path.replace('.csv', '_vaccine_summary.json')

    with st.spinner("🔬 AI模型正在预测中，请稍候..."):
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

                st.subheader("📊 预测结果")
                st.dataframe(df, use_container_width=True)

                # 下载按钮
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 下载结果 (CSV)",
                        data=f,
                        file_name=f"prediction_{unique_id}.csv",
                        mime="text/csv"
                    )

                # 统计信息
                st.subheader("📈 统计信息")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总表位数", len(df) - (1 if 'vaccine' in df['type'].values else 0))  # 减去疫苗行
                with col2:
                    if 'immunogenicity_score' in df.columns:
                        # 只计算表位的平均得分（排除疫苗行）
                        epitope_scores = df[df['type'] != 'vaccine']['immunogenicity_score']
                        if not epitope_scores.empty:
                            avg_score = epitope_scores.mean()
                            st.metric("平均免疫原性得分", f"{avg_score:.3f}")
                with col3:
                    if 'predicted_type' in df.columns:
                        tcell_count = (df['predicted_type'] == 'T-cell').sum()
                        st.metric("T细胞表位数", tcell_count)
                with col4:
                    if 'predicted_type' in df.columns:
                        bcell_count = (df['predicted_type'] == 'B-cell').sum()
                        st.metric("B细胞表位数", bcell_count)

                # 展示前5个高免疫原性表位（排除疫苗行）
                if 'immunogenicity_score' in df.columns:
                    st.subheader("🔝 高免疫原性表位 (前5)")
                    epitope_only = df[df['type'] != 'vaccine']
                    if not epitope_only.empty:
                        top5 = epitope_only.nlargest(5, 'immunogenicity_score')[['sequence', 'immunogenicity_score', 'predicted_type']]
                        st.table(top5)

                # 可选：显示一条简短提示，说明疫苗序列已包含在下载文件中
                if 'vaccine' in df['type'].values:
                    st.info("设计的疫苗序列已作为第一行包含在下载的CSV文件中。")

            else:
                st.error("预测完成，但未生成结果文件。请检查后台脚本。")
                if result.stderr:
                    with st.expander("查看错误日志"):
                        st.code(result.stderr)

        except subprocess.TimeoutExpired:
            st.error("⏰ 预测超时，请稍后重试或检查输入序列长度。")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ 运行AI模型时出错：{e.stderr}")
            with st.expander("查看错误日志"):
                st.code(e.stderr)
        except Exception as e:
            st.error(f"发生未知错误：{e}")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>基于深度学习的结核杆菌表位预测工具 | 仅供研究使用</p>",
    unsafe_allow_html=True
)
