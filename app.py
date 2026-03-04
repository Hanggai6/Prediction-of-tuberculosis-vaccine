import sys
import subprocess
import streamlit as st
import pandas as pd
import os
import uuid
import json  # 新增导入

# ------------------ 调试信息（部署后可删除） ------------------
st.write(f"**当前 Python 解释器路径**: `{sys.executable}`")
st.write(f"**Python 版本**: `{sys.version}`")

try:
    import torch
    st.success(f"✅ torch 导入成功！版本: {torch.__version__}")
except ImportError as e:
    st.error(f"❌ torch 导入失败: {e}")

# 列出已安装的包（可选，可注释掉以节省空间）
# result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
# st.text("已安装的包:\n" + result.stdout)
# ------------------------------------------------------------

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

# 预测逻辑
if submit_button:
    if not fasta_text.strip() and uploaded_file is None:
        st.warning("请先粘贴FASTA序列或上传文件。")
        st.stop()

    # 准备输入文件
    input_path = None
    unique_id = str(uuid.uuid4())
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    if uploaded_file is not None:
        input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info(f"已使用上传的文件: {uploaded_file.name}")
    else:
        text_content = fasta_text.strip()
        if not text_content.startswith('>'):
            text_content = ">user_provided_sequence\n" + text_content
        input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
        with open(input_path, "w") as f:
            f.write(text_content)
        st.info("已使用粘贴的文本序列。")

    output_path = os.path.join(temp_dir, f"output_{unique_id}.csv")
    summary_path = output_path.replace('.csv', '_vaccine_summary.json')

    with st.spinner("🔬 AI模型正在预测中，请稍候..."):
        try:
            # 关键修改：使用 sys.executable 确保子进程使用同一 Python 解释器
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

                # 统计信息（增加B细胞/T细胞计数）
                st.subheader("📈 统计信息")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总表位数", len(df))
                with col2:
                    if 'immunogenicity_score' in df.columns:
                        avg_score = df['immunogenicity_score'].mean()
                        st.metric("平均免疫原性得分", f"{avg_score:.3f}")
                with col3:
                    if 'predicted_type' in df.columns:
                        tcell_count = (df['predicted_type'] == 'T-cell').sum()
                        st.metric("T细胞表位数", tcell_count)
                with col4:
                    if 'predicted_type' in df.columns:
                        bcell_count = (df['predicted_type'] == 'B-cell').sum()
                        st.metric("B细胞表位数", bcell_count)

                # 展示前5个高免疫原性表位
                if 'immunogenicity_score' in df.columns:
                    st.subheader("🔝 高免疫原性表位 (前5)")
                    top5 = df.nlargest(5, 'immunogenicity_score')[['sequence', 'immunogenicity_score', 'predicted_type']]
                    st.table(top5)

                # 读取并展示疫苗设计摘要
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        vaccine_summary = json.load(f)

                    st.subheader("🧬 设计的疫苗序列")
                    st.code(vaccine_summary['vaccine_sequence'], language='text')

                    # 疫苗详细信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("疫苗长度 (aa)", vaccine_summary['total_length'])
                        st.metric("连接子类型", vaccine_summary['linker_type'])
                    with col2:
                        st.metric("分子量 (Da)", f"{vaccine_summary['molecular_weight']:.1f}")
                        st.metric("等电点", f"{vaccine_summary['isoelectric_point']:.2f}")
                    with col3:
                        st.metric("平均免疫原性", f"{vaccine_summary['average_immunogenicity']:.3f}")
                        st.metric("表位总数", vaccine_summary['num_epitopes'])

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
