import streamlit as st
import pandas as pd
import subprocess
import os
import uuid
import tempfile

# 设置页面配置（标题、图标、布局）
st.set_page_config(
    page_title="结核杆菌表位预测",
    page_icon="🧬",
    layout="centered"  # 内容居中
)

# 自定义CSS，让内容区域更美观
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
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 页面标题和介绍
st.markdown('<h1 class="main-title">🧬 结核杆菌表位预测工具</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">输入蛋白质序列（FASTA格式），AI将为你预测潜在的结核杆菌表位，辅助疫苗设计。</p>', unsafe_allow_html=True)

# --- 创建两个列，用于放置主要内容的容器（实际上由于layout=centered，已经是居中） ---
# 直接开始输入区域
st.markdown("---")  # 分割线

# 使用容器包裹输入部分，便于管理
input_container = st.container()

with input_container:
    st.subheader("📥 输入序列")

    # 文本输入区域（默认显示）
    fasta_text = st.text_area(
        "粘贴FASTA序列",
        height=200,
        placeholder="示例：\n>protein_name\nMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        key="fasta_text"
    )

    # 文件上传选项
    st.markdown("**或者上传FASTA文件**")
    uploaded_file = st.file_uploader(
        "选择文件 (支持 .fasta, .txt, .fa)",
        type=['fasta', 'txt', 'fa'],
        key="uploaded_file"
    )

    # 提交按钮
    submit_button = st.button("🚀 开始预测", type="primary")

st.markdown("---")  # 分割线

# --- 处理预测逻辑 ---
if submit_button:
    # 检查是否有输入
    if not fasta_text.strip() and uploaded_file is None:
        st.warning("请先粘贴FASTA序列或上传文件。")
        st.stop()  # 停止后续执行

    # 确定输入来源：优先使用文件上传，否则使用文本输入
    input_path = None
    unique_id = str(uuid.uuid4())
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    if uploaded_file is not None:
        # 使用上传的文件
        input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.info(f"已使用上传的文件: {uploaded_file.name}")
    else:
        # 使用文本输入
        # 如果用户没有提供FASTA头，自动添加一个
        text_content = fasta_text.strip()
        if not text_content.startswith('>'):
            text_content = ">user_provided_sequence\n" + text_content
        input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
        with open(input_path, "w") as f:
            f.write(text_content)
        st.info("已使用粘贴的文本序列。")

    # 定义输出文件路径
    output_path = os.path.join(temp_dir, f"output_{unique_id}.csv")

    # 显示加载动画
    with st.spinner("🔬 AI模型正在预测中，请稍候..."):
        try:
            # 调用后端脚本
            result = subprocess.run(
                ['python', '8vaccine_design_pipeline.py', '--input', input_path, '--output', output_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5分钟超时
            )

            # 检查输出文件是否存在
            if os.path.exists(output_path):
                # 读取结果
                df = pd.read_csv(output_path)

                # 显示结果
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

                # 统计信息卡片
                st.subheader("📈 统计信息")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总表位数", len(df))
                with col2:
                    if 'immunogenicity_score' in df.columns:
                        avg_score = df['immunogenicity_score'].mean()
                        st.metric("平均免疫原性得分", f"{avg_score:.3f}")
                with col3:
                    if 'predicted_type' in df.columns:
                        bcell_count = (df['predicted_type'] == 'B-cell').sum()
                        st.metric("B细胞表位数", bcell_count)

                # 可选：展示前几个表位
                st.subheader("🔝 高免疫原性表位 (前5)")
                if 'immunogenicity_score' in df.columns:
                    top5 = df.nlargest(5, 'immunogenicity_score')[['sequence', 'immunogenicity_score', 'predicted_type']]
                    st.table(top5)

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

# 页脚（可选）
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>基于深度学习的结核杆菌表位预测工具 | 仅供研究使用</p>",
    unsafe_allow_html=True
)
