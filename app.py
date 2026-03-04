import streamlit as st
import pandas as pd
import subprocess
import os
import uuid
import tempfile

# --- 页面标题 ---
st.title("结核杆菌表位预测工具")
st.write("输入蛋白质序列（FASTA格式），AI将为你预测可能的结核杆菌表位。")

# --- 选择输入方式 ---
input_method = st.radio(
    "选择输入方式：",
    ("上传FASTA文件", "粘贴FASTA文本")
)

input_path = None  # 用于存储最终输入文件的路径
unique_id = str(uuid.uuid4())

if input_method == "上传FASTA文件":
    uploaded_file = st.file_uploader("选择FASTA文件", type=['fasta', 'txt', 'fa'])
    if uploaded_file is not None:
        # 保存上传的文件到临时目录
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"文件 '{uploaded_file.name}' 上传成功！")

else:  # 粘贴FASTA文本
    fasta_text = st.text_area(
        "在这里粘贴FASTA格式的序列",
        height=200,
        placeholder=">序列名称（可选）\nMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG\n..."
    )
    if fasta_text:
        # 检查文本是否为空
        if fasta_text.strip():
            # 如果用户没有提供FASTA头，自动添加一个
            if not fasta_text.strip().startswith('>'):
                fasta_text = ">user_provided_sequence\n" + fasta_text.strip()
            # 保存到临时文件
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            input_path = os.path.join(temp_dir, f"input_{unique_id}.fasta")
            with open(input_path, "w") as f:
                f.write(fasta_text)
            st.success("文本已接收，准备预测...")
        else:
            st.warning("请输入FASTA序列")

# --- 当有了输入文件路径后，执行预测 ---
if input_path is not None:
    # 定义输出文件路径
    output_path = os.path.join("temp", f"output_{unique_id}.csv")

    # 显示进度
    with st.spinner("正在运行AI模型进行预测，请稍候..."):
        try:
            # 调用你的AI项目
            # 注意：根据你的8vaccine_design_pipeline.py，它需要 --input 和 --output 参数
            result = subprocess.run(
                ['python', '8vaccine_design_pipeline.py', '--input', input_path, '--output', output_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 设置超时时间5分钟，防止无限等待
            )

            # 检查输出文件是否存在
            if os.path.exists(output_path):
                # 读取CSV结果
                df = pd.read_csv(output_path)
                st.subheader("预测结果")
                st.dataframe(df)

                # 提供下载按钮
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="下载预测结果 (CSV)",
                        data=f,
                        file_name=f"prediction_{unique_id}.csv",
                        mime="text/csv"
                    )

                # 可选：显示一些统计信息
                st.subheader("结果统计")
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
                        st.metric("B细胞表位", bcell_count)
            else:
                st.error("预测完成，但未生成结果文件。请检查后台脚本。")
                # 显示错误日志以供调试
                if result.stderr:
                    st.text("错误日志：")
                    st.code(result.stderr)

        except subprocess.TimeoutExpired:
            st.error("预测超时，请稍后重试或检查输入序列长度。")
        except subprocess.CalledProcessError as e:
            st.error(f"运行AI模型时出错：{e.stderr}")
            # 显示错误日志
            if e.stderr:
                st.text("错误日志：")
                st.code(e.stderr)
        except Exception as e:
            st.error(f"发生未知错误：{e}")

# --- 清理临时文件（可选）---
# 注意：Streamlit Cloud 的临时目录会在每次部署时重置，但为了防止占用空间，
# 可以在 session 结束时清理，但 Streamlit 是无状态的，这里暂不处理。
