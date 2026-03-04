# app.py
import streamlit as st
import pandas as pd
import subprocess
import os
import uuid

# --- 页面标题 ---
st.title("结核杆菌表位预测工具")
st.write("上传你的蛋白质序列文件（FASTA格式），AI将为你预测可能的结核杆菌表位。")

# --- 文件上传部分 ---
# 创建一个文件上传器，只允许上传.fasta文件 [citation:3]
uploaded_file = st.file_uploader("选择FASTA文件", type=['fasta', 'txt'])

# --- 当用户上传文件后执行的操作 ---
if uploaded_file is not None:
    # 1. 保存用户上传的文件
    # 生成一个唯一的ID，避免文件名冲突
    unique_id = str(uuid.uuid4())
    # 定义输入文件的保存路径（放在一个名为 'temp' 的文件夹里）
    input_path = f"temp/input_{unique_id}.fasta"
    # 定义输出文件的保存路径
    output_path = f"temp/output_{unique_id}.csv"

    # 创建 'temp' 文件夹（如果它不存在的话）
    os.makedirs("temp", exist_ok=True)

    # 将用户上传的文件内容写入到我们指定的 input_path
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("文件上传成功！正在运行AI模型进行预测，请稍候...")

    # 2. 运行你的AI项目
    # 这里假设你的AI项目可以通过命令行方式运行，并且接受输入和输出文件路径作为参数
    # 请根据你项目的实际情况修改下面的命令 [citation:1]
    # 例如：你的项目叫 predict.py，可以用 python predict.py --input 文件路径 --output 文件路径
    try:
        # 这里用 echo 命令模拟你的AI项目运行，实际使用时需要替换成你的命令
        # result = subprocess.run(['python', 'your_project.py', '--input', input_path, '--output', output_path], capture_output=True, text=True, check=True)
        
        # === 请将上面的注释替换成你实际的命令，例如： ===
        result = subprocess.run(['python', 'your_project.py', '--input', input_path, '--output', output_path], capture_output=True, text=True, check=True)
        
        st.info("模型预测完成！")
        
        # 3. 读取并显示结果
        if os.path.exists(output_path):
            # 假设你的输出文件是CSV格式 [citation:4]
            df = pd.read_csv(output_path)
            st.subheader("预测结果")
            # 使用 st.dataframe 显示一个漂亮的、可交互的表格 [citation:9]
            st.dataframe(df)
            
            # 4. 提供下载按钮 [citation:4]
            with open(output_path, "rb") as f:
                st.download_button('点击下载预测结果', f, file_name=f"prediction_{unique_id}.csv")
        else:
            st.error("预测结果文件未生成，请检查你的AI项目。")

    except subprocess.CalledProcessError as e:
        st.error(f"运行AI模型时出错：{e.stderr}")
    except Exception as e:
        st.error(f"发生未知错误：{e}")

else:
    st.info("请在左侧上传你的FASTA文件。")