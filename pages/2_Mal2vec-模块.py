import streamlit as st
# import PIL
# from PIL import Image
# PIL.Image.MAX_IMAGE_PIXELS = 933120000

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

st.set_page_config(page_title="Mal2vec", page_icon="🌍", layout="wide")

@st.experimental_singleton
def scatter():
    color_list = ['backdoor', 'botnet', 'ddos', 'keylogger', 'ransomware', 'rootkit', 'sniff', 'spam', 'spoof',
                  'spyware', 'trojan', 'virus', 'worm']
    x = np.load("X.npy")
    colors = np.load("y.npy")

    choice = [
        45, 197, 21, 58, 161, 11, 100, 38, 154, 225, 126, 43, 68, 55, 104, 6, 76, 41, 7, 164, 18, 206, 32, 35, 172, 182,
        224, 138, 14, 29, 204, 170,
        175, 158, 179, 15, 71, 12, 51, 135, 62, 60, 101, 185, 152, 227, 66, 134, 198, 194, 193, 117, 1, 212, 67, 171,
        165, 39, 223, 109, 132, 65,
        47, 207, 37, 112, 163, 133, 4
    ]

    x = x[choice]
    colors = colors[choice]
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 13))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 6), dpi=800)
    ax = plt.subplot(aspect='equal')
    # sc = ax.scatter(x[:, 0], x[:, 1], lw=0,
    #                 c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # # We add the labels for each digit.
    # txts = []
    # for i in range(10):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, "", fontsize=0)  # str(color_list[i])
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    type0_x = []
    type0_y = []
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []
    type7_y = []
    type8_x = []
    type8_y = []
    type9_x = []
    type9_y = []
    type10_x = []
    type10_y = []
    type11_x = []
    type11_y = []
    type12_x = []
    type12_y = []

    for i in range(len(colors)):

        if colors[i] == 0:  # 第i行的label为0时
            type0_x.append(x[i][0])
            type0_y.append(x[i][1])
        if colors[i] == 1:  # 第i行的label为1时
            type1_x.append(x[i][0])
            type1_y.append(x[i][1])
        if colors[i] == 2:  # 第i行的label为2时
            type2_x.append(x[i][0])
            type2_y.append(x[i][1])
        if colors[i] == 3:  # 第i行的label为3时
            type3_x.append(x[i][0])
            type3_y.append(x[i][1])
        if colors[i] == 4:  # 第i行的label为4时
            type4_x.append(x[i][0])
            type4_y.append(x[i][1])
        if colors[i] == 5:  # 第i行的label为5时
            type5_x.append(x[i][0])
            type5_y.append(x[i][1])
        if colors[i] == 6:  # 第i行的label为6时
            type6_x.append(x[i][0])
            type6_y.append(x[i][1])
        if colors[i] == 7:  # 第i行的label为7时
            type7_x.append(x[i][0])
            type7_y.append(x[i][1])
        if colors[i] == 8:  # 第i行的label为8时
            type8_x.append(x[i][0])
            type8_y.append(x[i][1])
        if colors[i] == 9:  # 第i行的label为9时
            type9_x.append(x[i][0])
            type9_y.append(x[i][1])
        if colors[i] == 10:  # 第i行的label为10时
            type10_x.append(x[i][0])
            type10_y.append(x[i][1])
        if colors[i] == 11:  # 第i行的label为11时
            type11_x.append(x[i][0])
            type11_y.append(x[i][1])
        if colors[i] == 12:  # 第i行的label为12时
            type12_x.append(x[i][0])
            type12_y.append(x[i][1])
    # fig = plt.figure(figsize = (10, 6))
    # ax = fig.add_subplot(111)
    type0 = ax.scatter(type0_x, type0_y, c='brown', label=color_list[0])
    type1 = ax.scatter(type1_x, type1_y, c='lime', label=color_list[1])
    type2 = ax.scatter(type2_x, type2_y, c="darkviolet", label=color_list[2])
    type3 = ax.scatter(type3_x, type3_y, c='black', label=color_list[3])
    type4 = ax.scatter(type4_x, type4_y, c='r', label=color_list[4])
    type5 = ax.scatter(type5_x, type5_y, c="green", label=color_list[5])
    type6 = ax.scatter(type6_x, type6_y, c='c', label=color_list[6])
    type7 = ax.scatter(type7_x, type7_y, c='gray', label=color_list[7])
    type8 = ax.scatter(type8_x, type8_y, c="orange", label=color_list[8])
    type9 = ax.scatter(type9_x, type9_y, c='y', label=color_list[9])
    type10 = ax.scatter(type10_x, type10_y, c='cyan', label=color_list[10])
    type11 = ax.scatter(type11_x, type11_y, c="blue", label=color_list[11])
    type12 = ax.scatter(type12_x, type12_y, c="purple", label=color_list[12])
    plt.legend(loc='upper right')
    return f

# @st.experimental_singleton
# def load_image():
#     image = Image.open('vectors_visualization.png')
#     return image

cols = st.columns(6)

cols[0].caption("By Ilove510")
cols[-1].caption("第十五届信息安全作品赛")

st.write("# Mal2vec")
st.write("""**Mal2vec实现恶意软件源代码仓库的向量表示**""")

st.subheader("降维可视化")
st.pyplot(scatter())
# image = load_image()
# st.image(image, caption="基于恶意软件类型的MalCur数据集的降维可视化")
st.subheader("结论")
st.markdown("""
**不同类型的恶意软件仓库的向量各成一簇，向量表示方法有效**
""")
