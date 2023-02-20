import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MalTrace", page_icon="📊", layout="wide")


@st.experimental_singleton
def load_multi_class():
    # backdoor botnet ddos keylogger ransomware rootkit sniff spam spoof spyware trojan worm virus
    windows_list = [1072, 591, 714, 2616, 997, 259, 748, 1116, 824, 274, 1056, 553, 2633]
    linux_list = [880, 389, 795, 926, 331, 493, 917, 470, 818, 101, 444, 246, 851]
    mac_list = [77, 43, 49, 118, 44, 25, 228, 65, 505, 12, 69, 39, 121]
    android_list = [160, 91, 109, 116, 121, 64, 125, 129, 112, 79, 180, 95, 280]
    iphone_list = [77, 34, 71, 61, 55, 30, 67, 50, 108, 38, 60, 42, 100]
    iot_list = [58, 115, 80, 58, 54, 32, 68, 67, 33, 30, 82, 66, 140]
    multi_class = pd.DataFrame([windows_list, linux_list, mac_list, android_list, iphone_list, iot_list],
    index=['windows', 'linux', 'mac', 'android', 'iphone', 'iot'],
    columns=['backdoor', 'botnet', 'ddos', 'keylogger', 'ransomware', 'rootkit', 'sniff', 'spam', 'spoof', 'spyware', 'trojan', 'worm', 'virus'])
    return multi_class

@st.experimental_singleton
def load_hackers():
    # backdoor botnet ddos keylogger ransomware rootkit sniff spam spoof spyware trojan worm virus
    list_1 = ["screetsec", 20, 2651, 7910, 623, "screetsec/TheFatRat", "backdoor", "6.8k", "2k"]
    list_2 = ["n1nj4sec", 13, 1397, 6985, 474, "n1nj4sec/pupy", "RAT", "7.1k", "1.7k"]
    list_3 = ["RoganDawes", 1, 201, 3394, 280, "RoganDawes/P4wnP1", "USB attack", "3.4k", "0.6k"]
    list_4 = ["byt3bl33d3r", 17, 5667, 1431, 148, "Porchetta-Industries/ckMapExec", "pentest", "6.2k", "1.3k"]
    list_5 = ["ytisf", 17, 1013, 8444, 805, "ytisf/theZoo", "all", "8.8k", "2.2k"]
    list_6 = ["NYAN-x-CAT", 14, 2289, 2188, 253, "NYAN-x-CAT/AsyncRAT-C-Sharp", "RAT", "1.4k", "0.6k"]
    list_7 = ["pankoza-pl", 18, 71, 46, 42, "pankoza-pl/malwaredatabase", "all", "38", "24"]
    list_8 = ["noob-hackers", 16, 5478, 1560, 137, "noob-hackers/infect", "virus", "1.3k", "0.2k"]
    list_9 = ["swagkarna", 16, 1096, 868, 50, "swagkarna/Defeat-Defender-V1.2", "payload", "0.9k", "0.2k"]
    list_10 = ["s0md3v", 12, 5755, 440, 35, "s0md3v/Hash-Buster", "crack", "1.4k", "0.4k"]
    list_11 = ["malwaredllc", 12, 895, 7400, 313, "malwaredllc/byob", "botnet", "7.6k", "1.9k"]
    list_12 = ["Bitwise-01", 9, 1347, 3084, 662, "Bitwise-01/Instagram-", "brute force", "2.8k", "1.5k"]
    index_ = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    columns_ = ["username", "repos", "followers", "stars", "watchers", "famous repo", "famous repo's type",
                "famous repo's stars", "famous repo's forks"]

    hackers = pd.DataFrame([list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8, list_9, list_10, list_11, list_12],
                    index=index_, columns=columns_)
    return hackers

@st.experimental_singleton
def create_fig_1():
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(8, 7), dpi=400)

    # new_every_year
    new_comer = [61, 135, 248, 426, 704, 1101, 1744, 2414, 3463, 3976, 4802, 6496, 8402]
    year = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

    ax_1.set_title("New malware repositories per year")
    plt.xticks(rotation=60)  # 设置横坐标显示的角度，角度是逆时针
    N = len(year)  # 柱子总数
    values = new_comer  # 包含每个柱子对应值的序列
    index = np.arange(N)  # 包含每个柱子下标的序列
    width = 0.35  # 柱子的宽度
    ax_1.bar(index, values, width, label=None, color="#87CEFA")  # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    ax_1.set_xlabel('Year')  # 设置横轴标签
    ax_1.set_ylabel('Number of Repository')  # 设置纵轴标签
    plt.xticks(index, ('2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
                       '2020', '2021'))
    plt.yticks()
    for x, y in enumerate(values):
        plt.text(x, y + 100, '%s' % y, ha='center', va='bottom')
    ax_1 = plt.gca()  # gca:get current axis得到当前轴
    ax_1.spines['right'].set_color('none')
    ax_1.spines['top'].set_color('none')

    # every family
    fig_2, ax_2 = plt.subplots(1, 1, figsize=(8, 6), dpi=400)
    x = np.array(['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020'
                     , '2021'])  # x坐标
    y_backdoor = [3, 3, 15, 35, 47, 87, 165, 244, 378, 324, 443, 580, 693]
    y_botnet = [1, 5, 10, 17, 32, 58, 97, 156, 214, 235, 244, 314, 338]
    y_ddos = [0, 3, 7, 20, 34, 67, 123, 173, 319, 385, 518, 722, 955]
    y_keylogger = [2, 3, 17, 24, 73, 111, 205, 323, 581, 669, 882, 1308, 1672]
    y_ransomware = [0, 0, 0, 1, 0, 4, 26, 130, 243, 188, 247, 360, 538]
    y_rootkit = [1, 2, 6, 17, 41, 63, 75, 90, 106, 130, 135, 138, 152]
    y_sniff = [5, 18, 42, 54, 115, 137, 249, 259, 368, 335, 350, 412, 424]
    y_spam = [9, 13, 19, 38, 62, 110, 132, 183, 262, 362, 420, 787, 1157]
    y_spoof = [1, 9, 27, 53, 69, 102, 138, 211, 275, 369, 480, 643, 722]
    y_spyware = [0, 1, 1, 3, 8, 7, 21, 22, 48, 59, 77, 97, 138]
    y_trojan = [1, 4, 5, 16, 23, 41, 107, 131, 205, 207, 272, 382, 510]
    y_worm = [4, 5, 16, 29, 55, 77, 110, 132, 171, 177, 172, 162, 255]
    y_virus = [2, 8, 17, 54, 78, 119, 224, 326, 434, 551, 587, 972, 1280]

    ax_2.set_title("New repositories per type of malware per year")
    plt.xticks(rotation=60)  # 设置横坐标显示的角度，角度是逆时针，自己看
    plt.plot(x, y_backdoor, lw=1, c='black', marker='^', ms=4, label='backdoor')  # 绘制y1
    plt.plot(x, y_botnet, lw=1, c='r', marker='*', label='botnet')  # 绘制y2
    plt.plot(x, y_ddos, lw=1, c='black', marker='o', label='ddos')  # 绘制y2
    plt.plot(x, y_keylogger, lw=1, c='green', marker='h', label='keylogger')  # 绘制y2
    plt.plot(x, y_ransomware, lw=1, c='black', marker='x', label='ransomware')  # 绘制y2
    plt.plot(x, y_rootkit, lw=1, c='c', marker='*', label='rootkit')  # 绘制y2
    plt.plot(x, y_sniff, lw=1, c='gray', marker='p', label='sniff')  # 绘制y2
    plt.plot(x, y_spam, lw=1, c='orange', marker='^', label='spam')  # 绘制y2
    plt.plot(x, y_spoof, lw=1, c='y', marker='x', label='spoof')  # 绘制y2
    plt.plot(x, y_spyware, lw=1, c='g', marker='p', label='spyware')  # 绘制y2
    plt.plot(x, y_trojan, lw=1, c='orangered', marker='D', label='trojan')  # 绘制y2
    plt.plot(x, y_worm, lw=1, c='purple', marker='o', label='worm')  # 绘制y2
    plt.plot(x, y_virus, lw=1, c='gray', marker='s', label='virus')  # 绘制y2
    plt.xticks(x)  # x轴的刻度
    plt.xlabel('Year')  # x轴标注
    plt.ylabel('Number of Repository')  # y轴标注
    plt.legend()  # 图例
    plt.grid()  # 生成网格

    # every target
    fig_3, ax_3 = plt.subplots(1, 1, figsize=(8, 6), dpi=400)
    x = np.array(['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020',
                  '2021'])  # x坐标
    y_windows = [14, 37, 74, 129, 198, 328, 606, 765, 1217, 1324, 1512, 2021, 2656]
    y_linux = [14, 29, 71, 90, 138, 226, 360, 497, 696, 755, 821, 1035, 1211]
    y_mac = [1, 5, 14, 21, 38, 55, 72, 73, 117, 112, 143, 187, 176]
    y_android = [2, 4, 7, 27, 49, 57, 106, 129, 147, 157, 170, 176, 207]
    y_iphone = [1, 0, 8, 7, 9, 23, 24, 46, 58, 71, 71, 75, 101]
    y_iot = [0, 5, 8, 4, 5, 10, 17, 46, 48, 68, 63, 81, 101]

    ax_3.set_title("New malware repositories per target platform per year")
    plt.xticks(rotation=60)  # 设置横坐标显示的角度，角度是逆时针，自己看
    plt.plot(x, y_windows, lw=1, c='blue', marker='s', ms=4, label='windows')  # 绘制y1
    plt.plot(x, y_linux, lw=1, c='red', marker='o', label='linux')  # 绘制y2
    plt.plot(x, y_mac, lw=1, c='brown', marker='o', label='mac')  # 绘制y2
    plt.plot(x, y_android, lw=1, c='green', marker='o', label='android')  # 绘制y2
    plt.plot(x, y_iphone, lw=1, c='m', marker='o', label='iphone')  # 绘制y2
    plt.plot(x, y_iot, lw=1, c='c', marker='o', label='iot')  # 绘制y2
    plt.xticks(x)  # x轴的刻度
    plt.xlabel('Year')  # x轴标注
    plt.ylabel('Number of Repository')  # y轴标注
    plt.legend()  # 图例
    plt.grid()  # 生成网格

    return fig_1, fig_2, fig_3

@st.experimental_singleton
def create_fig_2():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=400)
    x = [1, 25, 57, 60, 63, 75, 93, 156, 200, 220]
    y = [5856, 2174, 1621, 1057, 445, 128, 39, 22, 12, 1]
    x.reverse()
    y.reverse()
    # 柱子的宽度
    width = 2
    # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.xticks(rotation=60)  # 设置横坐标显示的角度，角度是逆时针，自己看
    # 绘制柱状图, 每根柱子的颜色为紫罗兰色
    p2 = plt.bar(x, y, width, label=None, color="brown")  # tick_label=x
    for a, b in zip(x, y):
        plt.text(a, b, b, ha='left', va='bottom')
    # 设置横轴标签
    plt.xlabel('# Iteration')
    # 设置纵轴标签
    plt.ylabel('# Nodes Changed Values')
    # x 轴显示的刻度
    x_index = ['1', '58', '60', '63', '66', '156', '248']
    return fig

@st.experimental_singleton
def create_fig_3():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # , dpi=400)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    y = [20, 15, 15, 13, 12, 7, 6, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
    x_major_locator = "Community ID"
    y_major_locator = "Number of member"

    plt.xlabel(x_major_locator)
    plt.ylabel(y_major_locator)

    # 添加纵横轴的刻度
    plt.xticks([0, 4, 8, 12, 16, 20, 24])
    plt.yticks([0, 4, 8, 12, 16, 20, 24])
    plt.xlim(0, 24)
    plt.ylim(0, 24)
    plt.vlines(10, -5, 30, 'red', 'dashed')

    plt.scatter(x, y, s=25)
    return fig



cols = st.columns(6)
cols[0].caption("By Ilove510")
cols[-1].caption("第十五届信息安全作品赛")

st.write("# MalTrace")
st.write("""**MalTrace基于Github进行恶意软件情报分析**""")

st.subheader("图表分析", anchor="图表分析")
fig_1, fig_2, fig_3 = create_fig_1()
st.markdown(
    """
    ##### 基于恶意软件类型和恶意软件目标平台的数据集
"""
)
_, temp, _ = st.columns([1, 6, 1])
temp.dataframe(load_multi_class())
st.markdown(
    """
    ##### 每年新增恶意软件仓库数量的变化
    - 总的恶意软件仓库
    - 每种类型的恶意软件仓库
    - 每个目标平台的恶意软件仓库
"""
)
tab1, tab2, tab3 = st.tabs(["总的恶意软件仓库", "每种类型的恶意软件仓库", "每个目标平台的恶意软件仓库"])
with tab1:
    tab1_1, tab1_2, tab1_3 = st.columns([1.8, 4, 2])
    tab1_2.pyplot(fig_1)
with tab2:
    tab2_1, tab2_2, tab2_3 = st.columns([1.8, 4, 2])
    tab2_2.pyplot(fig_2)

with tab3:
    tab3_1, tab3_2, tab3_3 = st.columns([1.8, 4, 2])
    tab3_2.pyplot(fig_3)

st.subheader("识别有影响力的恶意软件作者", anchor="识别有影响力的恶意软件作者")
st.markdown(
    """
    ##### MalHITS算法迭代图
"""
)
_, temp, _ = st.columns([1.8, 4, 2])
temp.pyplot(create_fig_2())
st.markdown(
    """
    ##### 专业黑客举例
"""
)
_, temp, _ = st.columns([1, 6, 1])
temp.dataframe(load_hackers(), height=458)

st.markdown(
    """
    ##### 黑客活动举例
"""
)
line1_1, line1_2, line1_3 = st.columns(3)
line2_1, line2_2, line2_3 = st.columns(3)
line1_1.markdown("""
    ###### Malwaredllc
    在YouTube上发布视频，以推广自己在Github上的僵尸网络框架仓库(https://github.com/malwaredllc/byob)
""")
line1_2.markdown("""
    ###### NYAN-x-CAT 
    知名黑客，录制YouTube黑帽黑客教学视频，观看次数超过150,000
""")
line1_3.markdown("""
    ###### pankoza-pl
    在YouTube上发布编写木马感染电脑的视频(https://www.youtube.com/watch?v=b_weNggTL9U&t=20s) ，以推广自己的恶意软件仓库，发布8个月达到近万观看量

""")
line2_1.markdown("""
    ###### s0md3v 
    在区块链平台openware上自称“hash克星”，声称可以在几秒之内破解哈希，他还在该平台上推广自己在Github上的仓库(https://github.com/s0md3v/Hash-Buster)  
""")
line2_2.markdown("""
    ###### noob-hackers
    黑客组织(https://github.com/noob-hackers) ，在YouTube上拥有专属频道发布恶意软件开发教程
""")
line2_3.markdown("""
    ###### byt3bl33d3r
    在创作者平台patreon上推广他在Github上的多种类型的攻击性恶意软件仓库(https://github.com/byt3bl33d3r/MITMf) ，在个人博客上发布攻击性安全研究和教程(https://byt3bl33d3r.github.io/)
""")

st.subheader("社区分析", anchor="社区分析")
st.markdown(
    """
    ###### 社区大小分布图
"""
)
_, temp, _ = st.columns([1.8, 4, 2])
temp.pyplot(create_fig_3())
