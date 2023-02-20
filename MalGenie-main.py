import streamlit as st

st.set_page_config(page_title="MalGenie", page_icon="👋", layout="wide")

cols = st.columns(6)

cols[0].caption("By Ilove510")
cols[-1].caption("第十五届信息安全作品赛")

# st.balloons()
st.write("# MalGenie: 面向代码开源平台的恶意软件源代码检测系统")

st.sidebar.success("Welcome!")

st.markdown(
    """
    **MalGenie 是一个面向代码开源平台的恶意软件源代码检测系统，系统分为三个模块——MalMine、Mal2vec、MalTrace。**
    - MalMine&Mal2vec: 从Github识别恶意软件源代码仓库，获得恶意软件源代码数据集MalSet，并进行可视化。
    - MalTrace: 基于恶意软件源代码数据集MalSet，进行代码开源平台的恶意软件情报分析。  
    **👈 从侧边栏选择一个模块** 来查看MalGenie系统的具体示例。
    ### 动机
    - 现今，研究界的恶意软件源代码数据集非常匮乏，阻碍了恶意软件源代码研究的深入。
    - 而Github等代码开源平台有着大量的恶意软件源代码仓库，同时有着非常大的利用价值，比如中国黑客组织Deep Panda就利用了恶意仓库Empire。
    - 同时，专业黑客们积极创建恶意仓库，提高个人声誉，相互合作。
    因此，以Github为例，我们设计并实现了MalGenie：面向代码开源平台的恶意软件源代码检测系统。
    ### 代码
    MalGenie的Python参考实现在GitHub(https://github.com/dang-mai/MalGenie) 上可获得。
    ### 数据集
    我们公开MalGenie的数据集:
    - MalSet-恶意软件源代码数据集(http://124.221.95.158/MalSet.zip)
    - MalCur(http://124.221.95.158/MalCur.zip)
    ### 贡献者
    Ilove510团队
    ### 参考
    MalGenie: 面向代码开源平台的恶意软件源代码检测系统报告. Ilove510团队.
""")
