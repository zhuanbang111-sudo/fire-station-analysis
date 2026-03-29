import streamlit as st
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
# ==============================================================================
# 第一部分：导入系统所需的“核心工具组件”
# ==============================================================================
import streamlit as st          # 网页界面框架：负责把 Python 逻辑变成可操作的网页
import pandas as pd             # 表格处理工具：负责 Excel 读取与统计报表生成
import geopandas as gpd         # 空间地理工具：负责处理地图多边形与投影坐标计算
import requests                 # 网络通讯工具：模拟浏览器向高德地图请求实时路况数据
import math                     # 数学函数库：提供三角函数、平方根等基础空间计算能力
import time                     # 时间工具：记录日志时间，并控制 API 请求的频率间隔
import os                       # 操作系统接口：处理本地文件夹路径、创建临时文件
import folium                   # 地图引擎：在网页上绘制可缩放、可切换底图的交互式地图
from streamlit_folium import st_folium  # 网页桥接：将 Folium 地图对象嵌入 Streamlit 网页
import numpy as np              # 数值计算库：处理 300x300 高密度网格矩阵的极速运算
from scipy.spatial import cKDTree       # 空间索引算法：在几十万个点中瞬间找到“离我最近的路”
from skimage import measure             # 图像算法库：从连续的时间场中提取出“等高线”多边形
from shapely.geometry import Polygon    # 几何定义：在代码中构建标准的“面状”地理要素
from shapely.ops import unary_union     # 几何合并：把多个细碎、重叠的小圈融合成一个完整大圈
import tempfile                 # 临时空间：创建一个自动销毁的文件夹，用于存放打包 SHP 文件
import zipfile                  # 压缩打包：将 Shapefile 相关的多个文件合成一个 ZIP 压缩包
from io import BytesIO          # 内存缓冲区：在内存中模拟文件读写，提速下载并节省硬盘空间

# ==============================================================================
# 第二部分：坐标纠偏算法 (将高德“火星坐标”还原为“地球真实坐标”)
# ==============================================================================
def gcj02_to_wgs84(lng, lat):
    """ 
    由于国家测绘要求，高德地图坐标经过了非线性加密。
    如果不进行纠偏处理，生成的等时圈在标准卫星底图上会产生 300-500 米的偏移。
    """
    a = 6378137.0; ee = 0.00669342162296594323; pi = 3.1415926535897932384626

    def _transform_lat(x, y): # 内部辅助函数：计算纬度的偏移增量
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y * pi / 12.0) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
        return ret

    def _transform_lng(x, y): # 内部辅助函数：计算经度的偏移增量
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0
        return ret

    dlat = _transform_lat(lng - 105.0, lat - 35.0) # 计算纬度偏差
    dlng = _transform_lng(lng - 105.0, lat - 35.0) # 计算经度偏差
    radlat = lat / 180.0 * pi # 弧度转换
    magic = math.sin(radlat); magic = 1 - ee * magic * magic; sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi) # 纬度修正公式
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)     # 经度修正公式
    return lng - dlng, lat - dlat # 返回真正物理世界的经纬度

# ==============================================================================
# 第三部分：智能多 Key 轮询系统 (解决高德 API 每日调用额度限制)
# ==============================================================================
def smart_amap_request(url, params, api_keys, current_key_idx):
    """ 
    高德 API 每个 Key 有配额限制。当一张 Key 用完报“额度超限”时，
    程序会自动跳转到下一张 Key，实现 24 小时无间断大规模分析。
    """
    while current_key_idx < len(api_keys):
        params['key'] = api_keys[current_key_idx] # 配置当前活跃的 Key
        try:
            r = requests.get(url, params=params, timeout=5).json() # 发起网络请求
            # 如果状态码显示额度用尽 (10003, 10044) 或 Key 无效
            if r.get('status') == '0' and r.get('infocode') in ['10003', '10044', '10012']:
                current_key_idx += 1 # 自动切换到 Key 列表的下一个索引
                continue # 重新进入循环尝试新 Key
            return r, current_key_idx, True # 请求成功，返回数据包
        except:
            return None, current_key_idx, True # 网络闪断异常也视为一次尝试
    return None, current_key_idx, False # 所有的 Key 都已经报废

# ==============================================================================
# 第四部分：空间时间场算法 (本程序的核心“数学大脑”)
# ==============================================================================
def create_isoline_polygon(trail_points, target_sec, off_road_speed):
    """ 
    逻辑：将马路上离散的轨迹点，转化为覆盖全城的连续时间场。
    利用 300x300 像素网格，算出每个像素点在“车行时间+步行时间”下的通行总代价。
    """
    if len(trail_points) < 10: return None # 点太少无法构成有效等高线
    
    pts = np.array(trail_points); xy = pts[:, 0:2]; times = pts[:, 2] # 提取经纬度和时间
    # 投影修正：地球在不同纬度经度弧长不等，需乘以余弦值来保证空间计算的物理距离准确
    avg_lat = np.mean(xy[:, 1]); cos_lat = np.cos(np.radians(avg_lat))
    xy_scaled = xy.copy(); xy_scaled[:, 0] *= cos_lat 
    
    # 构建快速空间索引查找树：能在毫秒内算出像素点离马路有多远
    tree = cKDTree(xy_scaled); pad = 0.01 # 缓冲区 1.1 公里，防止等高线在边缘被生硬切断
    min_x, max_x = np.min(xy[:,0]) - pad, np.max(xy[:,0]) + pad
    min_y, max_y = np.min(xy[:,1]) - pad, np.max(xy[:,1]) + pad
    
    # 建立 300x300 的计算网格矩阵
    grid_size = 300 
    x_grid = np.linspace(min_x, max_x, grid_size); y_grid = np.linspace(min_y, max_y, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid); grid_xy = np.column_stack([X.ravel() * cos_lat, Y.ravel()])
    
    # 查询：每个像素点对应的“最近马路点”及其“直线距离”
    dists, indices = tree.query(grid_xy)
    # 核心代价公式：像素点总时间 = 消防车开到马路最近点的时间 + (离开马路后的直线距离 / 消防员步速)
    grid_T = times[indices] + (dists * 111320 / off_road_speed) 
    grid_T = grid_T.reshape((grid_size, grid_size)) # 还原为二维地形图像
    
    # 提取多边形：在地形图上找出数值刚好等于“设定时限”的轮廓线
    contours = measure.find_contours(grid_T, level=target_sec) 
    polygons = []
    for contour in contours:
        # 将像素网格坐标反向映射回真实地理经纬度
        c_lng = min_x + (contour[:, 1] / (grid_size - 1)) * (max_x - min_x)
        c_lat = min_y + (contour[:, 0] / (grid_size - 1)) * (max_y - min_y)
        if len(c_lng) >= 3:
            polygons.append(Polygon(np.column_stack([c_lng, c_lat]))) # 构造几何多边形
            
    # 融合：把可能重叠的多个小圈融合成一个完整的多边形，去掉空隙
    return unary_union(polygons) if polygons else None

# ==============================================================================
# 第五部分：路网实时爬虫引擎 (尊重真实路况数据)
# ==============================================================================
def run_cost_surface_engine(api_keys, key_idx, origin_lng, origin_lat, target_min, factor=0.8):
    """ 
    模拟消防车向四周扩散的过程：
    1. 搜索周围 5-15 公里内所有的建筑物、路口作为“模拟终点”。
    2. 向高德查询到达这些点的【实时路况】通行耗时。
    """
    # 动态推算搜索半径，最大 15 公里
    radius = min(int(target_min * 800 * 1.5), 15000) 
    anchors = []; api_calls = 0; url_around = "https://restapi.amap.com/v3/place/around"
    # 定义探测目标的分类代码 (如路口、车站、小区等)
    all_types = "190301|150700|190000|170000|120000|140000|090000|060000"

    # 阶段一：雷达探测。寻找消防站周边所有的“锚点”目的地
    for page in range(1, 13): # 最多探测 12 页共 600 个点
        p_params = {"location": f"{origin_lng:.6f},{origin_lat:.6f}", "radius": radius, "types": all_types, "offset": 50, "page": page}
        r, key_idx, is_called = smart_amap_request(url_around, p_params, api_keys, key_idx)
        if is_called: api_calls += 1
        if r and r.get('status') == '1': 
            anchors.extend([poi['location'] for poi in r['pois']]) # 存下经纬度
        else: break
    
    anchors = list(set(anchors)) # 去重，避免重复请求同一地点
    trail_points = []; o_lng, o_lat = gcj02_to_wgs84(origin_lng, origin_lat)
    trail_points.append((o_lng, o_lat, 0)) # 消防站起点的到达时间设为 0

    # 阶段二：路径规划。计算到每个锚点的真实车行耗时
    url_route = "https://restapi.amap.com/v3/direction/driving"
    for dest in anchors:
        params = {"origin": f"{origin_lng:.6f},{origin_lat:.6f}", "destination": dest, "strategy": 13} # 策略 13 代表躲避拥堵
        r, key_idx, is_called = smart_amap_request(url_route, params, api_keys, key_idx)
        if is_called: api_calls += 1
        if r and r.get('status') == '1' and r.get('route'):
            steps = r['route']['paths'][0]['steps'] # 获取导航步骤
            acc_t = 0 # 累计时间记录器
            for s in steps:
                dur = int(s['duration']) # 获取真实实时路况下的行驶秒数
                polyline = s['polyline'].split(';') # 提取该路段的转弯拐点集
                time_step = dur / max(1, len(polyline) - 1) # 平摊耗时到每个转弯点上
                for i, p in enumerate(polyline):
                    plng, plat = map(float, p.split(',')); w_lng, w_lat = gcj02_to_wgs84(plng, plat)
                    trail_points.append((w_lng, w_lat, acc_t + i * time_step)) # 将带时间戳的点存入点云
                acc_t += dur # 累加段耗时
                if acc_t > (target_min * 60 / factor) + 60: break # 如果已经超出时限，不再往更远处计算
        time.sleep(0.03) # 毫秒级停顿，模拟人类操作，保护 API 不被标记为机器人
    return trail_points, len(anchors), api_calls, key_idx

# ==============================================================================
# 第六部分：网页前端布局与状态初始化
# ==============================================================================
st.set_page_config(page_title="基于高德POI的消防站评估系统", layout="wide")

# 初始化 Session State 会话短期记忆，防止每次交互都清空计算结果
if 'iso_results' not in st.session_state: st.session_state.iso_results = [] # 存储多边形结果
if 'current_key_idx' not in st.session_state: st.session_state.current_key_idx = 0 # 存储 API Key 轮询位置
if 'map_renders' not in st.session_state: st.session_state.map_renders = 0 # 存储地图重绘次数，用于动态 Key
if 'logs' not in st.session_state: st.session_state.logs = [] # 存储运行日志

# --- 侧边栏：参数调节区 ---
st.sidebar.header("⚙️ 参数配置")
api_keys_input = st.sidebar.text_area("1. 输入高德 API Keys (多 Key 英文逗号分隔)")
api_key_list = [k.strip() for k in api_keys_input.split(',') if k.strip()]
t_limit = st.sidebar.slider("2. 可达性时间 (分钟)", 3, 15, 5) # 消防行业标准 5 分钟
factor = st.sidebar.slider("3. 消防特权通行系数", 0.7, 1.0, 0.8) # 模拟消防车闯红灯、走公交道的提速效果
walk_speed = st.sidebar.slider("4. 消防员下车后跑动速度 (米/秒)", 1.0, 5.0, 1.5, 0.1)
map_style = st.sidebar.selectbox("5. 地图显示风格", ("CartoDB positron", "OpenStreetMap", "CartoDB dark_matter"))
excel = st.sidebar.file_uploader("6. 上传包含 station_name, lng, lat 的消防站表格", type=["xlsx"])

# --- 主界面：左右 3:1 布局 ---
col_map, col_monitor = st.columns([3, 1])

with col_monitor:
    st.subheader("📊 运行监控")
    prog_bar = st.empty(); prog_txt = st.empty() # 进度条与百分比文字占位符
    st.divider()
    st.subheader("📜 运行日志")
    log_box = st.container(height=250, border=True) # 带自动滚动条的日志窗口
    st.divider()
    st.subheader("📈 汇总统计")
    stats_table_area = st.empty() # 🌟 数据汇总表格的预留占位符

# ==============================================================================
# 第七部分：核心业务流水线 (点击按钮后的执行逻辑)
# ==============================================================================
def add_log(msg): # 日志推送工具
    ts = time.strftime("%H:%M:%S", time.localtime()); full_msg = f"[{ts}] {msg}"
    st.session_state.logs.append(full_msg)
    with log_box: # 在右侧监控区实时打印
        if "成功" in msg: st.success(full_msg)
        elif "失败" in msg: st.error(full_msg)
        else: st.write(full_msg)

if st.sidebar.button("🚀 开始分析"):
    if not api_key_list or not excel:
        st.error("请先完成 API Key 配置并上传 Excel 数据！")
    else:
        df = pd.read_excel(excel) # 解析 Excel
        st.session_state.iso_results = []; st.session_state.logs = []; st.session_state.current_key_idx = 0
        
        add_log("🚀 启动分析流水线...")
        
        for i, row in df.iterrows(): # 循环遍历每一个消防站
            name = row['station_name']
            prog_bar.progress((i + 1) / len(df)) # 更新进度条
            prog_txt.write(f"正在分析站点: {name} ({i+1}/{len(df)})")
            
            add_log(f"正在探测周边路网: {name}")
            # 运行爬虫引擎，获取轨迹点云
            pts, p_cnt, a_cnt, new_idx = run_cost_surface_engine(api_key_list, st.session_state.current_key_idx, row['lng'], row['lat'], t_limit, factor)
            st.session_state.current_key_idx = new_idx
            if new_idx >= len(api_key_list): add_log("🚨 警告：所有 Key 额度已全部耗尽！"); break
            
            # 运行空间算法，从点云中提炼出闭合边界
            poly = create_isoline_polygon(pts, (t_limit * 60 / factor), walk_speed)
            if poly:
                w_lng, w_lat = gcj02_to_wgs84(row['lng'], row['lat']) # 消防站自身坐标纠偏
                # 构造 GeoDataFrame 空间数据表
                gdf = gpd.GeoDataFrame({
                    '站点名称': [name], 'API消耗': [a_cnt], 'POI锚点数': [p_cnt], '设定时限': [t_limit]
                }, geometry=[poly], crs="EPSG:4326")
                # 面积精算：将球面经纬度投影到平面墨卡托坐标系，计算精确面积（平方公里）
                area = gdf.to_crs("EPSG:3857").area.iloc[0] / 10**6
                gdf['覆盖面积(km²)'] = round(area, 2); gdf['lat'] = w_lat; gdf['lng'] = w_lng
                
                st.session_state.iso_results.append(gdf) # 将单个站点结果存入汇总盒子
                add_log(f"✅ {name} 计算成功！面积: {area:.2f} km²")
            
            # 🌟 实时刷新右侧汇总表：每处理完一个站，表就更新一行
            if st.session_state.iso_results:
                live_df = pd.concat(st.session_state.iso_results, ignore_index=True)
                stats_table_area.dataframe(live_df[['站点名称', '覆盖面积(km²)', 'API消耗', 'POI锚点数']], height=400, use_container_width=True)

        st.session_state.map_renders += 1 # 分析全部结束后，更新重绘计数器
        st.balloons() # 撒花庆祝

# ==============================================================================
# 第八部分：右侧【看板汇总表】持久化逻辑 (计算完依旧在右侧显示)
# ==============================================================================
with col_monitor:
    # 只要 session_state 里有数据（不管是正在算还是算完了），就一直把汇总表钉在右侧显示
    if st.session_state.iso_results:
        summary_all_df = pd.concat(st.session_state.iso_results, ignore_index=True)
        stats_table_area.dataframe(
            summary_all_df[['站点名称', '覆盖面积(km²)', 'API消耗', 'POI锚点数']], 
            height=450, 
            use_container_width=True
        )

# ==============================================================================
# 第九部分：地图可视化与成果下载 (GIS 标准化导出)
# ==============================================================================
with col_map:
    # 动态 Key：通过给地图组件加一个变化的 ID，强制浏览器重绘 DOM，解决 removeChild 报错
    m_key = f"fire_map_stable_render_{st.session_state.map_renders}"
    m = folium.Map(location=[22.54, 114.05], zoom_start=12, tiles=map_style) # 默认显示深圳
    
    if st.session_state.iso_results:
        # 将地图中心对焦到第一个成功分析的站点
        m.location = [st.session_state.iso_results[0]['lat'].iloc[0], st.session_state.iso_results[0]['lng'].iloc[0]]
        for res in st.session_state.iso_results:
            # 在地图上渲染紫色的等时圈半透明面
            folium.GeoJson(res.to_json(), style_function=lambda x: {
                'fillColor': '#8E44AD', 'color': '#732D91', 'weight': 1.5, 'fillOpacity': 0.4
            }).add_to(m)
            # 在消防站位置扎一个蓝色标记点
            folium.Marker([res['lat'].iloc[0], res['lng'].iloc[0]], tooltip=res['站点名称'].iloc[0]).add_to(m)
    
    # 将 Folium 交互地图钉入网页
    st_folium(m, width="100%", height=750, key=m_key)

# --- 底部下载中心 ---
if st.session_state.iso_results:
    st.divider(); st.subheader("💾 消防评估成果一键导出")
    # 合并所有的空间结果
    full_gdf = gpd.GeoDataFrame(pd.concat(st.session_state.iso_results, ignore_index=True), crs="EPSG:4326")
    
    d_col1, d_col2 = st.columns(2)
    with d_col1: # 导出简单的 CSV 表格
        csv_bin = full_gdf.drop(columns='geometry').to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 1. 导出统计报表 (CSV)", data=csv_bin, file_name="消防覆盖范围分析结果.csv", use_container_width=True)
    with d_col2: # 导出专业的 GIS Shapefile 打包文件
        zip_mem = BytesIO() # 建立内存二进制流
        with tempfile.TemporaryDirectory() as tmp_d: # 建立临时沙盒文件夹
            shp_path = os.path.join(tmp_d, "fire_result.shp")
            # 重命名列名以符合 SHP 列名长度限制
            exp_gdf = full_gdf[['站点名称', '覆盖面积(km²)', 'API消耗', 'geometry']].copy()
            exp_gdf.columns = ['Name', 'Area_km2', 'API_Cnt', 'geometry']
            exp_gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
            # 压缩打包
            with zipfile.ZipFile(zip_mem, "w", zipfile.ZIP_DEFLATED) as zf:
                for r, _, fs in os.walk(tmp_d):
                    for f in fs: zf.write(os.path.join(r, f), arcname=f)
        st.download_button("📦 2. 导出shp图层 (SHP Zip)", data=zip_mem.getvalue(), file_name="消防图层包_SHP.zip", type="primary", use_container_width=True)
