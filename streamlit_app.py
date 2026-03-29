# -*- coding: utf-8 -*-
"""
消防站点 5 分钟等时圈分析工具 v2.0
优化项：增加时间戳，增加坐标系选择功能、非道路距离惩罚、消防特权建模、双策略路径采样、车辆掉头特权系数
"""
# ==============================================================================
# 第一部分：导入系统所需的“核心工具组件”
# ==============================================================================
import streamlit as st
import pandas as pd
import geopandas as gpd
import requests
import math
import os
import folium
from streamlit_folium import st_folium
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from shapely.geometry import Polygon
from shapely.ops import unary_union
import tempfile
import zipfile
from io import BytesIO
from datetime import datetime
import pytz


# ==============================================================================
# 第二部分：坐标纠偏算法 (新增百度与火星坐标互转)
# ==============================================================================
def bd09_to_gcj02(bd_lon, bd_lat):
    """
    百度坐标系 (BD-09) -> 火星坐标系 (GCJ-02)
    百度地图使用的坐标系在火星坐标的基础上二次加密，需要先解密才能给高德使用。
    """
    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * x_pi)
    gcj_lng = z * math.cos(theta)
    gcj_lat = z * math.sin(theta)
    return gcj_lng, gcj_lat


def gcj02_to_wgs84(lng, lat):
    """
    火星坐标系 (GCJ-02) -> 地球坐标系 (WGS-84)
    将国内加密坐标还原为国际通用的标准 GPS 坐标，用于 GIS 导出和地图纠偏。
    """
    a = 6378137.0
    ee = 0.00669342162296594323
    pi = 3.1415926535897932384626

    def _transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y * pi / 12.0) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
        return ret

    def _transform_lng(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0
        return ret

    dlat = _transform_lat(lng - 105.0, lat - 35.0)
    dlng = _transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    return lng - dlng, lat - dlat


# ==============================================================================
# 第三部分：辅助工具函数 (请求控制、时间标签、坐标统一转换)
# ==============================================================================
def smart_amap_request(url, params, api_keys, current_key_idx):
    while current_key_idx < len(api_keys):
        params['key'] = api_keys[current_key_idx]
        try:
            r = requests.get(url, params=params, timeout=5).json()
            if r.get('status') == '0' and r.get('infocode') in ['10003', '10044', '10012']:
                current_key_idx += 1
                continue
            return r, current_key_idx, True
        except:
            return None, current_key_idx, True
    return None, current_key_idx, False


def get_time_tag():
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.now(beijing_tz)
    hour = now.hour
    time_str = now.strftime("%Y-%m-%d %H:%M")
    if 7 <= hour < 9:
        tag = "早高峰"
    elif 17 <= hour < 19:
        tag = "晚高峰"
    elif hour >= 23 or hour < 6:
        tag = "夜间低谷"
    else:
        tag = "平峰期"
    return f"{time_str} [{tag}]"


# ==============================================================================
# 第四部分：路网实时爬虫引擎 (重构版：解决掉头限制与探测盲区 + 消防特权建模)
# ==============================================================================
def run_cost_surface_engine(api_keys, key_idx, origin_lng, origin_lat, target_min, factor=0.8):
    """
    逻辑说明：
    1. 生成多起点：在车站东西南北50米设分身，模拟消防车出库即可跨线。
    2. 强制网格：即使周边没商店(POI)，也强制生成探测点，摸清每一条小路的尽头。
    3. 双策略采样与掉头折减：模拟消防车特权通行（新引入）。
    """
    # 自动计算搜索半径（米）：大约是目标分钟数对应的行车距离
    radius = min(int(target_min * 800 * 1.5), 15000)
    anchors = []  # 目的地篮子
    api_calls = 0  # API 记账

    # --- 核心优化 1：设置虚拟出口 (解决掉头问题) ---
    offset = 0.0005
    v_origins = [
        {"name": "中心", "lng": origin_lng, "lat": origin_lat},
        {"name": "东出口", "lng": origin_lng + offset, "lat": origin_lat},
        {"name": "西出口", "lng": origin_lng - offset, "lat": origin_lat},
        {"name": "北出口", "lng": origin_lng, "lat": origin_lat + offset},
        {"name": "南出口", "lng": origin_lng, "lat": origin_lat - offset}
    ]

    # --- 核心优化 2：全方向强制探测 (保持原逻辑不变) ---
    url_around = "https://restapi.amap.com/v3/place/around"
    all_types = "190301|150700|190000|170000|120000|140000|090000|060000"
    for page in range(1, 10):
        p_params = {"location": f"{origin_lng:.6f},{origin_lat:.6f}", "radius": radius, "types": all_types,
                    "offset": 50, "page": page, "key": api_keys[key_idx]}
        r, key_idx, is_called = smart_amap_request(url_around, p_params, api_keys, key_idx)
        if is_called: api_calls += 1
        if r and r.get('status') == '1' and r.get('pois'):
            anchors.extend([poi['location'] for poi in r['pois']])
        else:
            break

    for angle in range(0, 360, 45):
        for dist_step in [0.5, 1.0, 1.3]:
            rad = math.radians(angle)
            g_lng = origin_lng + (radius * dist_step * math.cos(rad)) / (111320 * math.cos(math.radians(origin_lat)))
            g_lat = origin_lat + (radius * dist_step * math.sin(rad)) / 111320
            anchors.append(f"{g_lng:.6f},{g_lat:.6f}")

    anchors = list(set(anchors))
    trail_points = []

    # --- 核心优化 3：最优出口匹配 + 消防特权建模 ---
    url_route = "https://restapi.amap.com/v3/direction/driving"  #高德地图API，路径规划

    # 新增：双策略轮询池 (13:速度优先-避堵走大路, 17:距离优先-抄近道穿小巷)
    strategies = [13, 17]

    for i, dest in enumerate(anchors):
        d_lng, d_lat = map(float, dest.split(','))
        best_o = min(v_origins, key=lambda o: math.sqrt((o['lng'] - d_lng) ** 2 + (o['lat'] - d_lat) ** 2))

        # 新增：双策略交替采样
        current_strategy = strategies[i % len(strategies)]

        # 替换原有的固定 strategy: 13
        params = {"origin": f"{best_o['lng']:.6f},{best_o['lat']:.6f}", "destination": dest,
                  "strategy": current_strategy,
                  "key": api_keys[key_idx]}
        r, key_idx, is_called = smart_amap_request(url_route, params, api_keys, key_idx)
        if is_called: api_calls += 1

        if r and r.get('status') == '1' and r.get('route'):
            steps = r['route']['paths'][0]['steps']
            acc_t = 0
            for s in steps:
                dur = int(s['duration'])

                # 新增：消防特权掉头代价折减
                # 如果导航动作包含掉头，将其耗时强行削减（如保留15%的时间）
                instruction = s.get('instruction', '')
                action = s.get('action', '')
                if '掉头' in instruction or action == '掉头':
                    dur = int(dur * 0.15)

                polyline = s['polyline'].split(';')
                t_step = dur / max(1, len(polyline) - 1)

                # 修复一个小细节：将内部循环变量改为 j，防止覆盖外部的 enumerate(anchors) 的 i
                for j, p in enumerate(polyline):
                    plng, plat = map(float, p.split(','))
                    w_lng, w_lat = gcj02_to_wgs84(plng, plat)
                    trail_points.append((w_lng, w_lat, acc_t + j * t_step))
                acc_t += dur
                if acc_t > (target_min * 60 / factor) + 60: break

    return trail_points, len(anchors), api_calls, key_idx


# ==============================================================================
# 第五部分：空间时间场算法 (重构版：引入步行截断 + 距离惩罚函数)
# ==============================================================================
def create_isoline_polygon(trail_points, target_sec, off_road_speed, max_walk_dist=300):
    """
    逻辑说明：
    1. 物理距离换算：计算每个地块中心点离最近马路的真实米数。
    2. 距离惩罚函数：离马路越远，步速受到指数级阻力，收缩假性覆盖面积。
    3. 深度截断：如果离马路超过 max_walk_dist (默认300米)，判定为不可达。
    """
    if len(trail_points) < 10: return None

    # 1. 整理数据并进行地球曲率缩放
    pts = np.array(trail_points)
    xy = pts[:, 0:2]
    times = pts[:, 2]
    avg_lat = np.mean(xy[:, 1])
    cos_lat = np.cos(np.radians(avg_lat))
    xy_scaled = xy.copy()
    xy_scaled[:, 0] *= cos_lat

    # 2. 建立空间快速索引 (cKDTree)
    tree = cKDTree(xy_scaled)

    # 3. 划分 300x300 的计算网格
    pad = 0.01
    min_x, max_x = np.min(xy[:, 0]) - pad, np.max(xy[:, 0]) + pad
    min_y, max_y = np.min(xy[:, 1]) - pad, np.max(xy[:, 1]) + pad
    grid_res = 300
    x_g = np.linspace(min_x, max_x, grid_res)
    y_g = np.linspace(min_y, max_y, grid_res)
    X, Y = np.meshgrid(x_g, y_g)
    grid_xy = np.column_stack([X.ravel() * cos_lat, Y.ravel()])

    # 4. 计算每个格子的时间逻辑
    dists, indices = tree.query(grid_xy)

    # 将“角度距离”换算成“物理米数”
    physical_meters = dists * 111320

    # --- 新增：距离惩罚函数 (Distance Penalty Function) ---
    # 基础理想步行时间
    base_walk_time = physical_meters / off_road_speed

    # 惩罚建模：假设前100米属于临街/正常深入范畴，阻力正常。
    # 超过100米后，受建筑、围墙、绿化阻挡，阻力呈平方级剧增。
    # 矩阵运算保证性能，不使用慢速的 for 循环。
    penalty_factor = 1.0 + np.maximum(0, physical_meters - 100) / 60.0
    actual_off_road_time = base_walk_time * (penalty_factor ** 2)

    # 总时间 = 开车到路边的时间 + 开车进入地块的时间
    grid_t = times[indices] + actual_off_road_time
    # --- 核心优化：距离截断逻辑 (维持原逻辑，作为双重保险) ---
    grid_t[physical_meters > max_walk_dist] = target_sec * 5
    # 5. 提取边界线并生成多边形
    grid_t = grid_t.reshape((grid_res, grid_res))
    contours = measure.find_contours(grid_t, level=target_sec)
    polys = []
    for c in contours:
        c_lng = min_x + (c[:, 1] / (grid_res - 1)) * (max_x - min_x)
        c_lat = min_y + (c[:, 0] / (grid_res - 1)) * (max_y - min_y)
        if len(c_lng) >= 3: polys.append(Polygon(np.column_stack([c_lng, c_lat])))

    # 合并多边形，修复细小缝隙
    return unary_union(polys).buffer(0.0001).buffer(-0.0001) if polys else None
# ==============================================================================
# 第六部分：网页 UI 布局
# ==============================================================================
st.set_page_config(page_title="", layout="wide")
st.markdown("<h1 style='text-align: center; color: #E74C3C;'>🚒 城市消防站可达性评估系统 v2.0</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #7F8C8D;'>支持多源坐标纠偏 (高德/百度) | 实时路况模拟 | GIS 成果导出(WGS84坐标)" "本版本优化项：调整时间戳为北京时间，增加坐标系选择功能、非道路距离惩罚、消防特权建模、双策略路径采样、车辆掉头特权系数</p>",
    unsafe_allow_html=True)
st.divider()

if 'iso_results' not in st.session_state: st.session_state.iso_results = []
if 'current_key_idx' not in st.session_state: st.session_state.current_key_idx = 0
if 'map_renders' not in st.session_state: st.session_state.map_renders = 0
if 'logs' not in st.session_state: st.session_state.logs = []

st.sidebar.header("⚙️ 核心参数配置")
api_keys_input = st.sidebar.text_area("1. 输入高德地图 API Keys (多 Key 逗号分隔)")
api_key_list = [k.strip() for k in api_keys_input.split(',') if k.strip()]
excel = st.sidebar.file_uploader("2. 上传消防站表格 (station_name, lng, lat)", type=["xlsx"])
# 🌟 新增坐标系选择按钮
coord_system = st.sidebar.radio(
    "3. 请选择上传数据的坐标系",
    ("高德坐标 (GCJ-02)", "百度坐标 (BD-09)"),
    help="明确一下消防站坐标系统（百度还是高德），选错会导致分析结果偏离实际位置 300-500 米"
)
t_limit = st.sidebar.slider("4. 到场时间要求 (分钟)", 3, 15, 5)
factor = st.sidebar.slider("5. 消防车特权通行系数", 0.7, 1.0, 0.8)
walk_speed = st.sidebar.slider("6. 小区内部消防车速度 (m/s)", 1.0, 5.0, 4.0, 0.1)
map_style = st.sidebar.selectbox("7. 地图风格", ("CartoDB positron", "OpenStreetMap", "CartoDB dark_matter"))

record_timestamp = st.sidebar.checkbox("8. 📌 自动记录路况时段标签", value=True)

col_map, col_monitor = st.columns([3, 1])

with col_monitor:
    st.subheader("📊 实时监控")
    prog_bar = st.empty()
    prog_txt = st.empty()
    st.divider()
    st.subheader("📜 运行日志")
    log_box = st.container(height=250, border=True)
    st.divider()
    st.subheader("📈 数据统计")
    stats_table_area = st.empty()  # 预留一个空位，等下用来展示最终的成绩单(数据表)


def add_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{ts}] {msg}"
    st.session_state.logs.append(full_msg)
    with log_box:
        if "成功" in msg:
            st.success(full_msg)
        elif "失败" in msg or "警告" in msg:
            st.error(full_msg)
        else:
            st.write(full_msg)


# ==============================================================================
# 第七部分：核心业务流
# ==============================================================================
if st.sidebar.button("🚀 开始分析"):
    if not api_key_list or not excel:
        st.error("请先完成高德地图 API Key 配置并上传 Excel 数据！")
    else:
        df = pd.read_excel(excel)
        st.session_state.iso_results = []
        st.session_state.logs = []
        st.session_state.current_key_idx = 0
        current_time_tag = get_time_tag() if record_timestamp else "未开启记录"
        add_log(f"🚀 启动分析流程... 输入坐标系: {coord_system}")

        for i, row in df.iterrows():
            name = row['station_name']
            prog_bar.progress((i + 1) / len(df))  # 推进绿色的进度条
            raw_lng, raw_lat = row['lng'], row['lat']
            prog_bar.progress((i + 1) / len(df))
            prog_txt.write(f"正在分析站点: {name} ({i + 1}/{len(df)})")

            # 🌟 坐标前置转换：喂给高德 API 的必须是 GCJ-02
            if coord_system == "百度坐标 (BD-09)":
                api_lng, api_lat = bd09_to_gcj02(raw_lng, raw_lat)
            else:
                api_lng, api_lat = raw_lng, raw_lat

            add_log(f"正在探测路网: {name}")
            pts, p_cnt, a_cnt, new_idx = run_cost_surface_engine(api_key_list, st.session_state.current_key_idx,
                                                                 api_lng, api_lat, t_limit, factor)
            st.session_state.current_key_idx = new_idx

            if new_idx >= len(api_key_list): add_log("🚨 报警：所有 Key 已耗尽！"); break

            poly = create_isoline_polygon(pts, (t_limit * 60 / factor), walk_speed)
            if poly:
                # 🌟 统一转换为 WGS-84 用于最终展示
                w_lng, w_lat = gcj02_to_wgs84(api_lng, api_lat)
                gdf = gpd.GeoDataFrame({
                    '站点名称': [name], 'API消耗': [a_cnt], 'POI锚点数': [p_cnt], '测算时刻': [current_time_tag]
                }, geometry=[poly], crs="EPSG:4326")
                area = gdf.to_crs("EPSG:3857").area.iloc[0] / 10 ** 6
                gdf['覆盖面积(km²)'] = round(area, 2)
                gdf['lat'] = w_lat
                gdf['lng'] = w_lng
                st.session_state.iso_results.append(gdf)
                add_log(f"✅ {name} 成功！面积: {area:.2f} km²")
            else:
                add_log(f"❌ {name} 失败！路网密度不足。")
            if st.session_state.iso_results:
                live_df = pd.concat(st.session_state.iso_results, ignore_index=True)
                stats_table_area.dataframe(live_df[['站点名称', '覆盖面积(km²)', 'API消耗','POI锚点数', '测算时刻']],
                                           height=400, width='stretch')

        st.session_state.map_renders += 1
        st.balloons()

with col_map:
    m_key = f"fire_map_{st.session_state.map_renders}"
    m = folium.Map(location=[22.54, 114.05], zoom_start=12, tiles=map_style)
    if st.session_state.iso_results:
        m.location = [st.session_state.iso_results[0]['lat'].iloc[0], st.session_state.iso_results[0]['lng'].iloc[0]]
        for res in st.session_state.iso_results:
            folium.GeoJson(res.to_json(),
                           style_function=lambda x: {'fillColor': '#8E44AD', 'color': '#732D91', 'weight': 1.5,
                                                     'fillOpacity': 0.4}).add_to(m)
            folium.Marker([res['lat'].iloc[0], res['lng'].iloc[0]], tooltip=res['站点名称'].iloc[0]).add_to(m)
    st_folium(m, width="100%", height=750, key=m_key)

if st.session_state.iso_results:
    st.divider()
    st.subheader("💾 成果导出")
    full_gdf = gpd.GeoDataFrame(pd.concat(st.session_state.iso_results, ignore_index=True), crs="EPSG:4326")
    # ==============================================================================
    # 🌟 在这里插入【数据清洗】
    # 目的 A：解决 Arrow 序列化警告。将非几何列统一转为标准类型，防止混合类型。
    for col in full_gdf.columns:
        if col != 'geometry':
            # 将 object 类型（混合类型）强制转为字符串，确保前端展示不报错
            if full_gdf[col].dtype == object:
                full_gdf[col] = full_gdf[col].astype(str)
    # 目的 B：防御性补全。确保刚才报错的“POI点数”等关键列如果缺失则补0
    required_cols = ['站点名称', '覆盖面积(km²)', 'API消耗', 'POI点数', '测算时刻','geometry']
    for c in required_cols:
        if c not in full_gdf.columns:
            full_gdf[c] = 0
    # ==============================================================================
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        csv_bin = full_gdf.drop(columns='geometry').to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 导出统计报表 (CSV)", data=csv_bin, file_name="分析.csv", width='stretch')
    with d_col2:  # 专业 GIS 规划师用的 Shapefile 打包下载
        zip_mem = BytesIO()  # 虚拟一个内存空间，速度比写在硬盘上快一万倍
        with tempfile.TemporaryDirectory() as tmp_d:  # 召唤一个隐形的临时文件夹
            shp_path = os.path.join(tmp_d, "fire_result.shp")
            # 🌟 避坑指南：Shapefile 的古老规矩，每一列的名字不能超过 10 个英文字符！
            # 如果不把中文改成短英文，程序打包时会当场崩溃给你看。
            # 1. 确保筛选时包含了所有 6 个字段
            exp_gdf = full_gdf[['站点名称', '覆盖面积(km²)', 'API消耗', 'POI锚点数', '测算时刻','geometry']].copy()
            # 2. 确保改名列表也是 6 个（顺序必须严格对应）
            # 原名: ['站点名称', '覆盖面积(km²)', 'API消耗', 'POI点数', '测算时刻', 'geometry']
            exp_gdf.columns = ['Name', 'Area_km2', 'API_Cnt', 'POI_Cnt', 'Time_Tag','geometry']
            # 存入那个隐形的文件夹里，自动生成 .shp, .dbf, .shx, .prj 四个兄弟文件
            exp_gdf.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')

            # 拿出打包器，把这四个兄弟全部塞进 ZIP 压缩包里
            with zipfile.ZipFile(zip_mem, "w", zipfile.ZIP_DEFLATED) as zf:
                for r, _, fs in os.walk(tmp_d):
                    for f in fs: zf.write(os.path.join(r, f), arcname=f)

        # 提供一个华丽的下载按钮，一键把压缩包交到用户手上
        st.download_button("📦 2. 导出专业地图图层...", data=zip_mem.getvalue(), file_name="图层（wgs84）.zip", width='stretch')
# ==============================================================================
# 第九部分：生成的表格可视化与导出逻辑 (保持稳定)
# ==============================================================================
with col_monitor:
    # 只要成绩箱里有东西，不管你在左边地图怎么乱点，右边这个表始终死死钉在墙上给你看
    if st.session_state.iso_results:
        summary_all_df = pd.concat(st.session_state.iso_results, ignore_index=True)
        stats_table_area.dataframe(
            summary_all_df[['站点名称', '覆盖面积(km²)', 'API消耗', 'POI锚点数', '测算时刻']],
            height=450,
            width='stretch'
        )
