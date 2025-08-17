import pulp

# --------------------------
# 1. 数据输入（根据实际情况补充）
# --------------------------
# 地块信息：新增"is_mixed"标记是否为合种大棚
plots = {
    1: {"type": "平旱地", "area": 30, "is_mixed": False},  # 非合种：必须种满
    2: {"type": "水浇地", "area": 20, "is_mixed": False},  # 非合种：必须种满
    3: {"type": "普通大棚", "area": 0.6, "is_mixed": True},  # 合种：占比0.2~0.4
    4: {"type": "智慧大棚", "area": 0.6, "is_mixed": True},  # 合种：占比0.2~0.4
    # ... 补充其他地块
}

# 作物信息（同上，简化示例）
crops = {
    1: {"type": "粮食", "suitable": ["平旱地", "水浇地"], 
        "yield": 500, "cost": 200, "price": 2.5, "demand": 10000},
    2: {"type": "豆类", "suitable": ["平旱地", "普通大棚"], 
        "yield": 300, "cost": 150, "price": 4, "demand": 5000},
    # ... 补充其他作物
}

# 时间参数
years = range(2024, 2031)  # 7年
seasons_per_year = {  # 不同地块的年种植季数
    "平旱地": 1, "水浇地": 2, "普通大棚": 2, "智慧大棚": 2
}

# --------------------------
# 2. 创建模型（目标：最大化利润）
# --------------------------
model = pulp.LpProblem("Mixed_Planting_Optimization", pulp.LpMaximize)

# --------------------------
# 3. 定义决策变量
# --------------------------
# x[地块ID][年份][季节][作物ID] = 种植面积（亩）
x = pulp.LpVariable.dicts(
    "PlantArea",
    (plots.keys(), years, range(1, max(seasons_per_year.values())+1), crops.keys()),
    lowBound=0,  # 面积不能为负
    cat=pulp.LpContinuous
)

# --------------------------
# 4. 目标函数（利润=收入-成本）
# --------------------------
scenario = 1  # 1=超产滞销；2=超产半价
total_profit = 0

for year in years:
    for crop_id in crops:
        # 计算总产（所有适配地块的产量之和）
        total_yield = pulp.lpSum(
            x[p_id][year][s][crop_id] * crops[crop_id]["yield"]
            for p_id in plots
            for s in range(1, seasons_per_year[plots[p_id]["type"]]+1)
            if plots[p_id]["type"] in crops[crop_id]["suitable"]
        )
        
        # 计算销售收入（分情景）
        demand = crops[crop_id]["demand"]
        price = crops[crop_id]["price"]
        if scenario == 1:
            revenue = pulp.lpMin(total_yield, demand) * price  # 超产部分不计收入
        else:
            excess = pulp.lpMax(0, total_yield - demand)
            revenue = demand * price + excess * price * 0.5  # 超产半价
        
        # 计算成本（所有种植面积的总成本）
        cost = pulp.lpSum(
            x[p_id][year][s][crop_id] * crops[crop_id]["cost"]
            for p_id in plots
            for s in range(1, seasons_per_year[plots[p_id]["type"]]+1)
        )
        
        total_profit += (revenue - cost)

model += total_profit  # 最大化总利润

# --------------------------
# 5. 核心约束（按你的思路调整）
# --------------------------

# 约束1：非合种地块必须种满（总面积=地块面积）
for p_id in plots:
    if not plots[p_id]["is_mixed"]:  # 只处理非合种地块
        p_area = plots[p_id]["area"]
        p_type = plots[p_id]["type"]
        for year in years:
            for s in range(1, seasons_per_year[p_type]+1):
                # 该地块该季的总种植面积 = 地块面积（必须种满）
                model += pulp.lpSum(x[p_id][year][s][crop_id] for crop_id in crops) == p_area, \
                    f"FullPlant_{p_id}_{year}_{s}"

# 约束2：合种大棚种植面积占比0.2~0.4
for p_id in plots:
    if plots[p_id]["is_mixed"]:  # 只处理合种地块
        p_area = plots[p_id]["area"]
        p_type = plots[p_id]["type"]
        min_area = 0.2 * p_area  # 最小种植面积（20%）
        max_area = 0.4 * p_area  # 最大种植面积（40%）
        for year in years:
            for s in range(1, seasons_per_year[p_type]+1):
                # 该地块该季的总种植面积 ≥ 20%地块面积
                model += pulp.lpSum(x[p_id][year][s][crop_id] for crop_id in crops) >= min_area, \
                    f"MixedMin_{p_id}_{year}_{s}"
                # 该地块该季的总种植面积 ≤ 40%地块面积
                model += pulp.lpSum(x[p_id][year][s][crop_id] for crop_id in crops) <= max_area, \
                    f"MixedMax_{p_id}_{year}_{s}"

# 约束3：作物-地块适配性（非适配地块不能种）
for p_id in plots:
    p_type = plots[p_id]["type"]
    unsuitable = [cid for cid in crops if p_type not in crops[cid]["suitable"]]
    for year in years:
        for s in range(1, seasons_per_year[p_type]+1):
            for crop_id in unsuitable:
                model += x[p_id][year][s][crop_id] == 0, \
                    f"Suitable_{p_id}_{year}_{s}_{crop_id}"

# 约束4：重茬限制（相邻季节不能种同一种作物）
for p_id in plots:
    p_type = plots[p_id]["type"]
    n_seasons = seasons_per_year[p_type]
    for year in years:
        for s in range(1, n_seasons):  # 相邻季节（如1和2季）
            for crop_id in crops:
                # 同一作物不能在相邻季节种植（面积和≤地块面积）
                model += x[p_id][year][s][crop_id] + x[p_id][year][s+1][crop_id] <= plots[p_id]["area"], \
                    f"NoReplant_{p_id}_{year}_{s}_{crop_id}"

# 约束5：豆类种植要求（每3年至少种1次）
for p_id in plots:
    beans = [cid for cid in crops if crops[cid]["type"] == "豆类"]
    for start in range(2024, 2028):  # 3年周期
        end = start + 2
        model += pulp.lpSum(
            x[p_id][y][s][cid] for y in range(start, end+1)
            for s in range(1, seasons_per_year[plots[p_id]["type"]]+1)
            for cid in beans
        ) >= 0.1,  # 至少种0.1亩
        f"BeanReq_{p_id}_{start}_{end}"

# --------------------------
# 6. 求解与输出
# --------------------------
# 用CBC求解器（PuLP自带）
status = model.solve(pulp.PULP_CBC_CMD(msg=0))
print(f"求解状态：{pulp.LpStatus[status]}")  # 输出"Optimal"表示找到最优解
print(f"7年最大总利润：{pulp.value(model.objective):.2f}元")

# 输出前1个地块2024年的种植方案（示例）
print("\n2024年地块1的种植方案：")
p_id = 1
year = 2024
for s in range(1, seasons_per_year[plots[p_id]["type"]]+1):
    print(f"第{s}季：")
    for crop_id in crops:
        area = x[p_id][year][s][crop_id].varValue  # 获取求解结果
        if area > 0.001:  # 只显示种植面积>0的作物
            print(f"  作物{crop_id}：{area:.2f}亩")