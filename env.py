import numpy as np
class Env:
    def __init__(self):
        # 地块性质:[是否合种,作物类型,面积,上一季的作物编码]
        # 适合类型编码定义：0=粮食/豆类，1=蔬菜/食用菌，2=仅粮食
        self.plots=np.array([
            [0, 0, 20.0, -1],  # 地块1：非合种，适合0，20亩，上季未种
            [1, 1, 0.6, -1],   # 地块2：合种，适合1，0.6亩，上季未种
            [0, 2, 15.0, -1]   # 地块3：非合种，适合2，15亩，上季未种
        ],dtype=object)

        # 作物性质:[作物类型编码(0=粮食,1=豆类,2=蔬菜)，利润，适合的地块编码,销量]
        self.crops=np.array([
            [0, 400, [0, 2],15],   # 作物0（小麦）：粮食，适合地块类型1、3
            [1, 300, [0],20],      # 作物1（黄豆）：豆类，适合地块类型1
            [2, 600, [1],0.3]       # 作物2（番茄）：蔬菜，适合地块类型2
            [3, 700, [1],0.3]       # 作物3（香菇）：菌类，适合地块类型2
        ])

        # 时间：此处假设历经0~3共4个季度
        self.time=0

        # 约束条件：0-滞销，1-50%
        self.mode=0
    
    def set_mode(self,mode):
        self.mode=mode
        return self.mode

    def reset(self):
        self.plots[:,3]=-1
        self.time=0
        self.mode=0
        return self._get_state
    
    def _get_state(self):
        # 展平所有地块性质+当前时间
        state=self.plots.flatten().tolist()
        state.append(self.time)
        return state
    
    def step(self,action):
        # action:列表，长度=地块数，每个元素都是作物ID（-1=不种，仅合种地块可用）
        total_reward = 0
        penalty = 0

        for plot_idx in range(self.plots.shape[0]):
            # 提取地块性质（用多维数组索引，清晰直观）
            is_mixed = self.plots[plot_idx, 0]
            suit_type = self.plots[plot_idx, 1]
            area = self.plots[plot_idx, 2]
            last_crop = self.plots[plot_idx, 3]
            crop_id = action[plot_idx]

            # 约束1：非合种地块不能不种
            if not is_mixed and crop_id == -1:
                penalty -= 100

            # 约束2：作物必须适合地块（利用重叠的适合类型编码）
            if crop_id != -1:
                crop_suit_list = self.crops[crop_id, 2]  # 作物适合的地块类型列表
                if suit_type not in crop_suit_list:
                    penalty -= 150

            # 约束3：重茬检查
            if crop_id != -1 and last_crop == crop_id:
                penalty -= 80

            # 计算利润
            if crop_id != -1:
                profit_per_acre = self.crops[crop_id, 1]
                plant_area = area * 0.3 if is_mixed else area  # 合种取30%面积
                max_area = self.crops[crop_id,3]
                total_reward += np.minimum(plant_area,max_area) * profit_per_acre
                if self.mode==1 and plant_area>max_area:
                    total_reward += (max_area-plant_area) * profit_per_acre

            # 更新上季作物记录
            self.plots[plot_idx, 3] = crop_id if crop_id != -1 else -1

        # 总奖励 = 利润 + 惩罚
        total_reward += penalty

        # 更新时间
        self.time += 1
        done = (self.time == 4)

        return self._get_state(), total_reward, done, {}