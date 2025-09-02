import pandas as pd
import numpy as np

class data():
    def __init__(self):
        '''====================数据读取================'''
        file_path_1=r'./2024CUMCM-C/data/附件1.xlsx'
        farmland_data=pd.read_excel(file_path_1,sheet_name='乡村的现有耕地')
        crop_data = pd.read_excel(file_path_1, sheet_name='乡村种植的农作物')
        # 手动处理23年数据集，2D
        yield_2023_data=pd.read_excel('./2024CUMCM-C/data/data.xlsx',sheet_name='亩产量',usecols='B:AP',skiprows=0)
        '''====================数据处理================'''
        # 目标：地块+季度-作物-年份
    
        # 地块＋季度
        self.plots_area = farmland_data['地块面积/亩'].tolist()
        self.plots_with_season_area = self.plots_area + self.plots_area[26:]
        self.plots_with_season_area = np.array(self.plots_with_season_area)

        # 初始化一个新的列表，按照先第一季度再第二季度的顺序
        plots_with_season = []
        # 获取唯一地块名称
        plots= farmland_data['地块名称'].tolist()
        # 先处理所有地块的第一季度
        for plot in plots:
            plots_with_season.append(f"{plot}_1季")
        # 再处理只有部分地块有第二季度的情况
        second_season_plots = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
                               "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14",
                               "E15", "E16",
                               "F1", "F2", "F3", "F4"]
        for plot in second_season_plots:
            plots_with_season.append(f"{plot}_2季")
        self.plots_with_season = np.array(plots_with_season)

        # 作物
        crops = crop_data['作物名称'].dropna()  # 删除 NaN
        crops = crops[~crops.str.contains('\(|（')]  # 删除包含说明的行
        self.crops = crops.unique() # 已经是np.array了

        # 年份
        self.years = np.array(range(2023, 2031))

        # 亩产量
        self.yield_2023 = yield_2023_data.to_numpy()

        '''====================矩阵生成================'''
        '''====================手动处理23年数据集================'''

        self.price_matrix = self.create_matrix() # 单价
        self.yield_matrix = self.create_matrix() # 亩产量
        self.cost_matrix = self.create_matrix() # 成本
        self.sale_matrix = np.zeros(len(self.crops),len(self.years)) # 销售量
        # self.plant_matrix = self.create_matrix()
        # 亩产量
        for i in range(len(self.years)):
            self.yield_matrix[:, :, i] = self.yield_2023
    def create_matrix(self):
        """
        创建一个三维零矩阵，用于存储不同地块、不同作物、不同年份的数据
        返回:
            numpy.ndarray: 一个三维零矩阵，维度为(地块数量, 作物种类数量, 年份数量)
        """
        return np.zeros((len(self.plots_with_season), len(self.crops), len(self.years)))
    
    def get_item(self, matrix, plot_with_season_name, crop_name, year_name):
        # 如果需要查找特定地块、作物和年份的数据
        # 使用 np.where 查找索引
        plot_idx = np.where(self.plots_with_season == plot_with_season_name)  # 查找 'A1' 在 plots 中的索引
        crop_idx = np.where(self.crops == crop_name)  # 查找 '黄豆' 在 crops 中的索引
        year_idx = np.where(self.years == year_name)  # 查找 2024 在 years 中的索引
        print(matrix[plot_idx, crop_idx, year_idx])
        return matrix[plot_idx, crop_idx, year_idx]

if __name__ == '__main__':
    data = data()