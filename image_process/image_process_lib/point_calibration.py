from scipy.optimize import linear_sum_assignment
import numpy as np
import cv2
from image_process_lib.board_detect import board_detect
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from camera.srv import pixel2world, pixel2worldRequest

class CoordinatePredictor:
    def __init__(self, degree=2):
        """
        初始化Z坐标预测器
        :param degree: 多项式回归的阶数（默认2阶）
        """
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
    
    def fit(self, pixel_coords, z_coords):
        """
        训练预测模型
        :param pixel_coords: 9个像素坐标的数组，形状为(9, 2)
        :param z_coords: 对应的9个Z坐标值，形状为(9,)
        """
        # 生成多项式特征
        X_poly = self.poly.fit_transform(pixel_coords)
        # 训练线性回归模型
        self.model.fit(X_poly, z_coords)
    
    def predict(self, pixel_coord):
        """
        预测目标点的Z坐标
        :param pixel_coord: 目标点的像素坐标，形状为(2,)
        :return: 预测的Z坐标
        """
        # 转换为多项式特征
        point_poly = self.poly.transform([pixel_coord])
        # 预测Z坐标
        return self.model.predict(point_poly)[0]
    

# 方块9点标定数据
pixel_coords = np.array(
    [[151,72], [642,80], [1117,72], 
     [157,357], [631,352], [1119,305],
     [160,652], [633,660], [1142,608],
])

x_coords = [-486.4594116210937, -480.4456176757812, -480.448272705078,
            -350.42529296875, -350.4407348632812, -370.4565124511718,
            -208.415054321289, -203.4114990234375, -227.3964691162109
            ]

y_coords = [-229.5760345458984, 7.131750106811523, 231.3762969970703,
            -227.9130554199218, 2.140347480773926, 232.1771087646484,
            -227.8733520507812, 2.132221221923828, 242.1852874755859
            ]

z_coords = [179.1824340820312, 181.31640625, 183.3018646240234,
            179.103515625, 181.3115234375, 182.3131866455078,
            178.2863159179687, 181.2923583984375, 182.09423828125
            ]

# 托盘9点标定数据
pixel_board = np.array(
    [[475,83], [470,623], 
     [847,86], [842,623],
     [589,374],[668,552],
     [590,174],[588,547]
])

x_board = [-481.0155334472656, -220.7591247558593, 
            -477.1913146972656, -219.8908081054687,
            -340.7634887695312,-254.8663635253906,
            -436.882568359375,-257.144775390625
            ]

y_board = [-73.38421630859375, -76.9220199584961,
            105.7991714477539, 102.481575012207,
            -18.5580940246582,19.6627616882324,
            -18.18140411376953,-19.36852645874023
            ]


# 2. 创建并训练预测模型
#方块预测
predictor_x = CoordinatePredictor(degree=2)
predictor_x.fit(pixel_coords, x_coords)
predictor_y = CoordinatePredictor(degree=2)
predictor_y.fit(pixel_coords, y_coords)
predictor_z = CoordinatePredictor(degree=2)
predictor_z.fit(pixel_coords, z_coords)

#托盘预测
board_x = CoordinatePredictor(degree=2)
board_x.fit(pixel_board, x_board)
board_y = CoordinatePredictor(degree=2)
board_y.fit(pixel_board, y_board)

def x_predict(px,py):
    return predictor_x.predict(np.array([px,py]))
def y_predict(px,py):
    return predictor_y.predict(np.array([px,py]))
def z_predict(px,py):
    return predictor_z.predict(np.array([px,py]))


class ArmCalibrator:
    def __init__(self):
        self.H = None  # 变换矩阵
        self.affine_matrix = None  # 仿射变换矩阵，用于齐次变换异常时兜底
        self.board_theta = 0
        self.src_points = [(1, 14),(1, 1),(10, 14),(10, 1)]
        self.dst_points = []
        
    def calibrate(self): #眼
        """
        通过4个对应点计算变换矩阵
        :param src_points: 机械臂坐标系中的4个点 [左上, 左下, 右上, 右下]
        :param dst_points: 实际坐标系中的对应4个点 [左上, 左下, 右上, 右下]
        """
        if len(self.src_points) != 4 or len(self.dst_points) != 4:
            raise ValueError("需要提供4个源点和4个目标点")
            
        src = np.array(self.src_points, dtype=np.float32)
        dst = np.array(self.dst_points, dtype=np.float32)
        if not np.all(np.isfinite(dst)):
            raise ValueError("托盘标定点包含无效数值，请检查托盘识别和像素到机械臂坐标预测")

        affine_matrix, _ = cv2.estimateAffine2D(src, dst)
        if affine_matrix is None or not np.all(np.isfinite(affine_matrix)):
            raise ValueError("托盘仿射标定失败，请检查4个角点是否识别正确")
        self.affine_matrix = affine_matrix.astype(np.float32)
        
        # 构建线性方程组 A * h = b
        A = []
        b = []
        for i in range(4):
            x, y = src[i]
            u, v = dst[i]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
            b.extend([u, v])
        
        A = np.array(A, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        
        # 解线性方程组
        try:
            h = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 使用最小二乘法作为备选方案
            h = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # 重构变换矩阵 (3x3)
        self.H = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1]
        ], dtype=np.float32)
        if not np.all(np.isfinite(self.H)):
            raise ValueError("托盘透视标定矩阵包含无效数值，请重新识别托盘")
    
    def transform(self, point):
        """
        将机械臂坐标系的点转换到实际坐标系
        :param point: 机械臂坐标系中的点 (x, y)
        :return: 实际坐标系中的点 (u, v)
        """
        if self.H is None:
            raise RuntimeError("请先进行标定")
        
        x, y = point
        src_vec = np.array([x, y, 1], dtype=np.float32)
        dst_vec = self.H @ src_vec
        
        # 齐次坐标归一化；分母异常时使用仿射矩阵兜底，避免产生 nan/inf。
        if np.isfinite(dst_vec[2]) and abs(dst_vec[2]) > 1e-6:
            u = dst_vec[0] / dst_vec[2]
            v = dst_vec[1] / dst_vec[2]
        elif self.affine_matrix is not None:
            affine_dst = self.affine_matrix @ src_vec
            u, v = affine_dst[0], affine_dst[1]
        else:
            raise ValueError("托盘标定投影失败：齐次坐标分母为0，请先重新标定托盘")

        if not np.all(np.isfinite([u, v])):
            raise ValueError("托盘标定投影结果包含无效数值，请重新标定托盘")
        return u, v

    def board_detector(self,board_bgr,pixel2world_client):
        self.dst_points = []
        board_cam_4_points = board_detect(board_bgr)
        if len(board_cam_4_points) != 4 or not np.all(np.isfinite(board_cam_4_points)):
            raise ValueError("托盘角点识别结果无效，请检查托盘是否完整进入画面")
        print("托盘像素坐标",board_cam_4_points)
        dx1=board_cam_4_points[2][0]-board_cam_4_points[0][0]
        dx2=board_cam_4_points[3][0]-board_cam_4_points[1][0]
        dx=(dx1+dx2)/2
        dy1=board_cam_4_points[2][1]-board_cam_4_points[0][1]
        dy2=board_cam_4_points[3][1]-board_cam_4_points[1][1]
        dy=(dy1+dy2)/2
        if dx==0:
            self.board_theta=0
        else:
            self.board_theta = np.degrees(np.arctan2(dy, dx))
        
        for (u, v) in board_cam_4_points:
################################################       9点标定法(获取方块坐标那里也要同步改)      #########################################################
            pred_X = board_x.predict(np.array([u, v]))
            pred_Y = board_y.predict(np.array([u, v]))
            self.dst_points.append( (pred_X, pred_Y) )
################################################       深度相机法        #########################################################
            # point3d = pixel2world_client(pixel2worldRequest(int(round(u)), int(round(v)))).world_position
            # self.dst_points.append([point3d[0],point3d[1]])

        # cv2.imshow('board_bgr', board_bgr)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()





