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
    

pixel_coords = np.array(
    [[214,88], [607,67], [1062,79], 
     [164,328], [608,334], [1062,304], 
     [188,639], [624,596], [1013,652],
])

x_coords = [-520.041015625, -528.2529296875, -518.8900146484375, 
            -395.3943786621093, -390.072998046875, -403.550048828125, 
            -233.0176391601562, -254.0950927734375, -223.22802734375]

y_coords = [-213.2770538330078, -7.045026779174804, 227.898727416992, 
            -241.3478698730468, -7.272053718566894, 227.9611663818359, 
            -229.2982330322265, 0.8525112271308899, 202.7843475341797]

z_coords = [169.2613677978515, 170.1771850585937, 171.8448028564453, 
            168.141860961914, 169.529571533203, 171.149169921875, 
            167.491729736328, 169.3529663085937, 170.4917602539062]



predictor_x = CoordinatePredictor(degree=2)
predictor_x.fit(pixel_coords, x_coords)
predictor_y = CoordinatePredictor(degree=2)
predictor_y.fit(pixel_coords, y_coords)
predictor_z = CoordinatePredictor(degree=2)
predictor_z.fit(pixel_coords, z_coords)

def x_predict(px,py):
    return predictor_x.predict(np.array([px,py]))
def y_predict(px,py):
    return predictor_y.predict(np.array([px,py]))
def z_predict(px,py):
    return predictor_z.predict(np.array([px,py]))


class ArmCalibrator:
    def __init__(self):
        self.H = None  # 变换矩阵
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
        
        # 齐次坐标归一化
        u = dst_vec[0] / dst_vec[2]
        v = dst_vec[1] / dst_vec[2]
        return u, v

    def board_detector(self,board_bgr,pixel2world_client):
        self.dst_points = []
        board_cam_4_points = board_detect(board_bgr)
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
            pred_X = x_predict(u, v)
            pred_Y = y_predict(u, v)
            self.dst_points.append( (pred_X, pred_Y) )
################################################       深度相机法        #########################################################
            # point3d = pixel2world_client(pixel2worldRequest(int(round(u)), int(round(v)))).world_position
            # self.dst_points.append([point3d[0],point3d[1]])

        # cv2.imshow('board_bgr', board_bgr)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows()






