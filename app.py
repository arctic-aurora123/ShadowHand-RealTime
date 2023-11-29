import cv2
import sys
import os
import numpy as np
import pygame
import torch
import time
import argparse
import torch.backends.cudnn as cudnn
import pyrealsense2 as rs
import jax.numpy as npj
from jax import grad, jit, vmap
from jax.experimental import optimizers
from torchvision.transforms import functional
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from manolayer import ManoLayer
from model import HandNet
from checkpoints import CheckpointIO
import utils
import sqlite3
from decompose import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

mano_layer = ManoLayer(center_idx=9, side="right", mano_root=".", use_pca=False, flat_hand_mean=True,)
mano_layer = jit(mano_layer)

conn = sqlite3.connect('hand.db')
conn.execute('CREATE TABLE IF NOT EXISTS pose (x REAL, y REAL, z REAL)')
print('Database connect successfully!')
cursor = conn.execute('SELECT * FROM pose')
rot = cursor.fetchall()

class RealSenseCapture:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
    
    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return np.flip(color_image, -1).copy()

# 定义OpenCVCapture类
class OpenCVCapture:
    # 初始化函数
    def __init__(self):
        # 创建一个视频流对象
        self.cap = cv2.VideoCapture(0)

    # 读取函数
    def read(self):
        # 读取视频流
        ret, frame = self.cap.read()
        # 将帧大小调整为640x480
        frame = cv2.resize(frame, (640, 480))
        # 将帧转换为numpy数组
        color_image = np.asanyarray(frame)
        # 返回翻转后的numpy数组
        return np.flip(color_image, -1).copy()

# 给下面的每一行代码都添加中文注释返回完整的代码
@jit
def hm_to_kp2d(hm):
    # 获取hm的形状
    b, c, w, h = hm.shape
    # 将hm的形状转换为(b,c,w*h)
    hm = hm.reshape(b,c,-1)
    # 计算hm的每一项的值
    hm = hm/npj.sum(hm,-1,keepdims=True)
    # 创建x和y的坐标图
    coord_map_x = npj.tile(npj.arange(0,w).reshape(-1,1), (1,h))
    coord_map_y = npj.tile(npj.arange(0,h).reshape(1,-1), (w,1))
    # 将坐标图的形状转换为(1,1,w*h)
    coord_map_x = coord_map_x.reshape(1,1,-1)
    coord_map_y = coord_map_y.reshape(1,1,-1)
    # 计算hm中每个像素点在x和y方向上的坐标
    x = npj.sum(coord_map_x * hm,-1,keepdims=True)
    y = npj.sum(coord_map_y * hm,-1,keepdims=True)
    # 将x和y的坐标拼接起来
    kp_2d = npj.concatenate((y,x),axis=-1)
    # 返回kp_2d
    return kp_2d

@jit
def reinit_root(joint_root,kp2d,camparam):
    uv = kp2d[0,9,:]
    xy = joint_root[...,:2]
    z = joint_root[...,2]
    joint_root = ((uv - camparam[0, 0, 2:4])/camparam[0, 0, :2]) * z
    joint_root = npj.concatenate((joint_root,z))
    return joint_root

@jit
def reinit_scale(joint,kp2d,camparam,bone,joint_root):
    z0 = joint_root[2:]
    xy0 = joint_root[:2]
    xy = joint[:,:2] * bone
    z = joint[:,2:] * bone
    kp2d = kp2d[0]
    s1 = npj.sum(((kp2d - camparam[0, 0, 2:4])*xy)/(camparam[0, 0, :2]*(z0+z)) - (xy0*xy)/((z0+z)**2))
    s2 = npj.sum((xy**2)/((z0+z)**2))
    s = s1/s2
    bone = bone * npj.max(npj.array([s,0.9]))
    return bone

@jit
def geo(joint):
    idx_a = npj.array([1,5,9,13,17])
    idx_b = npj.array([2,6,10,14,18])
    idx_c = npj.array([3,7,11,15,19])
    idx_d = npj.array([4,8,12,16,20])
    p_a = joint[:,idx_a,:]
    p_b = joint[:,idx_b,:]
    p_c = joint[:,idx_c,:]
    p_d = joint[:,idx_d,:]
    v_ab = p_a - p_b #(B, 5, 3)
    v_bc = p_b - p_c #(B, 5, 3)
    v_cd = p_c - p_d #(B, 5, 3)
    loss_1 = npj.abs(npj.sum(npj.cross(v_ab, v_bc, -1) * v_cd, -1)).mean()
    loss_2 = - npj.clip(npj.sum(npj.cross(v_ab, v_bc, -1) * npj.cross(v_bc, v_cd, -1)), -npj.inf, 0).mean()
    loss = 10000*loss_1 + 100000*loss_2

    return loss

@jit
def residuals(input_list,so3_init,beta_init,joint_root,kp2d,camparam):
    so3 = input_list['so3']
    beta = input_list['beta']
    bone = input_list['bone']
    so3 = so3[npj.newaxis,...]
    beta = beta[npj.newaxis,...]
    _, joint_mano, _ = mano_layer(
        pose_coeffs = so3,
        betas = beta
    )
    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
    bone_pred = bone_pred[:,npj.newaxis,...]
    reg = ((so3 - so3_init)**2)
    reg_beta = ((beta - beta_init)**2)
    joint_mano = joint_mano / bone_pred
    joint_mano = joint_mano * bone + joint_root
    geo_reg = geo(joint_mano)
    xy = (joint_mano[...,:2]/joint_mano[...,2:])
    uv = (xy * camparam[:, :, :2] ) + camparam[:, :, 2:4]
    errkp = ((uv - kp2d)**2)
    err = 0.01*reg.mean() + 0.01*reg_beta.mean() + 1*errkp.mean() + 100*geo_reg.mean()
    return err

@jit
def mano_de(params,joint_root,bone):
    so3 = params['so3']
    beta = params['beta']
    verts_mano, joint_mano, _ = mano_layer(
        pose_coeffs = so3[npj.newaxis,...],
        betas = beta[npj.newaxis,...]
    )

    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :],axis=1, keepdims=True)
    bone_pred = bone_pred[:,npj.newaxis,...]
    verts_mano = verts_mano / bone_pred
    verts_mano = verts_mano * bone  + joint_root
    v = verts_mano[0]
    print(v)
    return v

@jit
def mano_de_j(so3, beta):
    _, joint_mano, _ = mano_layer(
        pose_coeffs = so3[npj.newaxis,...],
        betas = beta[npj.newaxis,...]
    )

    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :],axis=1, keepdims=True)
    bone_pred = bone_pred[:,npj.newaxis,...]
    joint_mano = joint_mano / bone_pred
    j = joint_mano[0]
    print(j)
    return j

def plt_show(data):
    plt.clf()  # 清除之前画的图
    fig = plt.gcf()  # 获取当前图
    ax = fig.gca(projection='3d')  # 获取当前轴
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]
    ax.scatter(X, Y, Z)
    for i in range(len(X)):
        ax.text(X[i], Y[i], Z[i], i)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=-90)
    plt.pause(0.01)
    ## 不让窗口关闭
    plt.show()

# 定义一个live_application函数，用于实时应用，参数为capture和arg
def live_application(capture,arg):
    plt.ion()
    # 定义窗口大小
    window_size = 768
    # 初始化pygame
    pygame.init()
    # 设置显示窗口
    display = pygame.display.set_mode((window_size, window_size))
    # 设置窗口标题
    pygame.display.set_caption('Real Time Hand Recon')

    # 加载模型
    model = HandNet()
    # 将模型移动到设备
    model = model.to(device)
    # 加载模型参数
    checkpoint_io = CheckpointIO('.', model=model)
    load_dict = checkpoint_io.load('checkpoints/model.pt')
    # 设置模型为评估模式
    model.eval()

    # 加载 MANO_RIGHT.pkl 文件
    dd = pickle.load(open("MANO_RIGHT.pkl", 'rb'), encoding='latin1')
    # 获取MANO_RIGHT.pkl文件中的f参数
    face = np.array(dd['f'])

    # 初始化渲染器
    renderer = utils.MeshRenderer(face, img_size=256)

    # 获取参数
    cx = arg.cx
    cy = arg.cy
    fx = arg.fx
    fy = arg.fy

    # 循环读取帧
    while True:
        # 读取帧
        img = capture.read()
        # 如果读取的帧为空，则继续读取
        if img is None:
            continue
        # 如果帧的宽大于帧的高，则将帧的上下边缘缩减
        if img.shape[0] > img.shape[1]:
            margin = int((img.shape[0] - img.shape[1]) / 2)
            cy = cy - margin
            width = img.shape[1]
        # 如果帧的宽小于帧的高，则将帧的左右边缘缩减
        elif img.shape[0] < img.shape[1]:
            margin = int((img.shape[1] - img.shape[0]) / 2)
            cx = cx - margin
            width = img.shape[0]
        # 计算cx和cy的值
        cx = (cx * 256)/width
        cy = (cy * 256)/width
        # 计算fx和fy的值
        fx = (fx * 256)/width
        fy = (fy * 256)/width
        # 打印读取帧的信息
        print('reading frames...')
        break
    # 打印读取帧成功
    print('read frames successfully')
    # 初始化内参
    intr = torch.from_numpy(np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)).unsqueeze(0).to(device)

    # 将内参转换为numpy数组
    _intr = intr.cpu().numpy()

    # 初始化camparam
    camparam = np.zeros((1, 21, 4))
    # 将内参赋值给camparam
    camparam[:, :, 0] = _intr[:, 0, 0]
    camparam[:, :, 1] = _intr[:, 1, 1]
    camparam[:, :, 2] = _intr[:, 0, 2]
    camparam[:, :, 3] = _intr[:, 1, 2]
    
    # 初始化渲染器
    gr = jit(grad(residuals))
    lr = 0.03
    # 初始化优化器
    opt_init, opt_update, get_params = optimizers.adam(lr, b1=0.5, b2=0.5)
    opt_init = jit(opt_init)
    opt_update = jit(opt_update)
    get_params = jit(get_params)
    i = 0
    sum = np.zeros((16,3))
    m = 0
    # 循环读取帧
    with torch.no_grad():
        while True:
            i = i + 1
            # 读取帧
            img = capture.read()
            # 如果读取的帧为空，则继续读取
            if img is None:
                continue
            # 如果帧的宽大于帧的高，则将帧的上下边缘缩减
            if img.shape[0] > img.shape[1]:
                margin = int((img.shape[0] - img.shape[1]) / 2)
                img = img[margin:-margin]
            # 如果帧的宽小于帧的高，则将帧的左右边缘缩减
            elif img.shape[0] < img.shape[1]:
                margin = int((img.shape[1] - img.shape[0]) / 2)
                img = img[:, margin:-margin]
            # 将帧缩放到256*256
            img = cv2.resize(img, (256, 256),cv2.INTER_LINEAR)
            # 复制帧
            frame = img.copy()

            # 将帧转换为tensor
            img = functional.to_tensor(img).float()
            # 归一化帧
            img = functional.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
            # 将帧移动到设备
            img = img.unsqueeze(0).to(device)
            
            # 运行模型
            hm, so3, beta, joint_root, bone = model(img,intr)
            # 将hm转换为kp2d
            kp2d = hm_to_kp2d(hm.detach().cpu().numpy())*4
            # 将so3转换为numpy数组
            so3 = so3[0].detach().cpu().float().numpy()
            # 将beta转换为numpy数组
            beta = beta[0].detach().cpu().float().numpy()
            # 将bone转换为numpy数组
            bone = bone[0].detach().cpu().numpy()
            # 将joint_root转换为numpy数组
            joint_root = joint_root[0].detach().cpu().numpy()
            # 将so3转换为tensor
            so3 = npj.array(so3)
            # 将beta转换为tensor
            beta = npj.array(beta)
            # 将bone转换为tensor
            bone = npj.array(bone)
            # 将joint_root转换为tensor
            joint_root = npj.array(joint_root)
            # 将kp2d转换为tensor
            kp2d = npj.array(kp2d)
            # 将so3_init赋值为so3
            so3_init = so3
            # 将beta_init赋值为beta
            beta_init = beta
            # 重新初始化根节点
            joint_root = reinit_root(joint_root,kp2d, camparam)
            # 重新初始化 joint
            joint = mano_de_j(so3, beta)
            joint_np = np.array(joint)
            old_joint = np.zeros(joint.shape)
            # 重新初始化 bone
            bone = reinit_scale(joint,kp2d,camparam,bone,joint_root)
            # 初始化参数
            params = {'so3':so3, 'beta':beta, 'bone':bone}

            # 初始化优化器状态
            opt_state = opt_init(params)
            # 初始化n
            n = 0
            # 循环20次
            while n < 20:
                n = n + 1
                # 获取参数
                params = get_params(opt_state)
                # 计算梯度
                grads = gr(params,so3_init,beta_init,joint_root,kp2d,camparam)
                # 更新参数
                opt_state = opt_update(n, grads, opt_state)
            # 获取参数
            params = get_params(opt_state)
            # 计算v
            v = mano_de(params,joint_root,bone)
            # 渲染v
            frame = renderer(v,intr[0].cpu(),frame)
            # 将渲染结果显示到窗口
            display.blit(
                pygame.surfarray.make_surface(np.transpose(cv2.resize(np.flip(frame,1), (window_size, window_size),cv2.INTER_LINEAR), (1, 0, 2))),(0, 0))
            # 更新窗口
            pygame.display.update()
            so3 = np.array(so3)
            so3 = so3.reshape(16,3)
            rotation = axis_angle_to_rotations(so3)/np.pi*180*1.4
            # 输出16*3的矩阵，每个3维向量的分量分别表示绕x、y、z的角度
            conn.execute('DELETE FROM pose')
            for i in range(16):
                conn.execute('INSERT INTO pose (x,y,z) VALUES (?, ?, ?)',
                    (rotation[i][0], rotation[i][1], rotation[i][2]))
            conn.commit()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cx',
        type=float,
        default=321.2842102050781,
    )

    parser.add_argument(
        '--cy',
        type=float,
        default=235.8609161376953,
    )

    parser.add_argument(
        '--fx',
        type=float,
        default=612.0206298828125,
    )

    parser.add_argument(
        '--fy',
        type=float,
        default=612.2821044921875,
    )
    live_application(OpenCVCapture(),parser.parse_args())
    conn.close()
