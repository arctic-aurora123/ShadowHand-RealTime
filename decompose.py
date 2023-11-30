import numpy as np 
  
def axis_angle_to_rotation_matrix(axis_angle):  
    # 将三维轴角转化为旋转矩阵  
    axis = axis_angle / np.linalg.norm(axis_angle)  # 归一化轴向量  
    angle = np.linalg.norm(axis_angle)  # 计算旋转角度  
    c = np.cos(angle)  
    s = np.sin(angle)  
    zero = np.zeros_like(angle)  
    one = np.ones_like(angle)  
      
    rotation_matrix = np.zeros((3, 3))  
    rotation_matrix[0, 0] = c + axis[0] ** 2 * (1 - c)  
    rotation_matrix[0, 1] = axis[0] * axis[1] * (1 - c) - axis[2] * s  
    rotation_matrix[0, 2] = axis[0] * axis[2] * (1 - c) + axis[1] * s  
    rotation_matrix[1, 0] = axis[0] * axis[1] * (1 - c) + axis[2] * s  
    rotation_matrix[1, 1] = c + axis[1] ** 2 * (1 - c)  
    rotation_matrix[1, 2] = axis[1] * axis[2] * (1 - c) - axis[0] * s  
    rotation_matrix[2, 0] = axis[0] * axis[2] * (1 - c) - axis[1] * s  
    rotation_matrix[2, 1] = axis[1] * axis[2] * (1 - c) + axis[0] * s  
    rotation_matrix[2, 2] = c + axis[2] ** 2 * (1 - c)  
      
    return rotation_matrix
  
  
# 将轴角向量转换为绕x、y、z轴旋转的角度
def axis_angle_to_rotations(axis_angle):  
    rotations = np.zeros((16, 3))  # 保存转换后的角度  
    for i in range(16):  
        rotation_matrix = axis_angle_to_rotation_matrix(axis_angle[i])  # 将轴角向量转换为旋转矩阵  
        # 从旋转矩阵中提取绕x、y、z轴旋转的角度  
        rotations[i, 0] = -np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])    # 绕x轴的角度  
        rotations[i, 1] = np.arctan2(-rotation_matrix[0, 1], np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[0, 2] ** 2))    # 绕y轴的角度  
        rotations[i, 2] = np.arctan2(rotation_matrix[0, 0], rotation_matrix[1, 1])  
    return rotations  
  

