import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_rgb_stats(folder_path):
    r_values, g_values, b_values = [], [], []

    # 폴더 내의 모든 파일에 대해 반복
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        
        # 파일이 이미지인 경우에만 처리
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(file_path) as img:
                img = img.convert('RGB')  # 이미지를 RGB로 변환
                pixels = np.array(img)  # 이미지를 numpy 배열로 변환
                
                # R, G, B 채널 분리하여 각 리스트에 추가
                r_values.append(pixels[:,:,0].flatten())
                g_values.append(pixels[:,:,1].flatten())
                b_values.append(pixels[:,:,2].flatten())
    
    # R, G, B 채널별로 합과 표준편차 계산
    r_values = np.concatenate(r_values)
    g_values = np.concatenate(g_values)
    b_values = np.concatenate(b_values)
    
    r_sum, g_sum, b_sum = r_values.mean(), g_values.mean(), b_values.mean()
    r_std, g_std, b_std = r_values.std(), g_values.std(), b_values.std()

    return (r_sum, r_std), (g_sum, g_std), (b_sum, b_std)

# 폴더 경로 지정
folder_path = 'coco/train2017'

# 함수 호출 및 결과 출력
(r_sum, r_std), (g_sum, g_std), (b_sum, b_std) = calculate_rgb_stats(folder_path)
print(f'R Mean: {r_sum/255}, R Std: {r_std/255}')
print(f'G Mean: {g_sum/255}, G Std: {g_std/255}')
print(f'B Mean: {b_sum/255}, B Std: {b_std/255}')
