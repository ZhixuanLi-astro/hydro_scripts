# import imageio.v2 as imageio
# import glob
# import sys
#
# # Gather all relevant files and sort them
# try:
#     pname = sys.argv[1]
# except:
#     pname = 'fig_snow_2d'
#
# file_list = glob.glob('./plots/{}_0*.png'.format(pname)) 
# # sort them by the number in the filename 
# file_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0])) 
#
# # Read images
# images = [imageio.imread(fname) for fname in file_list]
#
# # Save as GIF
# # imageio.mimsave('fig_snow_2d_animation.gif', images, duration=0.2)  # duration in seconds per frame
#
# # If you want an MP4 (requires ffmpeg):
# imageio.mimsave('{}_animation.mp4'.format(pname), images, fps=40)
import subprocess
import glob
import sys
import os
import re  # 新增引入正则表达式库

def main():
    pname = sys.argv[1] if len(sys.argv) > 1 else 'fig_snow_2d'
    
    # 1. 先查找文件夹下所有匹配的 PNG 图片，并按名称排序
    all_files = sorted(glob.glob(f'./plots/{pname}_*.png'))
    
    if not all_files:
        print("Error: No files found.")
        return

    # 2. 从第一张图片中提取起始编号 (例如从 mmax_00951.png 里提取出 951)
    first_file = os.path.basename(all_files[0])
    # # 使用正则提取文件名中的数字部分
    # match = re.search(r'(\d+)', first_file)
    # if not match:
    #     print("Error: Could not determine start number from filename.")
    #     return
    num_str = int(first_file[:-4][-5:])
    start_number = int(num_str)
    
    print(f"Found {len(all_files)} files. Starting from number: {start_number}")

    # 3. 定义 ffmpeg 输入模式
    input_pattern = f'./plots/{pname}_%05d.png'
    output_file = f'{pname}_animation.mp4'

    # 4. 构建 ffmpeg 命令，核心改动是加入了 -start_number 参数
    cmd = [
        'ffmpeg',
        '-r', '20',                       # 输入帧率
        '-start_number', str(start_number), # 【关键】明确告诉 ffmpeg 从第几号图片开始读取
        '-i', input_pattern,              # 输入图片序列模式
        '-c:v', 'libx264',                # 视频编码器
        '-pix_fmt', 'yuv420p',            # 像素格式（兼容性最好）
        '-y',                             # 覆盖已有视频
        '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # 确保宽高为偶数
        output_file
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Video created successfully.")

if __name__ == '__main__':
    main()
