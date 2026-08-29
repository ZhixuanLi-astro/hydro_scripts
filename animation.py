import subprocess
import glob
import sys
import os
def make_mp4(pname, input_pattern, start_number):
    """用图片序列生成 MP4 视频"""
    output_file = f'{pname}_animation.mp4'
    cmd = [
        'ffmpeg',
        '-r', '10',                         # 输入帧率
        '-start_number', str(start_number), # 告诉 ffmpeg 从第几号图片开始读取
        '-i', input_pattern,                # 输入图片序列模式
        '-c:v', 'libx264',                  # 视频编码器
        '-pix_fmt', 'yuv420p',              # 像素格式（兼容性最好）
        '-y',                               # 覆盖已有视频
        '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # 确保宽高为偶数
        output_file
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Video created: {output_file}")


def make_gif(pname, input_pattern, start_number, fps=10, width=480):
    """用图片序列生成 GIF（两遍法，颜色更漂亮）"""
    palette_file = f'{pname}_palette.png'
    output_file = f'{pname}_animation.gif'

    # 第一步：生成调色板
    cmd_palette = [
        'ffmpeg',
        '-r', str(fps),
        '-start_number', str(start_number),
        '-i', input_pattern,
        '-vf', f"fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff",
        '-y', palette_file
    ]
    print(f"Running: {' '.join(cmd_palette)}")
    subprocess.run(cmd_palette, check=True)

    # 第二步：用调色板输出 GIF
    cmd_gif = [
        'ffmpeg',
        '-r', str(fps),
        '-start_number', str(start_number),
        '-i', input_pattern,
        '-i', palette_file,
        '-filter_complex',
        f"[0:v]fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
        '-y', output_file
    ]
    print(f"Running: {' '.join(cmd_gif)}")
    subprocess.run(cmd_gif, check=True)
    os.remove(palette_file)
    print(f"GIF created: {output_file}")


def main():
    pname = sys.argv[1] if len(sys.argv) > 1 else 'fig_snow_2d'
    # 第二个参数：mp4（默认）/ gif / both
    mode = sys.argv[2] if len(sys.argv) > 2 else 'mp4'

    # 1. 先查找文件夹下所有匹配的 PNG 图片，并按名称排序
    all_files = sorted(glob.glob(f'./plots/{pname}_*.png'))

    if not all_files:
        print("Error: No files found.")
        return

    # 2. 从第一张图片中提取起始编号 (例如从 fig_snow_2d_00951.png 里提取出 951)
    first_file = os.path.basename(all_files[0])
    num_str = int(first_file[:-4][-5:])
    start_number = int(num_str)

    print(f"Found {len(all_files)} files. Starting from number: {start_number}")

    # 3. 定义 ffmpeg 输入模式
    input_pattern = f'./plots/{pname}_%05d.png'

    if mode == 'gif':
        make_gif(pname, input_pattern, start_number)
    elif mode == 'both':
        make_mp4(pname, input_pattern, start_number)
        make_gif(pname, input_pattern, start_number)
    else:
        make_mp4(pname, input_pattern, start_number)


if __name__ == '__main__':
    main()
