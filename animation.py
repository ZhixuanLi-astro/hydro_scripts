import subprocess
import glob
import sys
import os

MODE_HELP = """\
usage: python animation.py <name> [mode] [fps] [width]

  name  : 图片前缀，例如 fig_snow_2d（读取 ./plots/<name>_XXXXX.png）
  mode  : mp4 (默认)   普通视频，H.264（不支持透明）
          gif          普通 GIF
          both         mp4 + gif
          alpha / mov  透明 .mov（QuickTime Animation, qtrle）
          prores       透明 .mov（ProRes 4444）
          apng         透明 APNG（.png 动图）
          gifalpha     透明 GIF（1-bit alpha）
          all          mp4 + gif + qtrle mov + apng + gifalpha
  fps   : 帧率（默认 10）
  width : 透明格式的缩放宽度（默认 1920；GIF 用 480）

note: 透明格式只有在 PNG 帧本身带 alpha（savefig(..., transparent=True)）时才透明。
"""


def make_mp4(pname, input_pattern, start_number, fps=10):
    """用图片序列生成 MP4 视频"""
    output_file = f'{pname}_animation.mp4'
    cmd = [
        'ffmpeg',
        '-r', str(fps),                     # 输入帧率
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


def make_mov_alpha(pname, input_pattern, start_number, fps=10,
                   codec='qtrle', width=1920):
    """透明背景视频 (.mov)：qtrle (Animation) 或 prores 4444，均带 alpha"""
    output_file = f'{pname}_animation_alpha.mov'
    if codec == 'prores':
        vcodec = ['-c:v', 'prores_ks', '-profile:v', '4444',
                  '-pix_fmt', 'yuva444p10le']
    else:
        vcodec = ['-c:v', 'qtrle', '-pix_fmt', 'argb']
    cmd = [
        'ffmpeg',
        '-r', str(fps),
        '-start_number', str(start_number),
        '-i', input_pattern,
        '-vf', f'scale={width}:-1:flags=lanczos',   # 原图过大，缩到常用视频宽度
        *vcodec,
        '-y',
        output_file
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Transparent video created: {output_file}")


def make_apng(pname, input_pattern, start_number, fps=10, width=1280):
    """透明背景动图 (APNG)：网页/文档友好，带完整 alpha"""
    output_file = f'{pname}_animation_alpha.png'
    cmd = [
        'ffmpeg',
        '-r', str(fps),
        '-start_number', str(start_number),
        '-i', input_pattern,
        '-vf', f'fps={fps},scale={width}:-1:flags=lanczos',
        '-c:v', 'apng',
        '-pix_fmt', 'rgba',
        '-plays', '0',
        '-f', 'apng',
        '-y',
        output_file
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"APNG created: {output_file}")


def make_gif_alpha(pname, input_pattern, start_number, fps=10, width=480):
    """带透明通道的 GIF（只有 1-bit alpha，边缘会发锯齿）"""
    palette_file = f'{pname}_palette_alpha.png'

    # 第一步：生成保留透明色的调色板
    cmd_palette = [
        'ffmpeg',
        '-r', str(fps),
        '-start_number', str(start_number),
        '-i', input_pattern,
        '-vf', f"fps={fps},scale={width}:-1:flags=lanczos,"
               f"palettegen=reserve_transparent=1:stats_mode=diff",
        '-y', palette_file
    ]
    print(f"Running: {' '.join(cmd_palette)}")
    subprocess.run(cmd_palette, check=True)

    # 第二步：用调色板输出带透明通道的 GIF
    cmd_gif = [
        'ffmpeg',
        '-r', str(fps),
        '-start_number', str(start_number),
        '-i', input_pattern,
        '-i', palette_file,
        '-filter_complex',
        f"[0:v]fps={fps},scale={width}:-1:flags=lanczos[x];"
        f"[x][1:v]paletteuse=alpha_threshold=128:dither=bayer:bayer_scale=5",
        '-y', f'{pname}_animation_alpha.gif'
    ]
    print(f"Running: {' '.join(cmd_gif)}")
    subprocess.run(cmd_gif, check=True)
    os.remove(palette_file)
    print(f"Transparent GIF created: {pname}_animation_alpha.gif")


def main():
    pname = sys.argv[1] if len(sys.argv) > 1 else 'fig_snow_2d'
    # 第二个参数：mp4（默认）/ gif / both / alpha / prores / apng / gifalpha / all
    mode = sys.argv[2] if len(sys.argv) > 2 else 'mp4'
    # 第三个参数（可选）：帧率；第四个参数（可选）：透明格式的缩放宽度
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    width = int(sys.argv[4]) if len(sys.argv) > 4 else 1920

    if mode in ('help', '-h', '--help'):
        print(MODE_HELP)
        return

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

    print(f"Mode = {mode}, fps = {fps}, width = {width}")

    if mode == 'gif':
        make_gif(pname, input_pattern, start_number, fps=fps)
    elif mode == 'both':
        make_mp4(pname, input_pattern, start_number, fps=fps)
        make_gif(pname, input_pattern, start_number, fps=fps)
    elif mode in ('alpha', 'mov', 'qtrle'):
        make_mov_alpha(pname, input_pattern, start_number, fps=fps,
                       codec='qtrle', width=width)
    elif mode in ('prores', 'prores4444'):
        make_mov_alpha(pname, input_pattern, start_number, fps=fps,
                       codec='prores', width=width)
    elif mode == 'apng':
        make_apng(pname, input_pattern, start_number, fps=fps, width=width)
    elif mode in ('gifalpha', 'gif_alpha'):
        make_gif_alpha(pname, input_pattern, start_number, fps=fps)
    elif mode == 'all':
        make_mp4(pname, input_pattern, start_number, fps=fps)
        make_gif(pname, input_pattern, start_number, fps=fps)
        make_mov_alpha(pname, input_pattern, start_number, fps=fps,
                       codec='qtrle', width=width)
        make_apng(pname, input_pattern, start_number, fps=fps, width=width)
        make_gif_alpha(pname, input_pattern, start_number, fps=fps)
    elif mode == 'mp4':
        make_mp4(pname, input_pattern, start_number, fps=fps)
    else:
        print(f"Unknown mode: {mode}\n")
        print(MODE_HELP)


if __name__ == '__main__':
    main()
