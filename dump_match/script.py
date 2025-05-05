#!/usr/bin/env python3
import subprocess
import sys

def run_with_params(script_path, sequences, output_files):
    """
    依次调用 script_path,每次传入一个 --sequences 和对应的 --output_file 参数。
    """
    for seq, out_file in zip(sequences, output_files):
        cmd = [
            sys.executable,
            script_path,
            "--sequences", seq,
            "--output_file", out_file
        ]
        print(f"▶ Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("↳ stdout:")
            print(result.stdout.strip() or "(no output)")
        except subprocess.CalledProcessError as e:
            print(f"‼ Error (exit {e.returncode}):")
            print(e.stderr.strip() or "(no error message)")
        print("-" * 60)

if __name__ == "__main__":
    # 要调用的脚本
    script_to_call = "/hpc2hdd/home/qzhang749/yanzhe/test/NACNet/OANet/dump_match/ETH.py"

    # 指定的序列名称列表
    sequence_list = [
        "exhibition_hall", "living_room", "observatory", "statue",
        "boulders", "door", "lecture_room", "lounge", "old_computer", "terrace_2"
    ]

    # 对应的 output_file 列表
    output_file_list = [
        f"ETH-{seq}-sift-1000-test.hdf5" for seq in sequence_list
    ]

    run_with_params(script_to_call, sequence_list, output_file_list)
