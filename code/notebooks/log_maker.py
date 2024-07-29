import os
import numpy as np
import sys
#adding multiprocessing
from multiprocessing import Pool
"""
file_li = [l.strip("\n") for l in open('extrafeat.txt')]
#done_list = [line.strip("\n") for line in open('log_07122.txt')]
for file in file_li:
    #if file in done_list: continue
    name=file.split(".")[1][1:]
    with open("feature_extra_list.txt", "a") as f:
        f.write(f"python featurize_ss_0726.py {name}.pdb\n")
f.close()
print("Done")
"""

def execute_command(command):
    os.system(command)

if __name__ == "__main__":
    # 실행할 명령어가 담긴 텍스트 파일 읽기
    print(sys.argv)

    with open('feature_extra_list.txt', "r") as file:

        commands = [line.strip() for line in file]

    # 멀티프로세스 풀 생성
    with Pool(int(sys.argv[1])) as pool:
        # 명령어를 병렬로 실행
        pool.map(execute_command, commands)

    # 모든 명령어가 실행된 후에 실행 종료 메시지 출력
    print("All commands have been executed and the process is finished.")

