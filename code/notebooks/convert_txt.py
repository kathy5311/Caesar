# 입력 파일 이름
input_file = "train_list_0730.csv"

# 출력 파일 이름
output_file = "train_list_0730.txt"

# 데이터를 쉼표로 구분하여 저장할 리스트
output_data = []

# 입력 파일 열기
with open(input_file, 'r') as file:
    # 각 줄에 대해 반복
    for line in file:
        # 띄어쓰기를 기준으로 데이터 분할
        data = line.split(",")
        #data = line.split(" ")
        
        # 쉼표로 구분된 문자열 생성하여 출력 데이터에 추가
        output_line = data[0]+'.'+data[1]+' '+data[2]
        output_data.append(output_line)

# 출력 파일에 결과 쓰기
with open(output_file, 'w') as file:
    for line in output_data:
        file.write(line)

print("변환 완료!")