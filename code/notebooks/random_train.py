import random
"""
# 파일에서 데이터 읽어오기
with open('train_list_pocket.txt', 'r') as file:
    data = file.readlines()
file.close()

#18000개만 뽑아오기
count=0
num = int(len(data)*0.08)
pocket_test_data=data[:num]
with open('train_list_pocket_test.txt','w') as file:
    for i in pocket_test_data:
        count+=1
        file.writelines(i)
print(count)
file.close()
"""
with open('/home/kathy531/Caesar/code/notebooks/balance_file/balanced_AA.txt', 'r') as file:
    data = file.readlines()
file.close()
print(data[:50])
# 데이터를 임의로 섞기
random.shuffle(data)
print(data[:50])

# 전체 데이터 중 80%를 추출
num_samples = int(len(data) * 0.8)
train_data = data[:num_samples]
valid_data=data[num_samples:]

# 추출된 데이터를 파일에 쓰기

count_t=0
with open('/home/kathy531/Caesar/code/notebooks/train_validset/trainset_bal_0820.txt', 'w') as file:
    for i in train_data:
        
        file.writelines(i)
        count_t+=1
print(count_t)
file.close()

count_v=0
with open('/home/kathy531/Caesar/code/notebooks/train_validset/validset_bal_0820.txt', 'w') as file:
    for i in valid_data:
        count_v+=1

        file.writelines(i)
print(count_v)
file.close()
