# w : 쓰기, r : 읽기, x : 배타적 생성
# a : 추가, t : 텍스트 모드, b : 바이너리 모드
# + : 읽기/쓰기
file = open('hello.txt', 'w') 
file.write('hello world!')
file.close()

# 읽기로 파일 열기
# 자동으로 파일 닫기
with open('hello.txt', 'r') as file:
    s = file.read()
    print(s)

with open('for_hello.txt', 'w') as file:
    for i in range(3):
        file.write(f'Hello, world! {i}\n')

with open('list_hello.txt', 'w') as file:
    lines = ['안녕하세요.\n', '파이썬\n', '코딩 도장입니다.\n', 'ㅅㄷㄴㅅ', 'test']
    file.writelines(lines)

with open('list_hello.txt', 'r') as file:
    lines = file.readlines() # 모든 라인 리스트로 가져오기
    print(lines)

with open('list_hello.txt', 'r') as file:
    line = None
    while line != '':
        line = file.readline() # 한 줄씩 읽기
        print(line.strip('\n'))

with open('for_hello.txt', 'r') as file:
    for line in file: # for 문으로 파일 읽기
        print(line.strip('\n'))
    # 파일 객체는 이터레이터임

import pickle
 
name = 'james'
age = 17
address = '서울시 서초구 반포동'
scores = {'korean': 90, 'english': 95, 'mathematics': 85, 'science': 82}
 
with open('james.p', 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
    pickle.dump(name, file)
    pickle.dump(age, file)
    pickle.dump(address, file)
    pickle.dump(scores, file)

with open('james.p', 'rb') as file:    # james.p 파일을 바이너리 읽기 모드(rb)로 열기
    name = pickle.load(file) # dump 한 순서대로 나옴
    age = pickle.load(file)
    address = pickle.load(file)
    scores = pickle.load(file)
    print(name)
    print(age)
    print(address)
    print(scores)