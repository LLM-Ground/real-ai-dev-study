print('Hello, world!')
print('Python Programming')

# python 은 ;(세미콜론)을 사용하지 않음
# 한 줄에 여러 구문을 사용할 때만 사용
print('Hello'); print('1234')

# 들여쓰기 자체가 문법이라 if 같은 구문의 코드 블럭은 항상 들여쓰기 해야됨
a = 10
if a == 10:
    print('10 입니다.')
# python 코딩 스타일 가이드에서는 공백 4칸을 추천

print('계산')
print(10+2)
print(10-2)
print(10*2)
print(10/2)
print(10//3) # 나머지 버림
print(10%3)
print(10**3)

num = 3.3
print(int(num))
print(int('10'))
print(type(num)) # 자료형(타입) 확인

# 몫과 나머지 한번에 구하기
quotient, remainder = divmod(5, 2) # 변수 여러 개 한번에 만들기
print(quotient, remainder)

a = b = c = 3 # 값이 같은 변수 여러 개 한번에 만들기
del b # 변수 삭제
x = None # 빈 변수 만들기, null 같은거임
# print(b) # NameError: name 'b' is not defined
print(a, c, x) # 여러 개 한번에 출력
print(a, c, x, sep=', ') # 구분자 지정(default ' ')
print(a,end='') # 출력의 마지막 문자 지정(default \n)
print(c,end='')
print(x)

# 불(boolean)과 비교연산자
print(True)
print(False)
print(a == c)
print(a != c)
d = 4
print(a > d)
print(a >= d)
print(a < d)
print(a <= d)

print('객체 비교')
print(1 == 1.0) # 값 비교
print(1 is 1.0) # 객체 비교
print(1 is not 1.0)
print(id(1)) # 고유값 확인
print(id(1.0))

print('논리연산')
print(True and True, True and False, False and False)
print(True or True, True or False, False or False)
print(not True, not False)
print(not True and False or not False) # not, and, or 순으로 판단
print(bool(1), bool(0), bool(0.0), bool('False'), bool(''), bool(None))
# 논리연산은 단락 평가(short-circuit evalution) 를 사용해서
# 중간에 결과가 확실해지면 이후는 확인하지 않음
print(True and 'python', False and 'python', True or 'python')

print('문자열')
print('문자열은 작은따옴표로 하나')
print("큰따옴표로 하나 똑같음")
string = '''근데
따옴표 3개로 묶으면
여러줄 표현 가능'''
print(string)
print(f'''그리고 이렇게 f문자열을 
사용해서 a({a}) 같은 변수를 사용할 수도 있음''')

print()
print('자료구조')
print('list', [1,2,3,4, 'a', True, [1,2,3]], [], list(), list('hello'))
print('range', range(0, 10), list(range(0, 10)), list(range(20, 3, -3)))
a = (1,2,5,3,6) # 튜플 선언
b = 1,4,56,8
c = (38) # 이렇게 요소 하나는 안됨
d = 38, # 뒤에 콤마를 붙여줘야됨
print('tuple', a, b, c, d, tuple('hello'))
x, y, z, w, q = a # 분해할당 가능. 근데 변수 개수가 같아야함
print(x, y, z, w, q)
print('특정값의 인덱스', b.index(56))
print('특정값의 개수', b.count(56))

# list, range, tuple, str 처럼 연속적인 값으로 이어진 자료형을 시퀀스 자료형이라고 함
# 시쿼스 자료형은 공통 기능이 있음
print(3 in a) # 안에 값이 있는지 확인
print(10 not in a) # 안에 값이 없는지 확인
print(a + b) # 시퀀스 객체 연결(range 는 안됨)
print(a * 3) # 시퀀스 객체 반복
print(len(a)) # 시퀀스 객체 요소 개수
print(a[3]) # 인덱스 사용
# 실제로는  __getitem__ 메서드를 호출해서 요소를 가져옴
print(a[-2]) # 인덱스 음수로 접근 가능
print(a[1:3]) # 슬라이스 사용
print(a[2:]) # 슬라이스 사용
print(a[:3]) # 슬라이스 사용
print(a[1:4:2]) # [시작인덱스 : 끝인덱스 : 증가폭]
print('hello, python'[slice(3, 10, 2)]) # slice 객체 사용
# 슬라이스에 요소 할당도 가능
list1 = list(range(0,10))
list1[2:5] = [20,30,40]
print(list1)
del list1[2:5]
print(list1)

print('딕셔너리')
lux = {'health': 490, 'mana': 334, 'melee': 550, 'armor': 18.72}
print(lux)
x = {100: 'hundred', False: 0, 3.5: [3.5, 3.5]}
print(x, '키에 모든 자료형 사용 가능')
print('빈 딕셔너리', {}, dict())
print(dict(health=490, mana=334, melee=550, armor=18.72))
print(dict(zip(['health', 'mana', 'melee', 'armor'], [490, 334, 550, 18.72])))
print(dict([('health', 490), ('mana', 334), ('melee', 550), ('armor', 18.72)]))
print('딕셔너리에 접근', lux['health'])
print('딕셔너리에 키가 있는지 확인', 'health' in lux)

if 'health' in lux:
    print('if 문 분기')
elif 'melee' in lux:
    print('elif 분기')
else:
    print('else')

print('삼항연산자 대신 if 문' if 'mana' in lux else 123)
'''
다음은 파이썬 문법 중에서 False로 취급하는 것들입니다.

None

False

0인 숫자들: 0, 0.0, 0j

비어 있는 문자열, 리스트, 튜플, 딕셔너리, 세트: '', "", [], (), {}, set()
'''

print('for 문')
for i in range(10):
    print(i, end=', ')
print()
for i in range(10, 0, -1):
    print(i, end=', ')
print()

# 시퀀스 객체로 반복
for letter in reversed('Python'):
    print(letter, end=' ')
print()
for index, value in enumerate([38, 21, 53, 62, 19]):
    print(index, value)

i = 1
while i <= 5:
    print('while 문 반복 ', i)
    i += 1

import random
i = 0
dice = [1, 2, 3, 4, 5, 6]
while i != 3:    # 3이 아닐 때 계속 반복
    # i = random.randint(1, 6)    # randint를 사용하여 1과 6 사이의 난수를 생성
    i = random.choice(dice)
    print(i)


# import copy             # copy 모듈을 가져옴
# b = copy.deepcopy(a)    # copy.deepcopy 함수를 사용하여 깊은 복사