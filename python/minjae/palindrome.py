# palindrome 판별

word = 'level'

# for 문으로 판별
is_palindrome = True
for i in range(len(word) // 2):
    if word[i] != word[-1 - i]:
        is_palindrome = False
        break
print(f'{word} is palindrome {is_palindrome}')

# 시퀀스 뒤집어서 검사
print(f'{word} is palindrome {word == word[::-1]}')

# list 와 reversed 사용
print(f'{word} is palindrome {list(word) == list(reversed(word))}')

# reversed 와 join 사용
print(f'{word} is palindrome {word == ''.join(reversed(word))}')


# N-gram : 문자열에서 N 개의 연속된 요소 추출 방법
'''
예)
hello => he, el, ll, lo
'''
text = 'Hello'

# for 문으로 출력
for i in range(len(text) - 1):
    print(text[i], text[i+1], sep='')

# 단어 단위 N-gram
text = 'this is python script'
words = text.split()
 
for i in range(len(words) - 1):
    print(words[i], words[i + 1])

# zip 으로 2-gram
text = 'Hello'
two_gram = zip(text, text[1:])
for i in two_gram:
    print(*i, sep='')

# zip과 list 표현식으로 N-gram 만들기
for z in zip(*[text[i:] for i in range(3)]):
    print(*z, sep='')
