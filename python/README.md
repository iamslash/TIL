# Abstract

python에 대해 정리한다.

# Usage

## pdb

pdb로 break point하고 싶은 라인에 다음을 복사하고 실행하자.

```python
import pdb; pdb.set_trace()
```
# Library

## regex

```python
import re
p = re.compile(r'(?P<word>\b\w*\b)')
m = p.search('(((( Lots of punctuation )))')
print(m.group('word'))
print(m.group(0))
print(m.group(1))
```

## numpy

## pandas

# References

* [파이썬 생존 안내서](https://www.slideshare.net/sublee/ss-67589513)
  * 듀랑고를 제작한 왓스튜디오의 이흥섭 PT
