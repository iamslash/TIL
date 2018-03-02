<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Intro](#intro)
- [References](#references)
- [Tips](#tips)
    - [필드 출력하기](#필드-출력하기)

<!-- markdown-toc end -->


# Abstract

c문법을 연상시키는 텍스트 처리 언어이다. 로그 파일등을 읽어서 원하는
정보를 선택하여 출력할 때 사용한다.

# Basic

awk command line은 보통 다음과 같은 형식을 갖는다.

```bash
awk 'program' input-file1 input-file2 ...
```

다음은 `df` 출력의 네 번째 항목을 `awk`와 `sed`를 사용하여 출력하는
예이다. `awk`가 훨씬 간단하다. 

```
$ df -h
$ df -h | sed -rn '2s/(\S+\s+){3}(\S+).*/\2/p'
# NR==2 는 2번째 record, $4는 4번째 column을 의미한다.
$ df -h | awk 'NR==2{ print $4 }'
```

# Builtin Variables

# References

* [awk 스크립트 가이드](https://mug896.gitbooks.io/awk-script/content/)
  * 친절한 한글
* [awk manual @ gnu](https://www.gnu.org/software/gawk/manual/gawk.html#Getting-Started)
* [부록 B. Sed 와 Awk 에 대한 간단한 입문서](https://wiki.kldp.org/HOWTO/html/Adv-Bash-Scr-HOWTO/sedawk.html)

# Tips

## 필드 출력하기

* 3번째 필드를 출력하자.

```bash
awk '{print $3}' a.txt
```

* 1, 5, 6번째 필드를 출력하자.

```bash
awk '{print $1 $5 $6}' a.txt
```
