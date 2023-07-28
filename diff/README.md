- [Materials](#materials)
- [Basic](#basic)
  - [Diff Example](#diff-example)
  - [Diff normal format](#diff-normal-format)
  - [Diff context format](#diff-context-format)
  - [Diff universal format](#diff-universal-format)

------

# Materials

* [diff output formats](https://www.slideshare.net/OhgyunAhn/diff-output-formats)
* [diff @ wikipedia](https://en.wikipedia.org/wiki/Diff)

# Basic

## Diff Example

```console
$ tree .
.
├── bar
│   ├── a.txt
│   └── ccc
│       └── c.txt
└── foo
    ├── a.txt
    └── bbb
        └── b.txt

4 directories, 4 files
```

## Diff normal format

* 변경된 줄만 출력한다.
* 변경된 곳의 주변을 파악하기 어렵다.

```console
$ diff -r foo bar

diff -r foo/a.txt bar/a.txt
2c2
< Good Morning
---
> Good Evening
Only in foo: bbb
Only in bar: ccc
```

형식은 다음과 같다. 

```
<변경 키워드>
< 원본 파일 라인
< 원본 파일 라인
---
> 새 파일 라인
> 새 파일 라인
```

`<변경 키워드>` 의 소문자 `a, d, c` 는 각각 `added, deleted, changed` 를 의미한다. 

다음은 `<변경 키워드>` 의 여러가지 예이다.

* `7a7,8` : 원본 파일 7 번째 줄에 새 파일의 7, 8 번째 줄이 추가되었다.
* `5,7c8,10` : 원본 파일 5, 7 번째 줄이 새 파일의 8, 10 번째 줄로 변경되었다.
* `5,7d3` : 원본 파일 5, 7 번째 줄이 삭제 되었고 이것은 새 파일의 3 번째 줄에 해당한다. 

## Diff context format

* 변경된 줄 전/후의 줄을 함께 출력한다.
* 변경된 곳의 주변을 확인할 수 있다.
* 주로 소스코드 배포할 때 사용한다.
* `-c <line-count>` 혹은 `--context[=line-count]` 옵션으로 출력할 주변 줄의 수를 설정할 수 있다. 기본값은 3 이다.

```console
$ diff -r -c foo/a.txt bar/a.txt

*** foo/a.txt	2020-07-02 13:16:28.000000000 +0900
--- bar/a.txt	2020-07-02 13:17:47.000000000 +0900
***************
*** 1,3 ****
  Hello World
! Good Morning
  Welcome Welcome
--- 1,3 ----
  Hello World
! Good Evening
  Welcome Welcome
Only in foo: bbb
Only in bar: ccc
```

형식은 다음과 같다. 

```
*** 원본 파일 수정시각
--- 새 파일 수정시각
***************
*** 원본 파일 범위 ****
[변경 키워드] 원본 파일 라인
--- 새 파일 범위 ---
[변경 키워드] 새 파일 라인
```

변경된 파일은 변경 키워드와 공백 이후에 출력된다. 변경되지 않은 라인은 2 개의 공백이후 출력된다.

`변경 키워드` 는 `!, +, -` 와 같다. 각각 `변경됨, 추가됨, 삭제됨` 을 의미한다.

## Diff universal format

* `-u line-count` 혹은 `--unified[=line-count]` 옵션으로 주변 줄 개수를 설정할 수 있다.
* context format 과 비슷하다. 변경된 부분의 주변 줄의 중복을 제거하고, 두 파일의 변화를 한번에 볼 수 있어 간략하다.

```console
$ diff -r -u foo bar

diff -r -u foo/a.txt bar/a.txt
--- foo/a.txt	2020-07-02 13:16:28.000000000 +0900
+++ bar/a.txt	2020-07-02 13:17:47.000000000 +0900
@@ -1,3 +1,3 @@
 Hello World
-Good Morning
+Good Evening
 Welcome Welcome
Only in foo: bbb
Only in bar: ccc
```

형식은 다음과 같다.

```
--- 원본파일 수정시각
+++ 새파일 수정시각
@@ -원본파일범위 +새파일범위 @@
[변경 키워드] 각 파일의 라인
```

출력되는 각 줄은 변경된 경우는 `변경 키워드` 로 시작하고, 변경되지 않은 경우 공백으로 시작한다. 변경 키워드는 `+, -` 와 같다. 각각 `추가됨, 삭제됨` 을 의미한다.
