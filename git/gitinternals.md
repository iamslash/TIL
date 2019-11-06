# Abstract

This explains Git under the hood.

# Materials

* [10.2 Git의 내부 - Git 개체 @ progit](https://git-scm.com/book/ko/v2/Git%EC%9D%98-%EB%82%B4%EB%B6%80-Git-%EA%B0%9C%EC%B2%B4)
* [Gerrit을 이용한 코드 리뷰 시스템 - Gerrit과 Git @ naverD2](https://d2.naver.com/helloworld/1859580)

# Directories

This is a directory structure of `.git`.

```
.
├── local
│   ├── david
│   │   └── HelloWorld
│   └── peter
│       └── HelloWorld
├── origin
│   ├── david
│   │   └── HelloWorld
│   └── peter
│       └── HelloWorld
└── upstream
    └── HelloWorld.git
        ├── hooks
        ├── info
        ├── objects
        │   ├── info
        │   └── pack
        └── refs
            ├── heads
            └── tags
```

# Git Objects

There are 3 kinds of Git objects, commit object, tree object, blob object.

```bash
$ cd local/david/HelloWorld

# There is no file yet.
$ find .git/objects -type f

# let's make a blob object.
$ echo 'test content' | git hash-object -w --stdin
d670460b4b4aece5915caf5c68d12f560a9fe3e4
$ tree -a
.
└── .git
    ├── HEAD
    ├── config
    ├── description
    ├── hooks
    │   ├── applypatch-msg.sample
    │   ├── commit-msg.sample
    │   ├── fsmonitor-watchman.sample
    │   ├── post-update.sample
    │   ├── pre-applypatch.sample
    │   ├── pre-commit.sample
    │   ├── pre-push.sample
    │   ├── pre-rebase.sample
    │   ├── pre-receive.sample
    │   ├── prepare-commit-msg.sample
    │   └── update.sample
    ├── info
    │   └── exclude
    ├── objects
    │   ├── d6
    │   │   └── 70460b4b4aece5915caf5c68d12f560a9fe3e4
    │   ├── info
    │   └── pack
    └── refs
        ├── heads
        └── tags

# cat the blob object.
$ git cat-file -p d670460b4b4aece5915caf5c68d12f560a9fe3e4
test content

# make a test.txt file.
$ echo 'version 1' > test.txt

# write test.txt blob object to objects directory.
$ git hash-object -w test.txt
83baae61804e65cc73a7201a7252750c76066a30

# modify the test.txt file.
$ echo 'version 2' > test.txt

# write test.txt blob object to bojects directory again.
$ git hash-object -w test.txt
1f7a7a472abf3dd9643fd615f6da379c4acb3e3a
$ tree -a .git/objects
 .git/objects
├── 1f
│   └── 7a7a472abf3dd9643fd615f6da379c4acb3e3a
├── 83
│   └── baae61804e65cc73a7201a7252750c76066a30
├── d6
│   └── 70460b4b4aece5915caf5c68d12f560a9fe3e4
├── info
└── pack

# let's get back to version 1
$ git cat-file -p 83baae61804e65cc73a7201a7252750c76066a30 > test.txt
$ cat test.txt
version 1

# let's get back to version 2
$ git cat-file -p 1f7a7a472abf3dd9643fd615f6da379c4acb3e3a > test.txt
$ cat test.txt
version 2

# what is the type of the object?
$ git cat-file -t 1f7a7a472abf3dd9643fd615f6da379c4acb3e3a
blob
```

# Tree objects

# 

