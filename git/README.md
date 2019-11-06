- [Materials](#materials)
- [Git Overview](#git-overview)
- [Git 설정](#git-%ec%84%a4%ec%a0%95)
- [Git Basic](#git-basic)
- [`.gitignore`](#gitignore)
- [Git log](#git-log)
- [Git edit commit](#git-edit-commit)
- [Git revert](#git-revert)
- [Git reset](#git-reset)
- [Git rebase, merge](#git-rebase-merge)
- [Git cherrypick](#git-cherrypick)
- [Git merge --squash](#git-merge---squash)
- [Git rebase -i](#git-rebase--i)
- [Git stash](#git-stash)
- [Git Clean](#git-clean)
- [Git tag](#git-tag)
- [Git grep](#git-grep)
- [Git filter-branch](#git-filter-branch)
- [Git merge](#git-merge)
- [Git LFS](#git-lfs)
- [Advanced](#advanced)
  - [내 작업에 서명하기](#%eb%82%b4-%ec%9e%91%ec%97%85%ec%97%90-%ec%84%9c%eb%aa%85%ed%95%98%ea%b8%b0)
  - [고급 Merge](#%ea%b3%a0%ea%b8%89-merge)
  - [Rerere](#rerere)
  - [Git으로 버그 찾기](#git%ec%9c%bc%eb%a1%9c-%eb%b2%84%ea%b7%b8-%ec%b0%be%ea%b8%b0)
  - [서브모듈](#%ec%84%9c%eb%b8%8c%eb%aa%a8%eb%93%88)
- [Git Tips](#git-tips)
  - [use cat instead of pager](#use-cat-instead-of-pager)
  - [git diff output](#git-diff-output)
  - [git diff](#git-diff)
  - [git blame](#git-blame)

# Materials

* [누구나 쉽게 이해할 수 있는 Git입문](https://backlog.com/git-tutorial/kr/)
  * 킹왕짱 튜토리얼
* [progit](https://git-scm.com/book/ko/v2)
  * 킹왕짱 메뉴얼
* `file:///C:/Program%20Files/Git/mingw64/share/doc/git-doc/`
  * git documents

# Git Overview

git 의 세계관은 `working directory, Index(staging area), local repository, origin remote repository, upstream remote repository` 와 같이 5 가지 영역을 갖는다.

![](img/reset-workflow.png)

[git branch strategy](gitbranchstrategy.md) 를 참고하여 branch 전략을 운영하자.

`HEAD` 는 현재 속해있는 branch 의 끝을 가리킨다. `ORIG_HEAD` 는 바로 이전의 `HEAD` 이다. `HEAD~` 는 `HEAD` 의 부모 `commit` 이다. `HEAD~2` 는 `HEAD` 의 부모의 부모 `commit` 이다. `HEAD^` 는 뭐지??? 또한 `HEAD@{15}` 과 같은 형태로 부모 commit 을 가리킬 수 있다. `MERGE_HEAD` 는 뭐지???

stash 역시 `STASH@{0}` 와 같은 형태로 특정 stash 를 가리킬 수 있다.

# Git 설정

```bash
## set name, email
$ git config --global user.name "David Sun"
$ git config --global user.email iamslash@gmail.com

## set editor
$ git config --global core.editor vim

## set commit message template
# vim $HOME/.gitmsg.txt
# 
# [AA]
$ git config --global commit.template $HOME/.gitmsg.txt

## set pager
$ git config --global core.pager less

## set mergetool
$ git config --global merge.tool p4mergetool
$ git config --global mergetool.p4mergetool.cmd \
"/Applications/p4merge.app/Contents/Resources/launchp4merge \$PWD/\$BASE \$PWD/\$REMOTE \$PWD/\$LOCAL \$PWD/\$MERGED"
$ git config --global mergetool.p4mergetool.trustExitCode false
$ git config --global mergetool.keepBackup false
$ git mergetool

## set difftool
$ git config --global diff.tool p4mergetool
$ git config --global difftool.p4mergetool.cmd \
"/Applications/p4merge.app/Contents/Resources/launchp4merge \$LOCAL \$REMOTE"
$ git difftool
```

# Git Basic

```bash
# make git repository with barebone
$ git init . --bare

# -10 : 10 counts of logs
# --oneline : just one line of log
# --graph : show me graph
# --decorate : with tags
# --all : all branches
$ git log -10 --oneline --graph --decorate --all

# -s : simple
$ git status -s

# list local branches
$ git branch
# list remote branches
$ git branch -r
# list branches all including remotes
$ git branch -a
# delete a branch
$ git branch -d feature/AddA
# checkout develop branch with tracking origin/develop
$ git checkout -b develop origin/develop
# same as above but after Git 1.6.2
$ git checkout -t origin/develop
# make an existing branch track a remote branch
$ git branch --set-upstream-to origin/develop

$ git commit -am "update blah blah"
$ git push origin develop
$ git push upstream develop

# compare working dir with local repo
$ git diff
# compare index with local repo
$ git diff --staged
$ git diff --cached

# reset to HEAD without work changes, index
$ git reset --hard HEAD

# list remote repositories
$ git remote -v
origin	git@github.com:iamslash/TIL.git (fetch)
origin	git@github.com:iamslash/TIL.git (push)
# add remote repositories
$ git remote add upstream git@github.com:davidsun/TIL.git
$ git remote -v
origin	git@github.com:iamslash/TIL.git (fetch)
origin	git@github.com:iamslash/TIL.git (push)
upstream	git@github.com:davidsun/TIL.git (fetch)
upstream	git@github.com:davidsun/TIL.git (push)

# merge using mergetool
$ $ git mergetool

# log of past HEAD
$ git reflog
...
b29fe09 (develop) HEAD@{23}: merge develop: Fast-forward
c8755e4 (tag: 0.2, upstream/master, origin/master) HEAD@{24}: checkout: moving from develop to master
b29fe09 (develop) HEAD@{25}: rebase finished: returning to refs/heads/develop
b29fe09 (develop) HEAD@{26}: rebase: added e, f
c8755e4 (tag: 0.2, upstream/master, origin/master) HEAD@{27}: rebase: checkout master
aea064d HEAD@{28}: commit: added e, f
690ef48 HEAD@{29}: reset: moving to ORIG_HEAD

# show specific HEAD
$ git show HEAD~4 -s --oneline --decorate
b29fe09 (develop) added e, f
$ git show HEAD@{28} -s --oneline --decorate
aea064d added e, f
$ git show master@{yesterday} -s --oneline --decorate

# get SHA1
$ git rev-parse master
d3b5f7e7a975b32dc39c7b12481710a032d3ae55

# current status
$ git log -10 --oneline --graph --decorate --all
* 6b068dd (HEAD -> feature/Foo) added j
* aee6c2c added i, x
* b6313df added h
* 2353cf5 added g
| * d3b5f7e (master) added g
|/
* b29fe09 (develop) added e, f
| *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
| |\
| |/
|/|

# log commits which are not in master but in feature/Foo
$ git log master..feature/Foo --oneline --graph
* 6b068dd (HEAD -> feature/Foo) added j
* aee6c2c added i, x
* b6313df added h
* 2353cf5 added g

$ git log feature/Foo..master --oneline --graph
* d3b5f7e (master) added g

# log commits which are not in origin/master but in checkouted branch
$ git log origin/master..HEAD --oneline
6b068dd (HEAD -> feature/Foo) added j
aee6c2c added i, x
b6313df added h
2353cf5 added g
b29fe09 (develop) added e, f

## 세 개 이상의 Refs

# these are same
$ git log refA..refB
$ git log ^refA refB
$ git log refB --not refA

# three branches
$ git log refA refB ^refC
$ git log refA refB --not refC

# log not common commits
$ git log master...feature/Foo --oneline
6b068dd (HEAD -> feature/Foo) added j
aee6c2c added i, x
b6313df added h
d3b5f7e (master) added g
2353cf5 added g

$ git log --left-right master...feature/Foo --oneline
git log --left-right master...feature/Foo --oneline
> 6b068dd (HEAD -> feature/Foo) added j
> aee6c2c added i, x
> b6313df added h
< d3b5f7e (master) added g
> 2353cf5 added g

```

# `.gitignore`

```
# 확장자가 .a인 파일 무시
*.a

# 윗 라인에서 확장자가 .a인 파일은 무시하게 했지만 lib.a는 무시하지 않음
!lib.a

# 현재 디렉토리에 있는 TODO파일은 무시하고 subdir/TODO처럼 하위디렉토리에 있는 파일은 무시하지 않음
/TODO

# build/ 디렉토리에 있는 모든 파일은 무시
build/

# doc/notes.txt 파일은 무시하고 doc/server/arch.txt 파일은 무시하지 않음
doc/*.txt

# doc 디렉토리 아래의 모든 .pdf 파일을 무시
doc/**/*.pdf
```

# Git log

```bash
# -p: with diff
$ git log -p -2
# --stat: with stat
$ git log --stat -2
# just oneline
$ git log --pretty=oneline -2
$ git log --oneline -2
# formatted print
$ git log --pretty=format:"%h - %an, %ar : %s"

# -S : search logs???
$ git log -S ZLIB_BUF_MAX --oneline
# -L : search line logs???
$ git log -L :git_deflate_bound:zlib.c
```

| 옵션 | 설명 |
|------|------|
| -p | 각 커밋에 적용된 패치를 보여준다. |
| --stat | 각 커밋에서 수정된 파일의 통계정보를 보여준다. |
| --shortstat | --stat 명령의 결과 중에서 수정한 파일, 추가된 라인, 삭제된 라인만 보여준다. |
| --name-only | 커밋 정보중에서 수정된 파일의 목록만 보여준다. |
| --name-status | 수정된 파일의 목록을 보여줄 뿐만 아니라 파일을 추가한 것인지, 수정한 것인지, 삭제한 것인지도 보여준다. |
| --abbrev-commit | 40자 짜리 SHA-1 체크섬을 전부 보여주는 것이 아니라 처음 몇 자만 보여준다. |
| --relative-date | 정확한 시간을 보여주는 것이 아니라 “2 weeks ago” 처럼 상대적인 형식으로 보여준다. |
| --graph | 브랜치와 머지 히스토리 정보까지 아스키 그래프로 보여준다. |
| --pretty | 지정한 형식으로 보여준다. 이 옵션에는 oneline, short, full, fuller, format이 있다. format은 원하는 형식으로 출력하고자 할 때 사용한다. |
| --oneline | --pretty=oneline --abbrev-commit 두 옵션을 함께 사용한 것과 같다. |

# Git edit commit

can edit last commit.

```bash
$ git log -10 --oneline --graph --decorate --all
*   998d92d (HEAD -> develop, upstream/develop, origin/develop) Merge tag '0.2' into develop
|\
| *   c8755e4 (tag: 0.2, upstream/master, origin/master, master) Merge branch 'hotfix/0.2'
| |\
| | * 51ba540 fixed blah blah
| |/
* |   99f9fd9 Merge tag '0.1' into develop
|\ \
| |/
| *   721d99f (tag: 0.1) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

$ vim forgotten.txt
$ git add .
$ git commit --amend

$ git log -2 --oneline --graph --decorate --all
*   35b9d0b (HEAD -> develop) Merge tag '0.2' into develop
|\
| | *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
| | |\
| |/ /
|/| /
| |/
```
# Git revert

can revert last commit.

```bash
$ rm forgotten.txt
$ git add .
$ git commit -am "removed forgotten.txt"

$ git log -3 --oneline --graph --decorate --all
* 690ef48 (HEAD -> develop) removed forgotten.txt
*   35b9d0b Merge tag '0.2' into develop
|\
| | *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
| | |\
| |/ /
|/| /
| |/

$ git revert HEAD

$ git log -3 --oneline --graph --decorate --all
* 1399858 (HEAD -> develop) Revert "removed forgotten.txt"
* 690ef48 removed forgotten.txt
*   35b9d0b Merge tag '0.2' into develop
|\
```

# Git reset

| command | desc |
|--------|------|
| `git reset --soft` | reset with work changes and index |
| `git reset --mixed` | reset with work changes and unstaged |
| `git reset --hard` | reset with nothing |

```bash
$ rm forgotten.txt
$ git status -s
 D forgotten.txt

$ git reset --hard HEAD
HEAD is now at 1399858 Revert "removed forgotten.txt"
$ git status -s

$ git log -3 --oneline --graph --decorate --all
* 1399858 (HEAD -> develop) Revert "removed forgotten.txt"
* 690ef48 removed forgotten.txt
*   35b9d0b Merge tag '0.2' into develop
|\

$ git reset --hard HEAD~
HEAD is now at 690ef48 removed forgotten.txt

$ git log -3 --oneline --graph --decorate --all
* 690ef48 (HEAD -> develop) removed forgotten.txt
*   35b9d0b Merge tag '0.2' into develop
|\
| | *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
| | |\
| |/ /
|/| /
| |/

$ vim e.md
$ git add .
$ vim f.md
$ git status -s
A  e.md
?? f.md

$ git reset --soft HEAD
$ git status -s
A  e.md
?? f.md

$ git reset --mixed HEAD
$ git status -s
?? e.md
?? f.md
```

# Git rebase, merge

```bash
$ git log -10 --oneline --graph --decorate --all
* 690ef48 (HEAD -> develop) removed forgotten.txt
*   35b9d0b Merge tag '0.2' into develop
|\
| | *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
| | |\
| |/ /
|/| /
| |/
| *   c8755e4 (tag: 0.2, upstream/master, origin/master, master) Merge branch 'hotfix/0.2'
| |\
| | * 51ba540 fixed blah blah
| |/
* |   99f9fd9 Merge tag '0.1' into develop
|\ \
| |/
| *   721d99f (tag: 0.1) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

$ git rebase master
First, rewinding head to replay your work on top of it...
Applying: removed forgotten.txt
Using index info to reconstruct a base tree...
A       forgotten.txt
Falling back to patching base and 3-way merge...
No changes -- Patch already applied.

$ git log -10 --oneline --graph --decorate --all
*   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
|\
| *   c8755e4 (HEAD -> develop, tag: 0.2, upstream/master, origin/master, master) Merge branch 'hotfix/0.2'
| |\
| | * 51ba540 fixed blah blah
| |/
* |   99f9fd9 Merge tag '0.1' into develop
|\ \
| |/
| *   721d99f (tag: 0.1) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

$ git reset --hard ORIG_HEAD
HEAD is now at 690ef48 removed forgotten.txt

$ git add .
$ git commit -am "added e, f"
$ git log -10 --oneline --graph --decorate --all
* aea064d (HEAD -> develop) added e, f
* 690ef48 removed forgotten.txt
*   35b9d0b Merge tag '0.2' into develop
|\
| | *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
| | |\
| |/ /
|/| /
| |/
| *   c8755e4 (tag: 0.2, upstream/master, origin/master, master) Merge branch 'hotfix/0.2'
| |\
| | * 51ba540 fixed blah blah
| |/
* |   99f9fd9 Merge tag '0.1' into develop
|\ \
| |/
| *   721d99f (tag: 0.1) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

$ git rebase master
First, rewinding head to replay your work on top of it...
Applying: removed forgotten.txt
Using index info to reconstruct a base tree...
A       forgotten.txt
Falling back to patching base and 3-way merge...
No changes -- Patch already applied.
Applying: added e, f

$ git log -10 --oneline --graph --decorate --all
* b29fe09 (HEAD -> develop) added e, f
| *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
| |\
| |/
|/|
* |   c8755e4 (tag: 0.2, upstream/master, origin/master, master) Merge branch 'hotfix/0.2'
|\ \
| * | 51ba540 fixed blah blah
|/ /
| *   99f9fd9 Merge tag '0.1' into develop
| |\
| |/
|/|
* |   721d99f (tag: 0.1) Merge branch 'release/0.1'
|\ \
| |/
| * 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

$ git checkout master
Switched to branch 'master'
Your branch is up to date with 'origin/master'.

$ git merge develop
Updating c8755e4..b29fe09
Fast-forward
 e.md | 1 +
 f.md | 1 +
 2 files changed, 2 insertions(+)
 create mode 100644 e.md
 create mode 100644 f.md

$ git log -10 --oneline --graph --decorate --all
* b29fe09 (HEAD -> master, develop) added e, f
| *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
| |\
| |/
|/|
* |   c8755e4 (tag: 0.2, upstream/master, origin/master) Merge branch 'hotfix/0.2'
|\ \
| * | 51ba540 fixed blah blah
|/ /
| *   99f9fd9 Merge tag '0.1' into develop
| |\
| |/
|/|
* |   721d99f (tag: 0.1) Merge branch 'release/0.1'
|\ \
| |/
| * 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off
```

# Git cherrypick

```bash
$ git checkout -b feature/Foo
$ vim g.md
$ git add .
$ git commit -am "added g"
$ git checkout master

$ git log -2 --oneline --graph --decorate --all
* 2353cf5 (feature/Foo) added g
* b29fe09 (HEAD -> master, develop) added e, f

$ git cherry-pick feature/Foo
$ git log -2 --oneline --graph --decorate --all
* d3b5f7e (HEAD -> master) added g
| * 2353cf5 (feature/Foo) added g
|/
```

# Git merge --squash

```bash
$ git checkout feature/Foo
$ vim h.md
$ git add .
$ git commit -am "added h"
$ vim i.md
$ git add .
$ git commit -am "added i"
$ vim j.md
$ git add .
$ git commit -am "added j"

$ git log -10 --oneline --graph --decorate --all
* e423161 (HEAD -> feature/Foo) added j
* e5abac2 added i
* b6313df added h
* 2353cf5 added g
| * d3b5f7e (master) added g
|/
* b29fe09 (develop) added e, f
| *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop

$ git checkout master
$ git merge --squash feature/Foo
Automatic merge went well; stopped before committing as requested
Squash commit -- not updating HEAD

$ git status -s
A  h.md
A  i.md
A  j.md

$ git commit
# Squashed commit of the following:

# commit e42316177b286cc97e3994dc24a1839fd15b86b0
# Author: iamslash <iamslash@gmail.com>
# Date:   Sat Nov 2 21:51:10 2019 +0900

#     added j

# commit e5abac27c3c91bb417375081c60fcffa3cf9c138
# Author: iamslash <iamslash@gmail.com>
# Date:   Sat Nov 2 21:50:54 2019 +0900

#     added i

# commit b6313dff15d8181239422c9867d0fd365acff74e
# Author: iamslash <iamslash@gmail.com>
# Date:   Sat Nov 2 21:50:37 2019 +0900

#     added h

# commit 2353cf57e958acc8b182d0e1004dbd8211b96226
# Author: iamslash <iamslash@gmail.com>
# Date:   Sat Nov 2 21:47:23 2019 +0900

#     added g

$ git log -10 --oneline --graph --decorate --all
* 76a238e (HEAD -> master) Squashed commit of the following:
* d3b5f7e added g
| * e423161 (feature/Foo) added j
| * e5abac2 added i
| * b6313df added h
| * 2353cf5 added g
|/
* b29fe09 (develop) added e, f
| *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop
| |\
| |/
|/|
* |   c8755e4 (tag: 0.2, upstream/master, origin/master) Merge branch 'hotfix/0.2'
|\ \
| * | 51ba540 fixed blah blah
|/ /
```

# Git rebase -i

make commits into one commit.

```bash
$ git reset --hard ORIG_HEAD
$ git log -10 --oneline --graph --decorate --all
* e423161 (feature/Foo) added j
* e5abac2 added i
* b6313df added h
* 2353cf5 added g
| * d3b5f7e (HEAD -> master) added g
|/
* b29fe09 (develop) added e, f
| *   998d92d (upstream/develop, origin/develop) Merge tag '0.2' into develop

$ git checkout feature/Foo
$ git log -5 --oneline --graph --decorate
* e423161 (HEAD -> feature/Foo) added j
* e5abac2 added i
* b6313df added h
* 2353cf5 added g
* b29fe09 (develop) added e, f

$ git show HEAD~4 -s --oneline
b29fe09 (develop) added e, f

$ git rebase -i HEAD~4
# pick 2353cf5 added g
# squash b6313df added h
# squash e5abac2 added i
# squash e423161 added j

$ git log -10 --oneline --graph --decorate --all
* c6906f8 (HEAD -> feature/Foo) added g
| * d3b5f7e (master) added g
|/
* b29fe09 (develop) added e, f

$ git log -1
commit c6906f8db77a8e44285910f8eba41b9a6432e337 (HEAD -> feature/Foo)
Author: iamslash <iamslash@gmail.com>
Date:   Sat Nov 2 21:47:23 2019 +0900

    added g

    added h

    added i

    added j

$ git reset --hard ORIG_HEAD
git log -10 --oneline --graph --decorate --all
* e423161 (HEAD -> feature/Foo) added j
* e5abac2 added i
* b6313df added h
* 2353cf5 added g
| * d3b5f7e (master) added g
|/
* b29fe09 (develop) added e, f

# modify commits
$ git rebase -i HEAD~2
# edit e5abac2 added i
# pick e423161 added j 
Stopped at e5abac2...  added i
You can amend the commit now, with

  git commit --amend

Once you are satisfied with your changes, run

  git rebase --continue

$ vim x.md
$ git add .
$ git commit --amend
[detached HEAD aee6c2c] added i, x
 Date: Sat Nov 2 21:50:54 2019 +0900
 2 files changed, 2 insertions(+)
 create mode 100644 i.md
 create mode 100644 x.md

$ git status
interactive rebase in progress; onto b6313df
Last command done (1 command done):
   edit e5abac2 added i
Next command to do (1 remaining command):
   pick e423161 added j
  (use "git rebase --edit-todo" to view and edit)
You are currently editing a commit while rebasing branch 'feature/Foo' on 'b6313df'.
  (use "git commit --amend" to amend the current commit)
  (use "git rebase --continue" once you are satisfied with your changes)

nothing to commit, working tree clean 

$ git log -10 --oneline --graph --decorate --all
* aee6c2c (HEAD) added i, x
| * e423161 (feature/Foo) added j
| * e5abac2 added i
|/
* b6313df added h
* 2353cf5 added g
| * d3b5f7e (master) added g

$ git rebase --continue

$ git log -10 --oneline --graph --decorate --all
* 6b068dd (HEAD -> feature/Foo) added j
* aee6c2c added i, x
* b6313df added h
* 2353cf5 added g
| * d3b5f7e (master) added g
|/
* b29fe09 (develop) added e, f
```

# Git stash

```bash
# save stash to top
$ git stash save
# save stash without index
$ git stash --keep-index
# ???
$ git stash -u
# stash interactively
$ git stash --patch

# list stash
$ git stash list

# load stash from top
$ git stash pop

# drop stash top
$ git stash drop
$ git stash drop stash@{0}

# clean all stashes
$ git stash clear

# let's apply stash with index.
$ git stash apply --index

# create branch with applying stash
$ git stash branch feature/Foo
```

# Git Clean

clean working directory,

```bash
$ vim y.md

# clean dryrun
$ git clean -d -n
Would remove y.md

# clean dryrun with .gitignore files
$ git clean -d -n -x
```

# Git tag

```bash
# list tags
$ git tag

# create lightweight tag
$ git tag 0.1

# create annotated tag
$ git tag -a 0.2

# remove tag
$ git tag -d 0.1
```

# Git grep

```bash
# -n or --line-number : print line numbers
$ git grep -n gmtime_r

# -c, --count : show counts 
$ git grep --count gmtime_r

# -p or --show-function : show functions matched with
$ git grep -p gmtime_r *.c

# --and : logical and
$ git grep --break --heading \
    -n -e '#define' --and \( -e LINK -e BUF_MAX \) v1.8.0
```

# Git filter-branch

can refine the history with `git filter-branch`.

```bash
# remote passwords.txt in the history
# --tree-filter : execute a argument and commit again
$ git filter-branch --tree-filter 'rm -f passwords.txt' HEAD
# Rewrite 6b9b3cf04e7c5686a9cb838c3f36a8cb6a0fc2bd (21/21)
# Ref 'refs/heads/master' was rewritten

# remove files with the pattern '*~ ' of all commits in the history.
$ git filter-branch --tree-filter 'rm -f *~' HEAD

# If you import from SVN you can trunk, tags, branch 
# Let's change SVN to root directory of all commits in the history.
$ git filter-branch --subdirectory-filter trunk HEAD
# Rewrite 856f0bf61e41a27326cdae8f09fe708d679f596f (12/12)
# Ref 'refs/heads/master' was rewritten

# modify emails of all commits in the history
$ git filter-branch --commit-filter '
        if [ "$GIT_AUTHOR_EMAIL" = "schacon@localhost" ];
        then
                GIT_AUTHOR_NAME="Scott Chacon";
                GIT_AUTHOR_EMAIL="schacon@example.com";
                git commit-tree "$@";
        else
                git commit-tree "$@";
        fi' HEAD
```

# Git merge

If it conflict just edit conflicted files and commit, push.

```bash
$ cd ~/tmp/L/david/HelloWorld
$ git log -10 --oneline --graph --decorate --all
*   7ead648 (HEAD -> develop, origin/develop) Merge branch 'develop' of D:/tmp/R/origin\HelloWorld into develop
|\
| *   998d92d (upstream/develop) Merge tag '0.2' into develop
| |\
| * \   99f9fd9 Merge tag '0.1' into develop
| |\ \
* | | | 190e701 added y.md
$ vim y.md
# sdiosdisdfjoiojsfdijo
#
# Hbllo Wbrld
$ git commit -am "update y.md"
$ git push origin develop

$ cd ~/tmp/L/peter/HelloWorld
git log -10 --oneline --graph --decorate --all
*   7ead648 (HEAD -> develop, origin/develop, master) Merge branch 'develop' of D:/tmp/R/origin\HelloWorld into develop
|\
| *   998d92d Merge tag '0.2' into develop
| |\
| * \   99f9fd9 Merge tag '0.1' into develop
| |\ \
* | | | 190e701 added y.md
* | | | b29fe09 added e, f
$ vim y.md
# sdiosdisdfjoiojsfdijo
#
# Hallo Warld
$ git commit -am "update y.md"
$ git push origin develop
To D:/tmp/R/origin\\HelloWorld
 ! [rejected]        develop -> develop (fetch first)
$ git pull origin develop
remote: Counting objects: 3, done.
remote: Total 3 (delta 0), reused 0 (delta 0)
Unpacking objects: 100% (3/3), done.
From D:/tmp/R/origin\\HelloWorld
 * branch            develop    -> FETCH_HEAD
   7ead648..5e197af  develop    -> origin/develop
Auto-merging y.md
CONFLICT (content): Merge conflict in y.md
Automatic merge failed; fix conflicts and then commit the result.
$ git status
On branch develop
Your branch and 'origin/develop' have diverged,
and have 1 and 1 different commits each, respectively.
  (use "git pull" to merge the remote branch into yours)

You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)

        both modified:   y.md

no changes added to commit (use "git add" and/or "git commit -a")
$ vim y.md
# sdiosdisdfjoiojsfdijo
#
# <<<<<<< HEAD
# Hallo Warld
# =======
# Hbllo Wbrld
# >>>>>>> 5e197af72259ed8e0f170a748341f03284cdb906
$ git commit -a
# Merge branch 'develop' of D:/tmp/R/origin\\HelloWorld into develop
$ git push origin develop
```

can merge with ignoring white spaces.

```bash
## 공백 무시하기
# 공백이 충돌의 전부라면 merge 를 취소하고 -Xignore-all-space 혹은 # -Xignore-space-change 를 추가하여 공백을 부시하고 merge 하자.
# -Xignore-all-space 는 모든 공백을 무시한다.
# -Xignore-space-change 는 여러공백을 하나로 취급한다.
# 스페이스를 탭으로 혹은 탭을 스페이스로 바꾸었을 때 유용하다
$ git merge -Xignore-space-change whitespace
Auto-merging hello.rb
Merge made by the 'recursive' strategy.
 hello.rb | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
``` 

can merge manually

```bash
# 충돌이 발생하면 index 에 3 가지 파일이 존재한다.
# Stage 1는 공통 조상 파일, Stage 2는 현재 개발자의 버전에 해당하는 파일, Stage 3은 MERGE_HEAD 가 가리키는 커밋의 파일이다.
# git show 를 이용해서 각 버전의 파일을 꺼낼 수 있다.
$ git show :1:hello.rb > hello.common.rb
$ git show :2:hello.rb > hello.ours.rb
$ git show :3:hello.rb > hello.theirs.rb
# ls-files -u 를 이용해서 Git blob 의 SHA-1 을 얻어오자.
# :1:hello.rb 는 Blob SHA-1 의 줄임말이다.
$ git ls-files -u
# 100755 ac51efdc3df4f4fd328d1a02ad05331d8e2c9111 1	hello.rb
# 100755 36c06c8752c78d2aff89571132f3bf7841a7b5c3 2	hello.rb
# 100755 e85207e04dfdd5eb0a1e9febbc67fd837c44a1cd 3	hello.rb

# 이제 working dir 에 3 가지 파일을 가져왔다. git merge-file 을
# 이용하여 merge 해보자.
$ dos2unix hello.theirs.rb
# dos2unix: converting file hello.theirs.rb to Unix format ...

$ git merge-file -p \
    hello.ours.rb hello.common.rb hello.theirs.rb > hello.rb

$ git diff -b
# diff --cc hello.rb
# index 36c06c8,e85207e..0000000
# --- a/hello.rb
# +++ b/hello.rb
# @@@ -1,8 -1,7 +1,8 @@@
#   #! /usr/bin/env ruby
#
#  +# prints out a greeting
#   def hello
# -   puts 'hello world'
# +   puts 'hello mundo'
#   end
#
#   hello()

# merge 후의 결과를 merge 하기 전의 브랜치와 비교
$ git diff --ours
# * Unmerged path hello.rb
# diff --git a/hello.rb b/hello.rb
# index 36c06c8..44d0a25 100755
# --- a/hello.rb
# +++ b/hello.rb
# @@ -2,7 +2,7 @@
#
#  # prints out a greeting
#  def hello
# -  puts 'hello world'
# +  puts 'hello mundo'
#  end
#
#  hello()

# merge 할 파일을 가져온 쪽과 비교. -b 를 이용하여 공백을 빼고 비교한다.
$ git diff --theirs -b
# * Unmerged path hello.rb
# diff --git a/hello.rb b/hello.rb
# index e85207e..44d0a25 100755
# --- a/hello.rb
# +++ b/hello.rb
# @@ -1,5 +1,6 @@
#  #! /usr/bin/env ruby
#
# +# prints out a greeting
#  def hello
#    puts 'hello mundo'
#  end

# 양쪽 모두와 비교
# -b 를 추가하여 공백은 무시하자.
$ git diff --base -b
# * Unmerged path hello.rb
# diff --git a/hello.rb b/hello.rb
# index ac51efd..44d0a25 100755
# --- a/hello.rb
# +++ b/hello.rb
# @@ -1,7 +1,8 @@
#  #! /usr/bin/env ruby
#
# +# prints out a greeting
#  def hello
# -  puts 'hello world'
# +  puts 'hello mundo'
#  end
#
#  hello()

# merge 를 완료했으니 필요없는 파일을 제거하자.
$ git clean -f
# Removing hello.common.rb
# Removing hello.ours.rb
# Removing hello.theirs.rb

## 충돌 파일 Checkout

# 이번에 서로다른 3개의 commit 을 갖는 branch 두 개가 있다.
$ git log --graph --oneline --decorate --all
# * f1270f7 (HEAD, master) update README
# * 9af9d3b add a README
# * 694971d update phrase to hola world
# | * e3eb223 (mundo) add more tests
# | * 7cff591 add testing script
# | * c3ffff1 changed text to hello mundo
# |/
# * b7dcc89 initial hello world code

# 충돌이 발생한다.
$ git merge mundo
# Auto-merging hello.rb
# CONFLICT (content): Merge conflict in hello.rb
# Automatic merge failed; fix conflicts and then commit the result.

# 다음은 hello.rb 의 충돌 내용이다.
# #! /usr/bin/env ruby
#
# def hello
# <<<<<<< HEAD
#   puts 'hola world'
# =======
#   puts 'hello mundo'
# >>>>>>> mundo
# end
#
# hello()

# --conflict 옵션에는 diff3 나 merge 를 넘길 수 있고 merge 가 기본 값이다. diff3 를 사용하면 “ours” 나 “theirs” 말고도 “base” 버전의 내용까지 제공한다.
$ git checkout --conflict=diff3 hello.rb
# #! /usr/bin/env ruby
#
# def hello
# <<<<<<< ours
#   puts 'hola world'
# ||||||| base
#   puts 'hello world'
# =======
#   puts 'hello mundo'
# >>>>>>> theirs
# end
#
# hello()

# 다음과 같이 global config 를 수정할 수도 있다.
$ git config --global merge.conflictstyle diff3

## Merge 로그

# Triple Dot 을 이용하여 양쪽 branch 의 모든 commit 을 얻어오자.
$ git log --oneline --left-right HEAD...MERGE_HEAD
# < f1270f7 update README
# < 9af9d3b add a README
# < 694971d update phrase to hola world
# > e3eb223 add more tests
# > 7cff591 add testing script
# > c3ffff1 changed text to hello mundo

# --merge 를 이용하여 충돌이 발생한 파일이 속한 커밋만 얻어오자.
$ git log --oneline --left-right --merge
# < 694971d update phrase to hola world
# > c3ffff1 changed text to hello mundo

## Combined Diff 형식

# merge 하다가 충돌이 났을 때 git diff 를 실행해보자.
# 이런 형식을 combined diff 라고 한다.
# 각 라인은 두개의 컬럼으로 구분할 수 있다.
# 첫번째 컬럼은 ours branch 와 working dir 의 차이
# 두번째 컬럼은 theirs branch 와 working dir 의 차이
$ git diff
# diff --cc hello.rb
# index 0399cd5,59727f0..0000000
# --- a/hello.rb
# +++ b/hello.rb
# @@@ -1,7 -1,7 +1,11 @@@
#   #! /usr/bin/env ruby
#
#   def hello
# ++<<<<<<< HEAD
#  +  puts 'hola world'
# ++=======
# +   puts 'hello mundo'
# ++>>>>>>> mundo
#   end
#
#   hello()

# 충돌을 해결하고 git diff 실행하자.
# merge 후에 무엇이 바뀌었는지 확인
$ vim hello.rb
$ git diff
# diff --cc hello.rb
# index 0399cd5,59727f0..0000000
# --- a/hello.rb
# +++ b/hello.rb
# @@@ -1,7 -1,7 +1,7 @@@
#   #! /usr/bin/env ruby
#
#   def hello
# -   puts 'hola world'
#  -  puts 'hello mundo'
# ++  puts 'hola mundo'
#   end
#
#   hello()

# merge 후에 무엇이 바뀌었는지 확인하기 위해
# git log -p 를 사용할 수도 있다.
$ git log --cc -p -1
# commit 14f41939956d80b9e17bb8721354c33f8d5b5a79
# Merge: f1270f7 e3eb223
# Author: Scott Chacon <schacon@gmail.com>
# Date:   Fri Sep 19 18:14:49 2014 +0200
#
#     Merge branch 'mundo'
#
#     Conflicts:
#         hello.rb
#
# diff --cc hello.rb
# index 0399cd5,59727f0..e1d0799
# --- a/hello.rb
# +++ b/hello.rb
# @@@ -1,7 -1,7 +1,7 @@@
#   #! /usr/bin/env ruby
#
#   def hello
# -   puts 'hola world'
#  -  puts 'hello mundo'
# ++  puts 'hola mundo'
#   end
#
#   hello()
```

# Git LFS

* [What is the advantage of git lfs? @ stackoverflow](https://stackoverflow.com/questions/35575400/what-is-the-advantage-of-git-lfs)
* [Git LFS @ bitbucket](https://www.atlassian.com/git/tutorials/git-lfs)
* [git-lfs @ github](https://help.github.com/en/github/managing-large-files/configuring-git-large-file-storage)

----

This is a git extension for very large file. It saves text pointers and binary file. It can improve the performance using seperated binary file downloaded lazily.

You can install on macOS `brew install git-lfs` and initialize `git lfs install` on the root directory of the repository. Next you can track big file with `git lfs track "*.png"`. This will add to `.gitattributes`. 

You need to focus no `~/.git/hooks/post-checkout,post-commit,post-merge,pre-push`. For example this is `pre-push`. The machine git-lfs was not installed can't push because of hooks.

```bash
!/bin/sh
command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting .git/hooks/pre-push.\n"; exit 2; }
git lfs pre-push "$@"
```

# Advanced

## 내 작업에 서명하기

```bash
## GPG 소개

# 설치된 개인키의 목록을 확인
$ gpg --list-keys
# /Users/schacon/.gnupg/pubring.gpg
# ---------------------------------
# pub   2048R/0A46826A 2014-06-04
# uid                  Scott Chacon (Git signing key) <schacon@gmail.com>
# sub   2048R/874529A9 2014-06-04

# 키를 만들자
$ gpg --gen-key

# 개인키가 이미 있다면 설정하자
$ git config --global user.signingkey 0A46826A

## 태그 서명하기
# -a 대신 -s 를 쓰자
$ git tag -s v1.5 -m 'my signed 1.5 tag'

# You need a passphrase to unlock the secret key for
# user: "Ben Straub <ben@straub.cc>"
# 2048-bit RSA key, ID 800430EB, created 2014-05-04

# tag 에 gpg 서명이 첨부되어 있다.
$ git show v1.5
# tag v1.5
# Tagger: Ben Straub <ben@straub.cc>
# Date:   Sat May 3 20:29:41 2014 -0700

# my signed 1.5 tag
# -----BEGIN PGP SIGNATURE-----
# Version: GnuPG v1

# iQEcBAABAgAGBQJTZbQlAAoJEF0+sviABDDrZbQH/09PfE51KPVPlanr6q1v4/Ut
# LQxfojUWiLQdg2ESJItkcuweYg+kc3HCyFejeDIBw9dpXt00rY26p05qrpnG+85b
# hM1/PswpPLuBSr+oCIDj5GMC2r2iEKsfv2fJbNW8iWAXVLoWZRF8B0MfqX/YTMbm
# ecorc4iXzQu7tupRihslbNkfvfciMnSDeSvzCpWAHl7h8Wj6hhqePmLm9lAYqnKp
# 8S5B/1SSQuEAjRZgI4IexpZoeKGVDptPHxLLS38fozsyi0QyDyzEgJxcJQVMXxVi
# RUysgqjcpT8+iQM1PblGfHR4XAhuOqN5Fx06PSaFZhqvWFezJ28/CLyX5q+oIVk=
# =EFTF
# -----END PGP SIGNATURE-----

# commit ca82a6dff817ec66f44342007202690a93763949
# Author: Scott Chacon <schacon@gee-mail.com>
# Date:   Mon Mar 17 21:52:11 2008 -0700

#     changed the version number

## tag 확인하기. 확인 작업을 하려면 서명한 사람의 GPG 공개키를 키 관리 시스템에 등록해두어야 한다.
$ git tag -v v1.4.2.1
# object 883653babd8ee7ea23e6a5c392bb739348b1eb61
# type commit
# tag v1.4.2.1
# tagger Junio C Hamano <junkio@cox.net> 1158138501 -0700

# GIT 1.4.2.1

# Minor fixes since 1.4.2, including git-mv and git-http with alternates.
# gpg: Signature made Wed Sep 13 02:08:25 2006 PDT using DSA key ID F3119B9A
# gpg: Good signature from "Junio C Hamano <junkio@cox.net>"
# gpg:                 aka "[jpeg image of size 1513]"
# Primary key fingerprint: 3565 2A26 2040 E066 C9A7  4A7D C0C6 D9A4 F311 9B9A

# 서명한 사람의 공개키가 없으면 다음과 같은 에러 메시지를 출력한다.
# gpg: Signature made Wed Sep 13 02:08:25 2006 PDT using DSA key ID F3119B9A
# gpg: Can't check signature: public key not found
# error: could not verify the tag 'v1.4.2.1'

## 커밋에 서명하기

# -S 를 추가하여 commit 에 서명해보자.
$ git commit -a -S -m 'signed commit'

# You need a passphrase to unlock the secret key for
# user: "Scott Chacon (Git signing key) <schacon@gmail.com>"
# 2048-bit RSA key, ID 0A46826A, created 2014-06-04

# [master 5c3386c] signed commit
#  4 files changed, 4 insertions(+), 24 deletions(-)
#  rewrite Rakefile (100%)
#  create mode 100644 lib/git.rb

# --show-signature 를 추가하여 서명을 확인 하자.
$ git log --show-signature -1
# commit 5c3386cf54bba0a33a32da706aa52bc0155503c2
# gpg: Signature made Wed Jun  4 19:49:17 2014 PDT using RSA key ID 0A46826A
# gpg: Good signature from "Scott Chacon (Git signing key) <schacon@gmail.com>"
# Author: Scott Chacon <schacon@gmail.com>
# Date:   Wed Jun 4 19:49:17 2014 -0700

#     signed commit

# git log 로 출력한 로그에서 커밋에 대한 서명 정보를 알려면 %G? 포맷을 이용한다.
$ git log --pretty="format:%h %G? %aN  %s"

# 5c3386c G Scott Chacon  signed commit
# ca82a6d N Scott Chacon  changed the version number
# 085bb3b N Scott Chacon  removed unnecessary test code
# a11bef0 N Scott Chacon  first commit

# --verify-signatures 를 추가하여 Merge 할 커밋 중 서명하지 않았거나 
# 신뢰할 수 없는 사람이 서명한 커밋이 있으면 Merge 되지 않는다.
$ git merge --verify-signatures non-verify
# fatal: Commit ab06180 does not have a GPG signature.

# Merge 할 커밋 전부가 신뢰할 수 있는 사람에 의해 서명된 커밋이면 
# 모든 서명을 출력하고 Merge를 수행한다.
$ git merge --verify-signatures signed-branch
# Commit 13ad65e has a good GPG signature by Scott Chacon (Git signing key) <schacon@gmail.com>
# Updating 5c3386c..13ad65e
# Fast-forward
#  README | 2 ++
#  1 file changed, 2 insertions(+)

# -S 를 추가하여 merge commit 을 서명해 보자.
$ git merge --verify-signatures -S  signed-branch
# Commit 13ad65e has a good GPG signature by Scott Chacon (Git signing key) <schacon@gmail.com>

# You need a passphrase to unlock the secret key for
# user: "Scott Chacon (Git signing key) <schacon@gmail.com>"
# 2048-bit RSA key, ID 0A46826A, created 2014-06-04

# Merge made by the 'recursive' strategy.
#  README | 2 ++
#  1 file changed, 2 insertions(+)

```

## 고급 Merge



```bash
### 다른 방식의 Merge

## Our/Their 선택하기

# 다시 hello.rg 로 돌아가서 충돌을 재현하자.
$ git merge mundo
# Auto-merging hello.rb
# CONFLICT (content): Merge conflict in hello.rb
# Resolved 'hello.rb' using previous resolution.
# Automatic merge failed; fix conflicts and then commit the result.

# -Xours 혹은 -Xtheirs 를 추가하여 충돌을 해결하자.
$ git merge -Xours mundo
# Auto-merging hello.rb
# Merge made by the 'recursive' strategy.
#  hello.rb | 2 +-
#  test.sh  | 2 ++
#  2 files changed, 3 insertions(+), 1 deletion(-)
#  create mode 100644 test.sh

# ???
$ git merge -s ours mundo
# Merge made by the 'ours' strategy.
$ git diff HEAD HEAD~

## 서브트리 Merge

# 다른 프로젝트를 내 프로젝트의 subtree 로 추가하자.
$ git remote add rack_remote https://github.com/rack/rack
$ git fetch rack_remote --no-tags
# warning: no common commits
# remote: Counting objects: 3184, done.
# remote: Compressing objects: 100% (1465/1465), done.
# remote: Total 3184 (delta 1952), reused 2770 (delta 1675)
# Receiving objects: 100% (3184/3184), 677.42 KiB | 4 KiB/s, done.
# Resolving deltas: 100% (1952/1952), done.
# From https://github.com/rack/rack
#  * [new branch]      build      -> rack_remote/build
#  * [new branch]      master     -> rack_remote/master
#  * [new branch]      rack-0.4   -> rack_remote/rack-0.4
#  * [new branch]      rack-0.9   -> rack_remote/rack-0.9
$ git checkout -b rack_branch rack_remote/master
# Branch rack_branch set up to track remote branch refs/remotes/rack_remote/master.
# Switched to a new branch "rack_branch"

# 두 프로젝트가 한 저장소에 있는 것처럼 보인다.
$ ls
# AUTHORS         KNOWN-ISSUES   Rakefile      contrib         lib
# COPYING         README         bin           example         test
$ git checkout master
# Switched to branch "master"
$ ls
# README

# rack_branch 를 master 의 하위 디렉토리로 만들어 보자.
$ git read-tree --prefix=rack/ -u rack_branch

# remote rack_branch 에서 변경된 내용을 적용하고 다시 
# master 로 merge 한다.
$ git checkout rack_branch
$ git pull
$ git checkout master
$ git merge --squash -s recursive -Xsubtree=rack rack_branch
# Squash commit -- not updating HEAD
# Automatic merge went well; stopped before committing as requested

# rack 하위 디렉토리와 rack_branch 의 차이
$ git diff-tree -p rack_branch

# rack 하위 디렉토리와 rack 프로젝트의 remote repo 의 master 의 차이 비교
$ git diff-tree -p rack_remote/master
```

## Rerere

rerere 는 "reuse recorded resolution" 이다. [7.9 Git 도구 - Rerere](https://git-scm.com/book/ko/v2/Git-%EB%8F%84%EA%B5%AC-Rerere) 의 그림을 참고해서 이해하자.

```bash
# rerere 를 활성화 하자.
$ git config --global rerere.enabled true

# 다음은 예로 사용할 hello.rb 이다.
# #! /usr/bin/env ruby
#
# def hello
#   puts 'hello world'
# end

# 충돌 발생
# rerere 기능 때문에 몇 가지 정보를 더 출력
$ git merge i18n-world
# Auto-merging hello.rb
# CONFLICT (content): Merge conflict in hello.rb
# Recorded preimage for 'hello.rb'
# Automatic merge failed; fix conflicts and then commit the result.

# 충돌난 파일 확인
$ git rerere status
# hello.rb

$ git rerere diff
# --- a/hello.rb
# +++ b/hello.rb
# @@ -1,11 +1,11 @@
#  #! /usr/bin/env ruby
#
#  def hello
# -<<<<<<<
# -  puts 'hello mundo'
# -=======
# +<<<<<<< HEAD
#    puts 'hola world'
# ->>>>>>>
# +=======
# +  puts 'hello mundo'
# +>>>>>>> i18n-world
#  end

# rerere 기능은 아니지만 ls-files -u 를 사용하여 이전/현재/대상
# 버전의 hash 를 확인
$ git ls-files -u
# 100644 39804c942a9c1f2c03dc7c5ebcd7f3e3a6b97519 1	hello.rb
# 100644 a440db6e8d1fd76ad438a49025a9ad9ce746f581 2	hello.rb
# 100644 54336ba847c3758ab604876419607e9443848474 3	hello.rb

# 충돌 해결후 rerere 가 기록할 내용 확인 
$ git rerere diff
# --- a/hello.rb
# +++ b/hello.rb
# @@ -1,11 +1,7 @@
#  #! /usr/bin/env ruby
#
#  def hello
# -<<<<<<<
# -  puts 'hello mundo'
# -=======
# -  puts 'hola world'
# ->>>>>>>
# +  puts 'hola mundo'
#  end

# 이제 commit 한다.
$ git add hello.rb
$ git commit
# Recorded resolution for 'hello.rb'.
# [master 68e16e5] Merge branch 'i18n'

# 이제 merge 를 되돌리고 rebase 해서 master 에 쌓아보자.
$ git reset --hard HEAD^
# HEAD is now at ad63f15 i18n the hello
$ git checkout i18n-world
# Switched to branch 'i18n-world'
$ git rebase master
# First, rewinding head to replay your work on top of it...
# Applying: i18n one word
# Using index info to reconstruct a base tree...
# Falling back to patching base and 3-way merge...
# Auto-merging hello.rb
# CONFLICT (content): Merge conflict in hello.rb
# Resolved 'hello.rb' using previous resolution.
# Failed to merge in the changes.
# Patch failed at 0001 i18n one word

# 다음은 rerere 로 merge 된 hello.rb 이다.
# #! /usr/bin/env ruby

# def hello
#   puts 'hola mundo'
# end

# 자동으로 충돌이 해결되었다.
$ git diff
# diff --cc hello.rb
# index a440db6,54336ba..0000000
# --- a/hello.rb
# +++ b/hello.rb
# @@@ -1,7 -1,7 +1,7 @@@
#   #! /usr/bin/env ruby

#   def hello
# -   puts 'hola world'
#  -  puts 'hello mundo'
# ++  puts 'hola mundo'
#   end

# 충돌이 발생한 시점의 상태로 파일 내용을 되돌리자.
$ git checkout --conflict=merge hello.rb
# $ cat hello.rb
# #! /usr/bin/env ruby
#
# def hello
# <<<<<<< ours
#   puts 'hola world'
# =======
#   puts 'hello mundo'
# >>>>>>> theirs
# end

# 총돌이 발생한 코드를 자동으로 다시 해결
$ git rerere
# Resolved 'hello.rb' using previous resolution.
# $ cat hello.rb
# #! /usr/bin/env ruby

# def hello
#   puts 'hola mundo'
# end

# 이제 rebase 한다.
$ git add hello.rb
$ git rebase --continue
# Applying: i18n one word

# 여러 번 Merge 하거나, Merge 커밋을 쌓지 않으면서도 토픽 브랜치를
# master 브랜치의 최신 내용으로 유지하거나, Rebase를 자주 한다면
# rerere 가 도움이 된다.
```

## Git으로 버그 찾기

```bash
# 파일 어노테이션(Blame)

$ git blame -L 69,82 Makefile
# b8b0618cf6fab (Cheng Renquan  2009-05-26 16:03:07 +0800 69) ifeq ("$(origin V)", "command line")
# b8b0618cf6fab (Cheng Renquan  2009-05-26 16:03:07 +0800 70)   KBUILD_VERBOSE = $(V)
# ^1da177e4c3f4 (Linus Torvalds 2005-04-16 15:20:36 -0700 71) endif
# ^1da177e4c3f4 (Linus Torvalds 2005-04-16 15:20:36 -0700 72) ifndef KBUILD_VERBOSE
# ^1da177e4c3f4 (Linus Torvalds 2005-04-16 15:20:36 -0700 73)   KBUILD_VERBOSE = 0
# ^1da177e4c3f4 (Linus Torvalds 2005-04-16 15:20:36 -0700 74) endif
# ^1da177e4c3f4 (Linus Torvalds 2005-04-16 15:20:36 -0700 75)
# 066b7ed955808 (Michal Marek   2014-07-04 14:29:30 +0200 76) ifeq ($(KBUILD_VERBOSE),1)
# 066b7ed955808 (Michal Marek   2014-07-04 14:29:30 +0200 77)   quiet =
# 066b7ed955808 (Michal Marek   2014-07-04 14:29:30 +0200 78)   Q =
# 066b7ed955808 (Michal Marek   2014-07-04 14:29:30 +0200 79) else
# 066b7ed955808 (Michal Marek   2014-07-04 14:29:30 +0200 80)   quiet=quiet_
# 066b7ed955808 (Michal Marek   2014-07-04 14:29:30 +0200 81)   Q = @
# 066b7ed955808 (Michal Marek   2014-07-04 14:29:30 +0200 82) endif

# -C 를 추가하여 GITServerHandler.m 을 여러 개의 파일로 리팩토링한 것을 찾아내자
$ git blame -C -L 141,153 GITPackUpload.m
# f344f58d GITServerHandler.m (Scott 2009-01-04 141)
# f344f58d GITServerHandler.m (Scott 2009-01-04 142) - (void) gatherObjectShasFromC
# f344f58d GITServerHandler.m (Scott 2009-01-04 143) {
# 70befddd GITServerHandler.m (Scott 2009-03-22 144)         //NSLog(@"GATHER COMMI
# ad11ac80 GITPackUpload.m    (Scott 2009-03-24 145)
# ad11ac80 GITPackUpload.m    (Scott 2009-03-24 146)         NSString *parentSha;
# ad11ac80 GITPackUpload.m    (Scott 2009-03-24 147)         GITCommit *commit = [g
# ad11ac80 GITPackUpload.m    (Scott 2009-03-24 148)
# ad11ac80 GITPackUpload.m    (Scott 2009-03-24 149)         //NSLog(@"GATHER COMMI
# ad11ac80 GITPackUpload.m    (Scott 2009-03-24 150)
# 56ef2caf GITServerHandler.m (Scott 2009-01-05 151)         if(commit) {
# 56ef2caf GITServerHandler.m (Scott 2009-01-05 152)                 [refDict setOb
# 56ef2caf GITServerHandler.m (Scott 2009-01-05 153)

## 이진 탐색

$ git bisect start
$ git bisect bad
$ git bisect good v1.0
# Bisecting: 6 revisions left to test after this
# [ecb6e1bc347ccecc5f9350d878ce677feb13d3b2] error handling on repo

$ git bisect good
# Bisecting: 3 revisions left to test after this
# [b047b02ea83310a70fd603dc8cd7a6cd13d15c04] secure this thing

# 발견했다. 표시하자.
$ git bisect bad
# Bisecting: 1 revisions left to test after this
# [f71ce38690acf49c1f3c9bea38e09d82a5ce6014] drop exceptions table

# $ 
git bisect good
# b047b02ea83310a70fd603dc8cd7a6cd13d15c04 is first bad commit
# commit b047b02ea83310a70fd603dc8cd7a6cd13d15c04
# Author: PJ Hyett <pjhyett@example.com>
# Date:   Tue Jan 27 14:48:32 2009 -0800
#
#     secure this thing
#
# :040000 040000 40ee3e7821b895e52c1695092db9bdc4c61d1730
# f24d3c6ebcfc639b1a3814550e62d60b8e68a8e4 M  config

# 찾았으니 HEAD 를 돌려놓자.
$ git bisect reset

# 프로젝트가 정상적으로 수행되면 0을 반환하고 문제가 있으면 1을 반환하는 스크립트를 만든다
$ git bisect start HEAD v1.0
$ git bisect run test-error.sh
```

## 서브모듈

```bash

## 서브모듈 시작하기

# 서브모듈 "DbConnect" 를 추가하자.
$ git submodule add https://github.com/chaconinc/DbConnector
# Cloning into 'DbConnector'...
# remote: Counting objects: 11, done.
# remote: Compressing objects: 100% (10/10), done.
# remote: Total 11 (delta 0), reused 11 (delta 0)
# Unpacking objects: 100% (11/11), done.
# Checking connectivity... done.

# .gitmodules 파일이 생성.
$ git status
# On branch master
# Your branch is up-to-date with 'origin/master'.
#
# Changes to be committed:
#   (use "git reset HEAD <file>..." to unstage)
#
#     new file:   .gitmodules
#     new file:   DbConnector

# 다음은 .gitmodules 파일의 내용이다.
# [submodule "DbConnector"]
#     path = DbConnector
#     url = https://github.com/chaconinc/DbConnector

# submodule 을 통째로 특별한 commit 으로 취급하낟.
$ git diff --cached DbConnector
# diff --git a/DbConnector b/DbConnector
# new file mode 160000
# index 0000000..c3f01dc
# --- /dev/null
# +++ b/DbConnector
# @@ -0,0 +1 @@
# +Subproject commit c3f01dc8862123d317dd46284b05b6892c7b29bc

# --submodule 을 추가하여 더 자세히 살펴보자.
$ git diff --cached --submodule
# diff --git a/.gitmodules b/.gitmodules
# new file mode 100644
# index 0000000..71fc376
# --- /dev/null
# +++ b/.gitmodules
# @@ -0,0 +1,3 @@
# +[submodule "DbConnector"]
# +       path = DbConnector
# +       url = https://github.com/chaconinc/DbConnector
# Submodule DbConnector 0000000...c3f01dc (new submodule)

# commit 하자.
# mode 160000 는 특별하다.
$ git commit -am 'added DbConnector module'
# [master fb9093c] added DbConnector module
#  2 files changed, 4 insertions(+)
#  create mode 100644 .gitmodules
#  create mode 160000 DbConnector

# push 하자.
$ git push origin master

## 서브모듈 포함한 프로젝트 Clone

# 서브모듈이 비어 있다.
$ git clone https://github.com/chaconinc/MainProject
# Cloning into 'MainProject'...
# remote: Counting objects: 14, done.
# remote: Compressing objects: 100% (13/13), done.
# remote: Total 14 (delta 1), reused 13 (delta 0)
# Unpacking objects: 100% (14/14), done.
# Checking connectivity... done.
$ cd MainProject
$ ls -la
# total 16
# drwxr-xr-x   9 schacon  staff  306 Sep 17 15:21 .
# drwxr-xr-x   7 schacon  staff  238 Sep 17 15:21 ..
# drwxr-xr-x  13 schacon  staff  442 Sep 17 15:21 .git
# -rw-r--r--   1 schacon  staff   92 Sep 17 15:21 .gitmodules
# drwxr-xr-x   2 schacon  staff   68 Sep 17 15:21 DbConnector
# -rw-r--r--   1 schacon  staff  756 Sep 17 15:21 Makefile
# drwxr-xr-x   3 schacon  staff  102 Sep 17 15:21 includes
# drwxr-xr-x   4 schacon  staff  136 Sep 17 15:21 scripts
# drwxr-xr-x   4 schacon  staff  136 Sep 17 15:21 src
$ cd DbConnector/
$ ls

# submodule init 을 추가하여 submodule 을 clone 하자.
$ git submodule init
# Submodule 'DbConnector' (https://github.com/chaconinc/DbConnector) registered for path 'DbConnector'
$ git submodule update
# Cloning into 'DbConnector'...
# remote: Counting objects: 11, done.
# remote: Compressing objects: 100% (10/10), done.
# remote: Total 11 (delta 0), reused 11 (delta 0)
# Unpacking objects: 100% (11/11), done.
# Checking connectivity... done.
# Submodule path 'DbConnector': checked out 'c3f01dc8862123d317dd46284b05b6892c7b29bc'

# --recurse-submodules 를 추가하면 간단히 submodule 을 포함하여
# clone 할 수 있다.
$ git clone --recurse-submodules https://github.com/chaconinc/MainProject
# Cloning into 'MainProject'...
# remote: Counting objects: 14, done.
# remote: Compressing objects: 100% (13/13), done.
# remote: Total 14 (delta 1), reused 13 (delta 0)
# Unpacking objects: 100% (14/14), done.
# Checking connectivity... done.
# Submodule 'DbConnector' (https://github.com/chaconinc/DbConnector) registered for path 'DbConnector'
# Cloning into 'DbConnector'...
# remote: Counting objects: 11, done.
# remote: Compressing objects: 100% (10/10), done.
# remote: Total 11 (delta 0), reused 11 (delta 0)
# Unpacking objects: 100% (11/11), done.
# Checking connectivity... done.
# Submodule path 'DbConnector': checked out 'c3f01dc8862123d317dd46284b05b6892c7b29bc'

### 서브모듈 포함한 프로젝트 작업

## 서브모듈 업데이트하기

# submodule 을 수정하지 않는 경우
# 단순히 submodule 을 fetch 하고 merge 한다.
$ git fetch
# From https://github.com/chaconinc/DbConnector
#    c3f01dc..d0354fc  master     -> origin/master
$ git merge origin/master
# Updating c3f01dc..d0354fc
# Fast-forward
#  scripts/connect.sh | 1 +
#  src/db.c           | 1 +
#  2 files changed, 2 insertions(+)

# git log 할 때 --submodule 를 사용하지 않고 submodule 의 로그를
# 보고 싶다면 diff.submodule 를 설정한다.
$ git config --global diff.submodule log
$ git diff
# Submodule DbConnector c3f01dc..d0354fc:
#   > more efficient db routine
#   > better connection routine

# 보다 간단히 submodule 을 최신화 하자.
$ git submodule update --remote DbConnector
# remote: Counting objects: 4, done.
# remote: Compressing objects: 100% (2/2), done.
# remote: Total 4 (delta 2), reused 4 (delta 2)
# Unpacking objects: 100% (4/4), done.
# From https://github.com/chaconinc/DbConnector
#    3f19983..d0354fc  master     -> origin/master
# Submodule path 'DbConnector': checked out 'd0354fc054692d3906c85c3af05ddce39a1c0644'

$ git config -f .gitmodules submodule.DbConnector.branch stable

$ git submodule update --remote
# remote: Counting objects: 4, done.
# remote: Compressing objects: 100% (2/2), done.
# remote: Total 4 (delta 2), reused 4 (delta 2)
# Unpacking objects: 100% (4/4), done.
# From https://github.com/chaconinc/DbConnector
#    27cf5d3..c87d55d  stable -> origin/stable
# Submodule path 'DbConnector': checked out 'c87d55d4c6d4b05ee34fbc8cb6f7bf4585ae6687'

$ git status
# On branch master
# Your branch is up-to-date with 'origin/master'.
#
# Changes not staged for commit:
#   (use "git add <file>..." to update what will be committed)
#   (use "git checkout -- <file>..." to discard changes in working directory)
#
#   modified:   .gitmodules
#   modified:   DbConnector (new commits)
#
# no changes added to commit (use "git add" and/or "git commit -a")

$ git config status.submodulesummary 1

$ git status
# On branch master
# Your branch is up-to-date with 'origin/master'.
#
# Changes not staged for commit:
#   (use "git add <file>..." to update what will be committed)
#   (use "git checkout -- <file>..." to discard changes in working directory)
#
#     modified:   .gitmodules
#     modified:   DbConnector (new commits)
#
# Submodules changed but not updated:
#
# * DbConnector c3f01dc...c87d55d (4):
#   > catch non-null terminated lines

$ git diff
# diff --git a/.gitmodules b/.gitmodules
# index 6fc0b3d..fd1cc29 100644
# --- a/.gitmodules
# +++ b/.gitmodules
# @@ -1,3 +1,4 @@
#  [submodule "DbConnector"]
#         path = DbConnector
#         url = https://github.com/chaconinc/DbConnector
# +       branch = stable
#  Submodule DbConnector c3f01dc..c87d55d:
#   > catch non-null terminated lines
#   > more robust error handling
#   > more efficient db routine
#   > better connection routine

$ git log -p --submodule
# commit 0a24cfc121a8a3c118e0105ae4ae4c00281cf7ae
# Author: Scott Chacon <schacon@gmail.com>
# Date:   Wed Sep 17 16:37:02 2014 +0200

#     updating DbConnector for bug fixes

# diff --git a/.gitmodules b/.gitmodules
# index 6fc0b3d..fd1cc29 100644
# --- a/.gitmodules
# +++ b/.gitmodules
# @@ -1,3 +1,4 @@
#  [submodule "DbConnector"]
#         path = DbConnector
#         url = https://github.com/chaconinc/DbConnector
# +       branch = stable
# Submodule DbConnector c3f01dc..c87d55d:
#   > catch non-null terminated lines
#   > more robust error handling
#   > more efficient db routine
#   > better connection routine

## 서브모듈 관리하기

# 서브모듈을 수정해보자. 서브모듈 디렉토리로 가서 브랜치를
# Checkout 하자.
$ git checkout stable
# Switched to branch 'stable'

# 서브모듈을 머지하자.
$ git submodule update --remote --merge
# remote: Counting objects: 4, done.
# remote: Compressing objects: 100% (2/2), done.
# remote: Total 4 (delta 2), reused 4 (delta 2)
# Unpacking objects: 100% (4/4), done.
# From https://github.com/chaconinc/DbConnector
#    c87d55d..92c7337  stable     -> origin/stable
# Updating c87d55d..92c7337
# Fast-forward
#  src/main.c | 1 +
#  1 file changed, 1 insertion(+)
# Submodule path 'DbConnector': merged in '92c7337b30ef9e0893e758dac2459d07362ab5ea'

# 이제 다른 사람이 DbConnector 서브모듈을 수정하고
# 우리가 DbConnector 를 수정했다.
$ cd DbConnector/
$ vim src/db.c
$ git commit -am 'unicode support'
# [stable f906e16] unicode support
#  1 file changed, 1 insertion(+)

$ git submodule update --remote --rebase
# First, rewinding head to replay your work on top of it...
# Applying: unicode support
# Submodule path 'DbConnector': rebased into '5d60ef9bbebf5a0c1c1050f242ceeb54ad58da94'

# --rebase 옵션이나 --merge 옵션을 지정하지 않으면 Git은 로컬 
# 변경사항을 무시하고 서버로부터 받은 해당 서브모듈의 버전으로 
# Reset을 하고 Detached HEAD 상태로 만든다.
$ git submodule update --remote
# Submodule path 'DbConnector': checked out '5d60ef9bbebf5a0c1c1050f242ceeb54ad58da94'

$ git submodule update --remote
# remote: Counting objects: 4, done.
# remote: Compressing objects: 100% (3/3), done.
# remote: Total 4 (delta 0), reused 4 (delta 0)
# Unpacking objects: 100% (4/4), done.
# From https://github.com/chaconinc/DbConnector
#    5d60ef9..c75e92a  stable     -> origin/stable
# error: Your local changes to the following files would be overwritten by checkout:
#     scripts/setup.sh
# Please, commit your changes or stash them before you can switch branches.
# Aborting
# Unable to checkout 'c75e92a2b3855c9e5b66f915308390d9db204aca' in submodule path 'DbConnector'

# 충돌이 발생하면 서브모듈 디렉토리로 가서 충돌을 해결한다.
$ git submodule update --remote --merge
# Auto-merging scripts/setup.sh
# CONFLICT (content): Merge conflict in scripts/setup.sh
# Recorded preimage for 'scripts/setup.sh'
# Automatic merge failed; fix conflicts and then commit the result.
# Unable to merge 'c75e92a2b3855c9e5b66f915308390d9db204aca' in submodule path 'DbConnector'

## 서브모듈 수정 사항 공유하기

# 서브모듈의 변경사항은 우리의 local repo 에만 있다.
# 이 상태에서 main repo 를 push 하면 안된다.
$ git diff
# Submodule DbConnector c87d55d..82d2ad3:
#   > Merge from origin/stable
#   > updated setup script
#   > unicode support
#   > remove unnecessary method
#   > add new option for conn pooling

# push 되지 않은 submodule 이 있는지 검사한다.
$ git push --recurse-submodules=check
# The following submodule paths contain changes that can
# not be found on any remote:
#   DbConnector
#
# Please try
#
#     git push --recurse-submodules=on-demand
#
# or cd to the path and use
#
#     git push
#
# to push them to a remote.

# push.recurseSubmodules 를 설정할 수도 있다.
$ git config push.recurseSubmodules check

# git 이 대신 push 를 하게 할 수도 있다.
$ git push --recurse-submodules=on-demand
# Pushing submodule 'DbConnector'
# Counting objects: 9, done.
# Delta compression using up to 8 threads.
# Compressing objects: 100% (8/8), done.
# Writing objects: 100% (9/9), 917 bytes | 0 bytes/s, done.
# Total 9 (delta 3), reused 0 (delta 0)
# To https://github.com/chaconinc/DbConnector
#    c75e92a..82d2ad3  stable -> stable
# Counting objects: 2, done.
# Delta compression using up to 8 threads.
# Compressing objects: 100% (2/2), done.
# Writing objects: 100% (2/2), 266 bytes | 0 bytes/s, done.
# Total 2 (delta 1), reused 0 (delta 0)
# To https://github.com/chaconinc/MainProject
#    3d6d338..9a377d1  master -> master


## 서브모듈 Merge 하기

$ git pull
# remote: Counting objects: 2, done.
# remote: Compressing objects: 100% (1/1), done.
# remote: Total 2 (delta 1), reused 2 (delta 1)
# Unpacking objects: 100% (2/2), done.
# From https://github.com/chaconinc/MainProject
#    9a377d1..eb974f8  master     -> origin/master
# Fetching submodule DbConnector
# warning: Failed to merge submodule DbConnector (merge following commits not found)
# Auto-merging DbConnector
# CONFLICT (submodule): Merge conflict in DbConnector
# Automatic merge failed; fix conflicts and then commit the result.

### 서브모듈 팁 ???
## 서브모듈 Foreach 여행???
## 유용한 Alias???
## 서브모듈 사용할 때 주의할 점들???
```


# Git Tips

## use cat instead of pager

pager 를 less 로 설정하면 `git diff` 의 출력을 페이지 단위로 확인이 가능하다. 그러나 `q` 를 선택하면 출력내용이 사라진다.
pager 를 cat 로 설정하면 `git diff` 의 출력은 사라지지 않는다.

```bash
git config --global core.pager cat
git config --global core.pager less
```

## git diff output

* [diff output formats](https://www.slideshare.net/OhgyunAhn/diff-output-formats)

----

diff 는 normal format, context format, unified format 과 같이 다양한 출력형식을 가지고 있다. `git diff` 의 출력형식은 unified format 이다.

```
--- 원파일 수정시각
+++ 새파일 수정시각
@@ -원파일범위 +새파일범위 @@
[변경 키워드] 파일의 라인
```

## git diff

```bash
# diff between working directory and index
git diff

# diff between index and repository
git diff --cached

# diff with local branches
git diff <branch name> <branch name>

# diff between local branch and remote branch
git diff <branch name> <origin/branch name>

# diff with commits
git diff <commit id> <commit id>

# ???
git diff <a>..<b>
```

## git blame

파일별 수정이력을 확인할 수 있다.

```bash
git blame a.py
```