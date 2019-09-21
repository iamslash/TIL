# Materials

* [progit](https://git-scm.com/book/ko/v2)
  * 킹왕짱 메뉴얼


# Git 도구

## 리비전 조회하기

```bash
## short hash
$ git show 1c002dd4b536e7479fe34593e72e6c6c1819e53b
$ git show 1c002dd4b536e7479f
$ git show 1c002d

## 짧고 중복되지 않는 해시 값
git log --abbrev-commit --pretty=oneline
# ca82a6d changed the version number
# 085bb3b removed unnecessary test code
# a11bef0 first commit

## 브랜치로 가리키기
# Git은 자동으로 브랜치와 HEAD가 지난 몇 달 동안에 가리켰었던 커밋을 모두 
# 기록하는데 이 로그를 “Reflog” 라고 부른다.
$ git show ca82a6dff817ec66f44342007202690a93763949
$ git show topic1
$ git rev-parse topic1
# ca82a6dff817ec66f44342007202690a93763949

## RefLog로 가리키기
$ git reflog
# 734713b HEAD@{0}: commit: fixed refs handling, added gc auto, updated
# d921970 HEAD@{1}: merge phedders/rdocs: Merge made by the 'recursive' strategy.
# 1c002dd HEAD@{2}: commit: added some blame and merge stuff
# 1c36188 HEAD@{3}: rebase -i (squash): updating HEAD
# 95df984 HEAD@{4}: commit: # This is a combination of two commits.
# 1c36188 HEAD@{5}: rebase -i (squash): updating HEAD
# 7e05da5 HEAD@{6}: rebase -i (pick): updating HEAD

#  HEAD가 5번 전에 가리켰던 것
$ git show HEAD@{5}

# 어제 날짜의 master 브랜치
$ git show master@{yesterday}

# git log -g 명령을 사용하면 git reflog 결과를 git log 명령과 같은 형태로 볼 수 있다.
$ git log -g master
# commit 734713bc047d87bf7eac9674765ae793478c50d3
# Reflog: master@{0} (Scott Chacon <schacon@gmail.com>)
# Reflog message: commit: fixed refs handling, added gc auto, updated
# Author: Scott Chacon <schacon@gmail.com>
# Date:   Fri Jan 2 18:32:33 2009 -0800

#     fixed refs handling, added gc auto, updated tests

# commit d921970aadf03b3cf0e71becdaab3147ba71cdef
# Reflog: master@{1} (Scott Chacon <schacon@gmail.com>)
# Reflog message: merge phedders/rdocs: Merge made by recursive.
# Author: Scott Chacon <schacon@gmail.com>
# Date:   Thu Dec 11 15:08:43 2008 -0800

#     Merge commit 'phedders/rdocs'

## 계통 관계로 가리키기

# 이름 끝에 ^ (캐럿) 기호를 붙이면 Git은 해당 커밋의 부모를 찾는다.
$ git log --pretty=format:'%h %s' --graph
# * 734713b fixed refs handling, added gc auto, updated tests
# *   d921970 Merge commit 'phedders/rdocs'
# |\
# | * 35cfb2b Some rdoc changes
# * | 1c002dd added some blame and merge stuff
# |/
# * 1c36188 ignore *.gem
# * 9b29157 add open3_detach to gemspec file list

$ git show HEAD^
# commit d921970aadf03b3cf0e71becdaab3147ba71cdef
# Merge: 1c002dd... 35cfb2b...
# Author: Scott Chacon <schacon@gmail.com>
# Date:   Thu Dec 11 15:08:43 2008 -0800

#     Merge commit 'phedders/rdocs'

# Windows 에서는 ^^ "*^" 을 사용한다.
$ git show HEAD^     # will NOT work on Windows
$ git show HEAD^^    # OK
$ git show "HEAD^"   # OK

# d921970^2 는 “d921970의 두 번째 부모” 를 의미한다. 
$ git show d921970^
# commit 1c002dd4b536e7479fe34593e72e6c6c1819e53b
# Author: Scott Chacon <schacon@gmail.com>
# Date:   Thu Dec 11 14:58:32 2008 -0800
#
#     added some blame and merge stuff

$ git show d921970^2
# commit 35cfb2b795a55793d7cc56a6cc2060b4bb732548
# Author: Paul Hedderly <paul+git@mjr.org>
# Date:   Wed Dec 10 22:22:03 2008 +0000
#
#     Some rdoc changes

# HEAD~2 는 명령을 실행할 시점의 “첫 번째 부모의 첫 번째 부모” , 즉 “조부모” 를 가리킨다. 
$ git show HEAD~3
# commit 1c3618887afb5fbcbea25b7c013f4e2114448b8d
# Author: Tom Preston-Werner <tom@mojombo.com>
# Date:   Fri Nov 7 13:47:59 2008 -0500
#
#     ignore *.gem

# HEAD~2 와 HEAD^^^ 는 같다.
$ git show HEAD^^^
# commit 1c3618887afb5fbcbea25b7c013f4e2114448b8d
# Author: Tom Preston-Werner <tom@mojombo.com>
# Date:   Fri Nov 7 13:47:59 2008 -0500
#
#     ignore *.gem

### 범위로 커밋 가리키기

# A <- B <- E <- F -< master
#       \
#        <- C <- D -< experiment

## Double Dot
# experiment 브랜치의 커밋들 중에서 아직 master 브랜치에 Merge 
# 하지 않은 것들만 보고 싶으면 master..experiment 라고 사용한다. 
# 이 표현은 “master에는 없지만, experiment에는 있는 커밋” 을 의미

$ git log master..experiment
# C
# D

$ git log experiment..master
# F
# E

# origin 저장소의 master 브랜치에는 없고 현재 Checkout 중인 브랜치에만 있는 커밋
$ git log origin/master..HEAD

## 세 개 이상의 Refs

# ^ 이나 --not 옵션 뒤에 브랜치 이름은 그 브랜치에 없는 커밋
# 아래는 모두 같은 표현이다.
$ git log refA..refB
$ git log ^refA refB
$ git log refB --not refA

# 세개 이상의 브랜치에 적용
$ git log refA refB ^refC
$ git log refA refB --not refC

## Triple Dot
# Triple Dot은 양쪽에 있는 두 Refs 사이에서 공통으로 가지는 것을 제외하고 서로 다른 커밋
$ git log master...experiment
# F
# E
# D
# C

$ git log --left-right master...experiment
# < F
# < E
# > D
# > C
```

## 대화형 명령

```bash
# 대화형 모드 진입
$ git add -i
#            staged     unstaged path
#   1:    unchanged        +0/-1 TODO
#   2:    unchanged        +1/-1 index.html
#   3:    unchanged        +5/-1 lib/simplegit.rb

# *** Commands ***
#   1: status     2: update      3: revert     4: add untracked
#   5: patch      6: diff        7: quit       8: help
What now>

# Staging Area에 파일 추가하고 추가 취소하기
What now> 2
#            staged     unstaged path
#   1:    unchanged        +0/-1 TODO
#   2:    unchanged        +1/-1 index.html
#   3:    unchanged        +5/-1 lib/simplegit.rb
Update>> 1,2
#            staged     unstaged path
# * 1:    unchanged        +0/-1 TODO
# * 2:    unchanged        +1/-1 index.html
#   3:    unchanged        +5/-1 lib/simplegit.rb
Update>> [enter]
# updated 2 paths

# *** Commands ***
#   1: status     2: update      3: revert     4: add untracked
#   5: patch      6: diff        7: quit       8: help
What now> 1
#            staged     unstaged path
#   1:        +0/-1      nothing TODO
#   2:        +1/-1      nothing index.html
#   3:    unchanged        +5/-1 lib/simplegit.rb
# *** Commands ***
#   1: status     2: update      3: revert     4: add untracked
#   5: patch      6: diff        7: quit       8: help
What now> 3
#            staged     unstaged path
#   1:        +0/-1      nothing TODO
#   2:        +1/-1      nothing index.html
#   3:    unchanged        +5/-1 lib/simplegit.rb
Revert>> 1
#            staged     unstaged path
# * 1:        +0/-1      nothing TODO
#   2:        +1/-1      nothing index.html
#   3:    unchanged        +5/-1 lib/simplegit.rb
Revert>> [enter]
# *** Commands ***
#   1: status     2: update      3: revert     4: add untracked
#   5: patch      6: diff        7: quit       8: help
What now> 1
#            staged     unstaged path
#   1:    unchanged        +0/-1 TODO
#   2:        +1/-1      nothing index.html
#   3:    unchanged        +5/-1 lib/simplegit.rb
# *** Commands ***
#   1: status     2: update      3: revert     4: add untracked
#   5: patch      6: diff        7: quit       8: help
What now> 6
#            staged     unstaged path
#   1:        +1/-1      nothing index.html
Review diff>> 1
# diff --git a/index.html b/index.html
# index 4d07108..4335f49 100644
# --- a/index.html
# +++ b/index.html
# @@ -16,7 +16,7 @@ Date Finder

#  <p id="out">...</p>

# -<div id="footer">contact : support@github.com</div>
# +<div id="footer">contact : email.support@github.com</div>

#  <script type="text/javascript">

## 파일의 일부분만 Staging Area에 추가하기

# *** Commands ***
#   1: status     2: update      3: revert     4: add untracked
#   5: patch      6: diff        7: quit       8: help
What now> 5
# diff --git a/lib/simplegit.rb b/lib/simplegit.rb
# index dd5ecc4..57399e0 100644
# --- a/lib/simplegit.rb
# +++ b/lib/simplegit.rb
# @@ -22,7 +22,7 @@ class SimpleGit
#    end

#    def log(treeish = 'master')
# -    command("git log -n 25 #{treeish}")
# +    command("git log -n 30 #{treeish}")
#    end

#    def blame(path)
Stage this hunk [y,n,a,d,/,j,J,g,e,?]? ?
# y - stage this hunk
# n - do not stage this hunk
# a - stage this and all the remaining hunks in the file
# d - do not stage this hunk nor any of the remaining hunks in the file
# g - select a hunk to go to
# / - search for a hunk matching the given regex
# j - leave this hunk undecided, see next undecided hunk
# J - leave this hunk undecided, see next hunk
# k - leave this hunk undecided, see previous undecided hunk
# K - leave this hunk undecided, see previous hunk
# s - split the current hunk into smaller hunks
# e - manually edit the current hunk
# ? - print help
What now> 1
#            staged     unstaged path
#   1:    unchanged        +0/-1 TODO
#   2:        +1/-1      nothing index.html
#   3:        +1/-1        +4/-0 lib/simplegit.rb
```

## Stashing과 Cleaning
## 내 작업에 서명하기
## 검색
## 히스토리 단장하기
## Reset 명확히 알고 가기
## 고급 Merge
## Rerere
## Git으로 버그 찾기
## 서브모듈
## Bundle
## Replace
## Credential 저장소

# Git맞춤

## Git 설정하기
## Git Attributes
## Git Hooks
## 정책 구현하기

# Git과 여타 버전 관리 시스템
## Git: 범용 Client
## Git으로 옮기기

# Git의 내부
## Plumbing 명령과 Porcelain 명령
## Git 개체
## Git Refs
## Packfile
## Refspec
## 데이터 전송 프로토콜
## 운영 및 데이터 복구
## 환경변수