# Materials

* [progit](https://git-scm.com/book/ko/v2)
  * 킹왕짱 메뉴얼

# Git Overview

git 은 `working directory, staging area, local repository, remote repository` 와 같이 4 가지 영역을 관리한다.

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

```bash
# stashing 은 하던 것을 commit 하지 않고 잠시 보관해두는 것이다.
# 여러 세트를 보관할 수 있고 다시 복원할 수 있다.

## 하던 일을 stash 하기
$ git status
# Changes to be committed:
#   (use "git reset HEAD <file>..." to unstage)
#
#   modified:   index.html
#
# Changes not staged for commit:
#   (use "git add <file>..." to update what will be committed)
#   (use "git checkout -- <file>..." to discard changes in working directory)
#
#   modified:   lib/simplegit.rb

# stack 에 새로운 stash 가 생성된다.
$ git stash
# Saved working directory and index state \
#   "WIP on master: 049d078 added the index file"
# HEAD is now at 049d078 added the index file
# (To restore them type "git stash apply")

# working directory 는 깨끗해 졌다.
$ git status
# On branch master
# nothing to commit, working directory clean

# stash 를 확인하자.
$ git stash list
# stash@{0}: WIP on master: 049d078 added the index file
# stash@{1}: WIP on master: c264051 Revert "added file_size"
# stash@{2}: WIP on master: 21d80a5 added number to log

# 이제 stash 를 working directory 에 적용해 보자.
$ git stash apply
# On branch master
# Changes not staged for commit:
#   (use "git add <file>..." to update what will be committed)
#   (use "git checkout -- <file>..." to discard changes in working directory)
#
#   modified:   index.html
#   modified:   lib/simplegit.rb
#
# no changes added to commit (use "git add" and/or "git commit -a")

# --index 를 추가하여 staged 상태까지 적용해보자.
$ git stash apply --index
# On branch master
# Changes to be committed:
#   (use "git reset HEAD <file>..." to unstage)
#
#   modified:   index.html
#
# Changes not staged for commit:
#   (use "git add <file>..." to update what will be committed)
#   (use "git checkout -- <file>..." to discard changes in working directory)
#
#   modified:   lib/simplegit.rb

# 적용한 stash 는 버리자.
$ git stash list
# stash@{0}: WIP on master: 049d078 added the index file
# stash@{1}: WIP on master: c264051 Revert "added file_size"
# stash@{2}: WIP on master: 21d80a5 added number to log
$ git stash drop stash@{0}
# Dropped stash@{0} (364e91f3f268f0900bc3ee613f9f733e82aaed43)

## Stash 를 만드는 새로운 방법
$ git status -s
# M  index.html
#  M lib/simplegit.rb

# 이미 staging area 에 있는 파일은 stash 하지 말자.
$ git stash --keep-index
# Saved working directory and index state WIP on master: 1b65b17 added the index file
# HEAD is now at 1b65b17 added the index file
$ git status -s
# M  index.html

# --include-untracked, -u 를 추가하여 untracked files 도 stash 해보자.
$ git status -s
# M  index.html
#  M lib/simplegit.rb
# ?? new-file.txt

$ git stash -u
# Saved working directory and index state WIP on master: 1b65b17 added the index file
# HEAD is now at 1b65b17 added the index file

$ git status -s

# --patch 를 추가하여 interactive 하게 처리해 보자.
$ git stash --patch
# diff --git a/lib/simplegit.rb b/lib/simplegit.rb
# index 66d332e..8bb5674 100644
# --- a/lib/simplegit.rb
# +++ b/lib/simplegit.rb
# @@ -16,6 +16,10 @@ class SimpleGit
#          return `#{git_cmd} 2>&1`.chomp
#        end
#      end
# +
# +    def show(treeish = 'master')
# +      command("git show #{treeish}")
# +    end

#  end
#  test
Stash this hunk [y,n,q,a,d,/,e,?]? y

# Saved working directory and index state WIP on master: 1b65b17 added the index file

## Stash 를 적용한 브랜치 만들기

# branch 를 만들고 stash 를 복원해 준다.
$ git stash branch testchanges
# M index.html
# M lib/simplegit.rb
# Switched to a new branch 'testchanges'
# On branch testchanges
# Changes to be committed:
#   (use "git reset HEAD <file>..." to unstage)

#   modified:   index.html

# Changes not staged for commit:
#   (use "git add <file>..." to update what will be committed)
#   (use "git checkout -- <file>..." to discard changes in working directory)

#   modified:   lib/simplegit.rb

# Dropped refs/stash@{0} (29d385a81d163dfd45a452a2ce816487a6b8b014)

## 워킹 디렉토리 청소하기
# git clean 은 untracked files 를 모두 지운다.

# -n 을 추가하여 가상으로 실행해 보자.
$ git clean -d -n
# Would remove test.o
# Would remove tmp/

# .gitignore 에 등록된 파일은 지우지 않는다. -X 를 추가하여
# .gitignore 에 등록된 파일도 지우자.
$ git status -s
#  M lib/simplegit.rb
# ?? build.TMP
# ?? tmp/

$ git clean -n -d
# Would remove build.TMP
# Would remove tmp/

$ git clean -n -d -x
# Would remove build.TMP
# Would remove test.o
# Would remove tmp/

# -i 를 추가하여 interactive 하게 실행해보자.
$ git clean -x -i
# Would remove the following items:
#   build.TMP  test.o
# *** Commands ***
#     1: clean                2: filter by pattern    3: select by numbers    4: ask each             5: quit
#     6: help
What now>

```

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

## 검색

```bash
## Git Grep

# -n or --line-number 르 ㄹ추가하여 라인 번호도 출력한다.
$ git grep -n gmtime_r
# compat/gmtime.c:3:#undef gmtime_r
# compat/gmtime.c:8:      return git_gmtime_r(timep, &result);
# compat/gmtime.c:11:struct tm *git_gmtime_r(const time_t *timep, struct tm *result)
# compat/gmtime.c:16:     ret = gmtime_r(timep, result);
# compat/mingw.c:826:struct tm *gmtime_r(const time_t *timep, struct tm *result)
# compat/mingw.h:206:struct tm *gmtime_r(const time_t *timep, struct tm *result);
# date.c:482:             if (gmtime_r(&now, &now_tm))
# date.c:545:             if (gmtime_r(&time, tm)) {
# date.c:758:             /* gmtime_r() in match_digit() may have clobbered it */
# git-compat-util.h:1138:struct tm *git_gmtime_r(const time_t *, struct tm *);
# git-compat-util.h:1140:#define gmtime_r git_gmtime_r

# -c, --count 를 추가하여 몇개 찾았는지 표시해 보자.
$ git grep --count gmtime_r
# compat/gmtime.c:4
# compat/mingw.c:1
# compat/mingw.h:1
# date.c:3
# git-compat-util.h:2

# -p or --show-function 을 추가하여 매칭되는 라인이 있는 함수나 메서드를 찾아보자.
$ git grep -p gmtime_r *.c
# date.c=static int match_multi_number(timestamp_t num, char c, const char *date,
# date.c:         if (gmtime_r(&now, &now_tm))
# date.c=static int match_digit(const char *date, struct tm *tm, int *offset, int *tm_gmt)
# date.c:         if (gmtime_r(&time, tm)) {
# date.c=int parse_date_basic(const char *date, timestamp_t *timestamp, int *offset)
# date.c:         /* gmtime_r() in match_digit() may have clobbered it */

# --and 를 추가하여 logical and 를 해보자.
$ git grep --break --heading \
    -n -e '#define' --and \( -e LINK -e BUF_MAX \) v1.8.0
# v1.8.0:builtin/index-pack.c
# 62:#define FLAG_LINK (1u<<20)

# v1.8.0:cache.h
# 73:#define S_IFGITLINK  0160000
# 74:#define S_ISGITLINK(m)       (((m) & S_IFMT) == S_IFGITLINK)

# v1.8.0:environment.c
# 54:#define OBJECT_CREATION_MODE OBJECT_CREATION_USES_HARDLINKS

# v1.8.0:strbuf.c
# 326:#define STRBUF_MAXLINK (2*PATH_MAX)

# v1.8.0:symlinks.c
# 53:#define FL_SYMLINK  (1 << 2)

# v1.8.0:zlib.c
# 30:/* #define ZLIB_BUF_MAX ((uInt)-1) */
# 31:#define ZLIB_BUF_MAX ((uInt) 1024 * 1024 * 1024) /* 1GB */

## Git 로그 검색

# -S 를 추가하여 log 에서 검색해보자.
$ git log -S ZLIB_BUF_MAX --oneline
# e01503b zlib: allow feeding more than 4GB in one go
# ef49a7a zlib: zlib can only process 4GB at a time

## 라인 로그 검색
$ git log -L :git_deflate_bound:zlib.c
# commit ef49a7a0126d64359c974b4b3b71d7ad42ee3bca
# Author: Junio C Hamano <gitster@pobox.com>
# Date:   Fri Jun 10 11:52:15 2011 -0700

#     zlib: zlib can only process 4GB at a time

# diff --git a/zlib.c b/zlib.c
# --- a/zlib.c
# +++ b/zlib.c
# @@ -85,5 +130,5 @@
# -unsigned long git_deflate_bound(z_streamp strm, unsigned long size)
# +unsigned long git_deflate_bound(git_zstream *strm, unsigned long size)
#  {
# -       return deflateBound(strm, size);
# +       return deflateBound(&strm->z, size);
#  }


# commit 225a6f1068f71723a910e8565db4e252b3ca21fa
# Author: Junio C Hamano <gitster@pobox.com>
# Date:   Fri Jun 10 11:18:17 2011 -0700

#     zlib: wrap deflateBound() too

# diff --git a/zlib.c b/zlib.c
# --- a/zlib.c
# +++ b/zlib.c
# @@ -81,0 +85,5 @@
# +unsigned long git_deflate_bound(z_streamp strm, unsigned long size)
# +{
# +       return deflateBound(strm, size);
# +}
# +
```

## 히스토리 단장하기

```bash
## 마지막 커밋을 수정하기

# 마지막 commit 메시지만 수정
# changes not staged 는 commit 되지 않는다.
$ git commit --amend

# changes not staged 와 함께 마지막 메시지를 수정
# SHA-1 값이 바뀐다. 물론 git reflog 로 메지만 바뀐 
# commit 을 추적할 수 있다.
$ git add .
$ git commit --amend

# commit 메시지 편집없이 마지막 commit 수정
$ git commit --amend --no-edit

## 커밋 메시지를 여러 개 수정하기

# 마지막 3 개의 메시지를 interactive 하게 수정
$ git rebase -i HEAD~3

pick f7f3f6d changed my name a bit
pick 310154e updated README formatting and added blame
pick a5f4a0d added cat-file

# Rebase 710f0f8..a5f4a0d onto 710f0f8
#
# Commands:
#  p, pick = use commit
#  r, reword = use commit, but edit the commit message
#  e, edit = use commit, but stop for amending
#  s, squash = use commit, but meld into previous commit
#  f, fixup = like "squash", but discard this commit's log message
#  x, exec = run command (the rest of the line) using shell
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
# Note that empty commits are commented out

# 위의 명령은 log 와 순서가 반대이다.
$ git log --pretty=format:"%h %s" HEAD~3..HEAD
# a5f4a0d added cat-file
# 310154e updated README formatting and added blame
# f7f3f6d changed my name a bit

## 커밋 순서 바꾸기

# commit message 를 아래와 같이 바꾸어 보자.
$ git rebase -i HEAD~3
# pick f7f3f6d changed my name a bit
# pick 310154e updated README formatting and added blame
# pick a5f4a0d added cat-file

```

## Reset 명확히 알고 가기

```bash
```

## 고급 Merge

```bash
```

## Rerere

```bash
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
```

## Bundle

```bash
```

## Replace

```bash
```

## Credential 저장소

```bash
```

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