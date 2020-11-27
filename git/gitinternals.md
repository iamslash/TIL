
- [Abstract](#abstract)
- [Materials](#materials)
- [Directories](#directories)
- [Git Objects](#git-objects)
  - [Tree objects](#tree-objects)
  - [Commit Objects](#commit-objects)
- [Git References](#git-references)
  - [The HEAD](#the-head)
  - [Tags](#tags)
  - [Remotes](#remotes)
- [Packfile](#packfile)
- [The Refspec](#the-refspec)
  - [The Refspec Fetch](#the-refspec-fetch)
  - [Refspec Push](#refspec-push)
  - [Deleting References](#deleting-references)
- [Maintenance and Data Recovery](#maintenance-and-data-recovery)
  - [Maintenance](#maintenance)
  - [Data Recovery](#data-recovery)
  - [Removing Objects](#removing-objects)
- [Environment Variables](#environment-variables)
  - [Global Behavior](#global-behavior)
  - [Repository Locations](#repository-locations)
  - [Pathspecs](#pathspecs)
  - [Committing](#committing)
  - [Networking](#networking)
  - [Diffing and Merging](#diffing-and-merging)
  - [Debugging](#debugging)
  - [Miscellaneous](#miscellaneous)

-----

# Abstract

This explains Git under the hood.

# Materials

* [How not to be afraid of GIT anymore](https://www.freecodecamp.org/news/how-not-to-be-afraid-of-git-anymore-fe1da7415286/)
  * [Now that you’re not afraid of GIT anymore, here’s how to leverage what you know](https://www.freecodecamp.org/news/now-that-youre-not-afraid-of-git-anymore-here-s-how-to-leverage-what-you-know-11e710c7f37b/)
* [10.2 Git의 내부 - Git 개체 @ progit](https://git-scm.com/book/ko/v2/Git%EC%9D%98-%EB%82%B4%EB%B6%80-Git-%EA%B0%9C%EC%B2%B4)
* [Gerrit을 이용한 코드 리뷰 시스템 - Gerrit과 Git @ naverD2](https://d2.naver.com/helloworld/1859580)

# Directories

This is a directory structure of `.git`.

```
.
├── clone
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

There are 3 kinds of Git objects, commit, tree, blob object.

```bash
$ cd clone/david/HelloWorld

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

# write test.txt to blob object in objects directory.
$ git hash-object -w test.txt
83baae61804e65cc73a7201a7252750c76066a30

# modify the test.txt file.
$ echo 'version 2' > test.txt

# write test.txt to blob object in ojects directory again.
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

## Tree objects

```bash
# master^{tree} means the tree object of the commit object pointed by master
$ git cat-file -p master^{tree}
100644 blob a906cb2a4a904a152e80877d4088654daad0c859      README
100644 blob 8f94139338f9404f26296befa88755fc2598c289      Rakefile
040000 tree 99f1a6d12cb4b6f19c8655fca46c3ecf317074e0      lib

$ git cat-file -p 99f1a6d12cb4b6f19c8655fca46c3ecf317074e0
100644 blob 47c6340d6459e05787f644c2447d2595f5d3a54b      simplegit.rb

# let's make a tree object. git make a tree object from index
# update the index with adding test.txt.
# update-index: make a index
#  --add: add the test.txt to the index
#  --cacheinfo: test.txt exists in database not in working directory
#  100644: regular file, (100755: executable file, 120000: symbloic link)
$ git update-index --add --cacheinfo 100644 \
  83baae61804e65cc73a7201a7252750c76066a30 test.txt

# write the index to the tree object.
$ git write-tree
d8329fc1cc938780ffdd9f94e0d364e0ea74f579

# show the content of the tree object
$ git cat-file -p d8329fc1cc938780ffdd9f94e0d364e0ea74f579
100644 blob 83baae61804e65cc73a7201a7252750c76066a30      test.txt

# show the type of the tree object
$ git cat-file -t d8329fc1cc938780ffdd9f94e0d364e0ea74f579
tree

# make a new file.
$ echo 'new file' > new.txt

# update the index adding test.txt
$ git update-index --add --cacheinfo 100644 \
  1f7a7a472abf3dd9643fd615f6da379c4acb3e3a test.txt

# update the index adding new.txt
$ git update-index --add new.txt

# save the index to the tree object.
$ git write-tree
0155eb4229851634a0f03eb265b69f5a2d56f341

# show the content of the tree object.
$ git cat-file -p 0155eb4229851634a0f03eb265b69f5a2d56f341
100644 blob fa49b077972391ad58037050f2a75f74e3671e92      new.txt
100644 blob 1f7a7a472abf3dd9643fd615f6da379c4acb3e3a      test.txt

# update index with adding the tree object.
$ git read-tree --prefix=bak d8329fc1cc938780ffdd9f94e0d364e0ea74f579

# save the index to the tree object.
$ git write-tree
3c4e9cd789d88d8d89c1073707c3585e41b0e614

# show the content of the tree object
$ git cat-file -p 3c4e9cd789d88d8d89c1073707c3585e41b0e614
040000 tree d8329fc1cc938780ffdd9f94e0d364e0ea74f579      bak
100644 blob fa49b077972391ad58037050f2a75f74e3671e92      new.txt
100644 blob 1f7a7a472abf3dd9643fd615f6da379c4acb3e3a      test.txt
```

## Commit Objects

```bash
# make the commit object with hash of the tree object.
$ echo 'first commit' | git commit-tree d8329f
fdf4fc3344e67ab068f836878b6c4951e3b15f3d

# show the conent of the commit object.
$ git cat-file -p fdf4fc3
tree d8329fc1cc938780ffdd9f94e0d364e0ea74f579
author Scott Chacon <schacon@gmail.com> 1243040974 -0700
committer Scott Chacon <schacon@gmail.com> 1243040974 -0700

first commit

$ echo 'second commit' | git commit-tree 0155eb -p fdf4fc3
cac0cab538b970a37ea1e769cbbde608743bc96d

$ echo 'third commit'  | git commit-tree 3c4e9c -p cac0cab
1a410efbd13591db07496601ebc7a059dd55cfe9

# show logs
$ git log --stat 1a410e
commit 1a410efbd13591db07496601ebc7a059dd55cfe9
Author: Scott Chacon <schacon@gmail.com>
Date:   Fri May 22 18:15:24 2009 -0700

    third commit

 bak/test.txt | 1 +
 1 file changed, 1 insertion(+)

commit cac0cab538b970a37ea1e769cbbde608743bc96d
Author: Scott Chacon <schacon@gmail.com>
Date:   Fri May 22 18:14:29 2009 -0700

    second commit

 new.txt  | 1 +
 test.txt | 2 +-
 2 files changed, 2 insertions(+), 1 deletion(-)

commit fdf4fc3344e67ab068f836878b6c4951e3b15f3d
Author: Scott Chacon <schacon@gmail.com>
Date:   Fri May 22 18:09:34 2009 -0700

    first commit

 test.txt | 1 +
 1 file changed, 1 insertion(+)

# show past objects we made
$ find .git/objects -type f
.git/objects/01/55eb4229851634a0f03eb265b69f5a2d56f341 # tree 2
.git/objects/1a/410efbd13591db07496601ebc7a059dd55cfe9 # commit 3
.git/objects/1f/7a7a472abf3dd9643fd615f6da379c4acb3e3a # test.txt v2
.git/objects/3c/4e9cd789d88d8d89c1073707c3585e41b0e614 # tree 3
.git/objects/83/baae61804e65cc73a7201a7252750c76066a30 # test.txt v1
.git/objects/ca/c0cab538b970a37ea1e769cbbde608743bc96d # commit 2
.git/objects/d6/70460b4b4aece5915caf5c68d12f560a9fe3e4 # 'test content'
.git/objects/d8/329fc1cc938780ffdd9f94e0d364e0ea74f579 # tree 1
.git/objects/fa/49b077972391ad58037050f2a75f74e3671e92 # new.txt
.git/objects/fd/f4fc3344e67ab068f836878b6c4951e3b15f3d # commit 1
```

# Git References

Git saves them to `.git/refs/*`. If we use references it is very easy point the commit object.

```bash
$ find .git/refs
.git/refs
.git/refs/heads
.git/refs/tags

$ find .git/refs -type f

# make a reference with using echo command.
$ echo 1a410efbd13591db07496601ebc7a059dd55cfe9 > .git/refs/heads/master

$ git log --pretty=oneline master
1a410efbd13591db07496601ebc7a059dd55cfe9 third commit
cac0cab538b970a37ea1e769cbbde608743bc96d second commit
fdf4fc3344e67ab068f836878b6c4951e3b15f3d first commit

# make a ref with using plumbing command.
$ git update-ref refs/heads/master 1a410efbd13591db07496601ebc7a059dd55cfe9

# make a ref test.
$ git update-ref refs/heads/test cac0ca
```

## The HEAD

HEAD is a symbolic reference to the branch you're currently on.

```bash
$ cat .git/HEAD
ref: refs/heads/master

$ cat .git/HEAD
ref: refs/heads/test

$ git symbolic-ref HEAD
refs/heads/master

# set the value of HEAD.
$ git symbolic-ref HEAD refs/heads/test
$ cat .git/HEAD
ref: refs/heads/test

# we have to keep the refs rules
$ git symbolic-ref HEAD test
fatal: Refusing to point HEAD outside of refs/
```

## Tags

```bash
# set lightweight tag
$ git update-ref refs/tags/v1.0 cac0cab538b970a37ea1e769cbbde608743bc96d

# set annotated tag
$ git tag -a v1.1 1a410efbd13591db07496601ebc7a059dd55cfe9 -m 'test tag'

$ cat .git/refs/tags/v1.1
9585191f37f7b0fb9444f35a9bf50de191beadc2

$ git cat-file -p 9585191f37f7b0fb9444f35a9bf50de191beadc2
object 1a410efbd13591db07496601ebc7a059dd55cfe9
type commit
tag v1.1
tagger Scott Chacon <schacon@gmail.com> Sat May 23 16:48:58 2009 -0700

test tag

# You can view the public key by running this in a clone of the Git repository
$ git cat-file blob junio-gpg-pub

```

## Remotes

remote refs is readonly and a kind of bookmarks.

```bash
$ git remote add origin git@github.com:schacon/simplegit-progit.git
$ git push origin master
Counting objects: 11, done.
Compressing objects: 100% (5/5), done.
Writing objects: 100% (7/7), 716 bytes, done.
Total 7 (delta 2), reused 4 (delta 1)
To git@github.com:schacon/simplegit-progit.git
  a11bef0..ca82a6d  master -> master

$ cat .git/refs/remotes/origin/master
ca82a6dff817ec66f44342007202690a93763949
```

# Packfile

```bash
# These are what we made.
$ find .git/objects -type f
.git/objects/01/55eb4229851634a0f03eb265b69f5a2d56f341 # tree 2
.git/objects/1a/410efbd13591db07496601ebc7a059dd55cfe9 # commit 3
.git/objects/1f/7a7a472abf3dd9643fd615f6da379c4acb3e3a # test.txt v2
.git/objects/3c/4e9cd789d88d8d89c1073707c3585e41b0e614 # tree 3
.git/objects/83/baae61804e65cc73a7201a7252750c76066a30 # test.txt v1
.git/objects/95/85191f37f7b0fb9444f35a9bf50de191beadc2 # tag
.git/objects/ca/c0cab538b970a37ea1e769cbbde608743bc96d # commit 2
.git/objects/d6/70460b4b4aece5915caf5c68d12f560a9fe3e4 # 'test content'
.git/objects/d8/329fc1cc938780ffdd9f94e0d364e0ea74f579 # tree 1
.git/objects/fa/49b077972391ad58037050f2a75f74e3671e92 # new.txt
.git/objects/fd/f4fc3344e67ab068f836878b6c4951e3b15f3d # commit 1

$ curl https://raw.githubusercontent.com/mojombo/grit/master/lib/grit/repo.rb > repo.rb
$ git checkout master
$ git add repo.rb
$ git commit -m 'added repo.rb'
[master 484a592] added repo.rb
 3 files changed, 709 insertions(+), 2 deletions(-)
 delete mode 100644 bak/test.txt
 create mode 100644 repo.rb
 rewrite test.txt (100%)

$ git cat-file -p master^{tree}
100644 blob fa49b077972391ad58037050f2a75f74e3671e92      new.txt
100644 blob 033b4468fa6b2a9547a70d88d1bbe8bf3f9ed0d5      repo.rb
100644 blob e3f094f522629ae358806b17daf78246c27c007b      test.txt

# show how large that object
$ git cat-file -s 033b4468fa6b2a9547a70d88d1bbe8bf3f9ed0d5
22044

$ echo '# testing' >> repo.rb
$ git commit -am 'modified repo a bit'
[master 2431da6] modified repo.rb a bit
 1 file changed, 1 insertion(+)

$ git cat-file -p master^{tree}
100644 blob fa49b077972391ad58037050f2a75f74e3671e92      new.txt
100644 blob b042a60ef7dff760008df33cee372b945b6e884e      repo.rb
100644 blob e3f094f522629ae358806b17daf78246c27c007b      test.txt

$ git cat-file -s b042a60ef7dff760008df33cee372b945b6e884e
22054

# let's pack objects.
$ git gc
Counting objects: 18, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (14/14), done.
Writing objects: 100% (18/18), done.
Total 18 (delta 3), reused 0 (delta 0)

$ find .git/objects -type f
.git/objects/bd/9dbf5aae1a3862dd1526723246b20206e5fc37
.git/objects/d6/70460b4b4aece5915caf5c68d12f560a9fe3e4
.git/objects/info/packs
.git/objects/pack/pack-978e03944f5c581011e6998cd0e9e30000905586.idx
.git/objects/pack/pack-978e03944f5c581011e6998cd0e9e30000905586.pack

# show the history of packing.
$ git verify-pack -v .git/objects/pack/pack-978e03944f5c581011e6998cd0e9e30000905586.idx
2431da676938450a4d72e260db3bf7b0f587bbc1 commit 223 155 12
69bcdaff5328278ab1c0812ce0e07fa7d26a96d7 commit 214 152 167
80d02664cb23ed55b226516648c7ad5d0a3deb90 commit 214 145 319
43168a18b7613d1281e5560855a83eb8fde3d687 commit 213 146 464
092917823486a802e94d727c820a9024e14a1fc2 commit 214 146 610
702470739ce72005e2edff522fde85d52a65df9b commit 165 118 756
d368d0ac0678cbe6cce505be58126d3526706e54 tag    130 122 874
fe879577cb8cffcdf25441725141e310dd7d239b tree   136 136 996
d8329fc1cc938780ffdd9f94e0d364e0ea74f579 tree   36 46 1132
deef2e1b793907545e50a2ea2ddb5ba6c58c4506 tree   136 136 1178
d982c7cb2c2a972ee391a85da481fc1f9127a01d tree   6 17 1314 1 \
  deef2e1b793907545e50a2ea2ddb5ba6c58c4506
3c4e9cd789d88d8d89c1073707c3585e41b0e614 tree   8 19 1331 1 \
  deef2e1b793907545e50a2ea2ddb5ba6c58c4506
0155eb4229851634a0f03eb265b69f5a2d56f341 tree   71 76 1350
83baae61804e65cc73a7201a7252750c76066a30 blob   10 19 1426
fa49b077972391ad58037050f2a75f74e3671e92 blob   9 18 1445
b042a60ef7dff760008df33cee372b945b6e884e blob   22054 5799 1463
033b4468fa6b2a9547a70d88d1bbe8bf3f9ed0d5 blob   9 20 7262 1 \
  b042a60ef7dff760008df33cee372b945b6e884e
1f7a7a472abf3dd9643fd615f6da379c4acb3e3a blob   10 19 7282
non delta: 15 objects
chain length = 1: 3 objects
.git/objects/pack/pack-978e03944f5c581011e6998cd0e9e30000905586.pack: ok
```

# The Refspec

## The Refspec Fetch

```bash
$ git remote add origin https://github.com/schacon/simplegit-progit
```

* `.git/config`

```
[remote "origin"]
    url = https://github.com/schacon/simplegit-progit
    fetch = +refs/heads/*:refs/remotes/origin/*
```

The format of Refspec in fetch is `+<remote refspec pattern>:<local refspec pattern>`. `+` means to allow not even fast-forward updates. For example, `+refs/heads/*:refs/remotes/origin/*` tells that fetch refs in remote `refs/heads` to refs in local `refs/remotes/origin`.

If you want to access master branches use these commands. They are all same.

```bash
$ git log origin/master
$ git log remotes/origin/master
$ git log refs/remotes/origin/master
```

If you want to fetch just master branch use this.

```
fetch = +refs/heads/master:refs/remotes/origin/master
```

You can fetch specific branch using Refspec like this.

```bash
$ git fetch origin master:refs/remotes/origin/mymaster

# you can use 2 refspec at the sametime.
$ git fetch origin master:refs/remotes/origin/mymaster \
     topic:refs/remotes/origin/topic
# use '+' sign to fetch not even in fast-forward.
$ git fetch origin master:refs/remotes/origin/mymaster \
     +topic:refs/remotes/origin/topic
```

But cannot use Glob pattern.

```
fetch = +refs/heads/qa*:refs/remotes/origin/qa*
```

But can use Glob pattern for namespace

```
[remote "origin"]
    url = https://github.com/schacon/simplegit-progit
    fetch = +refs/heads/master:refs/remotes/origin/master
    fetch = +refs/heads/qa/*:refs/remotes/origin/qa/*
```

## Refspec Push

The format of Refspec in push is `<local refspec pattern>:<remote refspec pattern>`.

If you want to push local `master` branch to remote `qa/master` branch use this.

```bash
$ git push origin master:refs/heads/qa/master
```

If you want to push automatically like that save this to `.git/config`.

```
[remote "origin"]
    url = https://github.com/schacon/simplegit-progit
    fetch = +refs/heads/*:refs/remotes/origin/*
    push = refs/heads/master:refs/heads/qa/master
```

## Deleting References

You can delete references using Refspec.

```bash
# There is no local Refspec.
$ git push origin :topic

# This is same with before.
$ git push origin --delete topic
```

# Maintenance and Data Recovery

## Maintenance

```bash
# You can run auto gc manually as follows:
$ git gc --auto

$ find .git/refs -type f
.git/refs/heads/experiment
.git/refs/heads/master
.git/refs/tags/v1.0
.git/refs/tags/v1.1

$ git gc

# made a packed-refs file.
$ cat .git/packed-refs
# pack-refs with: peeled fully-peeled
cac0cab538b970a37ea1e769cbbde608743bc96d refs/heads/experiment
ab1afef80fac8e34258ff41fc1b867c702daa24b refs/heads/master
cac0cab538b970a37ea1e769cbbde608743bc96d refs/tags/v1.0
9585191f37f7b0fb9444f35a9bf50de191beadc2 refs/tags/v1.1
^1a410efbd13591db07496601ebc7a059dd55cfe9
```

## Data Recovery

You can recover a branch using reflog.

```bash
$ git log --pretty=oneline
ab1afef80fac8e34258ff41fc1b867c702daa24b modified repo a bit
484a59275031909e19aadb7c92262719cfcdf19a added repo.rb
1a410efbd13591db07496601ebc7a059dd55cfe9 third commit
cac0cab538b970a37ea1e769cbbde608743bc96d second commit
fdf4fc3344e67ab068f836878b6c4951e3b15f3d first commit

$ git reset --hard 1a410efbd13591db07496601ebc7a059dd55cfe9
HEAD is now at 1a410ef third commit

$ git log --pretty=oneline
1a410efbd13591db07496601ebc7a059dd55cfe9 third commit
cac0cab538b970a37ea1e769cbbde608743bc96d second commit
fdf4fc3344e67ab068f836878b6c4951e3b15f3d first commit

$ git reflog
1a410ef HEAD@{0}: reset: moving to 1a410ef
ab1afef HEAD@{1}: commit: modified repo.rb a bit
484a592 HEAD@{2}: commit: added repo.rb

$ git log -g
commit 1a410efbd13591db07496601ebc7a059dd55cfe9
Reflog: HEAD@{0} (Scott Chacon <schacon@gmail.com>)
Reflog message: updating HEAD
Author: Scott Chacon <schacon@gmail.com>
Date:   Fri May 22 18:22:37 2009 -0700

		third commit

commit ab1afef80fac8e34258ff41fc1b867c702daa24b
Reflog: HEAD@{1} (Scott Chacon <schacon@gmail.com>)
Reflog message: updating HEAD
Author: Scott Chacon <schacon@gmail.com>
Date:   Fri May 22 18:15:24 2009 -0700

       modified repo.rb a bit

$ git branch recover-branch ab1afef
$ git log --pretty=oneline recover-branch
ab1afef80fac8e34258ff41fc1b867c702daa24b modified repo a bit
484a59275031909e19aadb7c92262719cfcdf19a added repo.rb
1a410efbd13591db07496601ebc7a059dd55cfe9 third commit
cac0cab538b970a37ea1e769cbbde608743bc96d second commit
fdf4fc3344e67ab068f836878b6c4951e3b15f3d first commit
```

This is more a desperative situation.

```bash

$ git branch -D recover-branch
$ rm -Rf .git/logs/

# If you run it with the --full option, it shows you all objects that aren’t pointed to by another object. 
# In this case, you can see your missing commit after the string “dangling commit”. You can recover it the same way, by adding a branch that points to that SHA-1.
$ git fsck --full
Checking object directories: 100% (256/256), done.
Checking objects: 100% (18/18), done.
dangling blob d670460b4b4aece5915caf5c68d12f560a9fe3e4
dangling commit ab1afef80fac8e34258ff41fc1b867c702daa24b
dangling tree aea790b9a58f6cf6f2804eeac9f0abbe9631e4c9
dangling blob 7108f7ecb345ee9d0084193f147cdad4d2998293
```

## Removing Objects

```bash
$ curl https://www.kernel.org/pub/software/scm/git/git-2.1.0.tar.gz > git.tgz

$ git add git.tgz

$ git commit -m 'add git tarball'
[master 7b30847] add git tarball
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 git.tgz

$ git rm git.tgz
rm 'git.tgz'

$ git commit -m 'oops - removed large tarball'
[master dadf725] oops - removed large tarball
 1 file changed, 0 insertions(+), 0 deletions(-)
 delete mode 100644 git.tgz

$ git gc
Counting objects: 17, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (13/13), done.
Writing objects: 100% (17/17), done.
Total 17 (delta 1), reused 10 (delta 0)

$ git count-objects -v
count: 7
size: 32
in-pack: 17
packs: 1
size-pack: 4868
prune-packable: 0
garbage: 0
size-garbage: 0

$ git verify-pack -v .git/objects/pack/pack-29…69.idx \
  | sort -k 3 -n \
  | tail -3
dadf7258d699da2c8d89b09ef6670edb7d5f91b4 commit 229 159 12
033b4468fa6b2a9547a70d88d1bbe8bf3f9ed0d5 blob   22044 5792 4977696
82c99a3e86bb1267b236a4b6eff7868d97489af1 blob   4975916 4976258 1438

# If you pass --objects to rev-list, it lists all the commit SHA-1s and also the blob SHA-1s with the file paths associated with them. You can use this to find your blob’s name:
$ git rev-list --objects --all | grep 82c99a3
82c99a3e86bb1267b236a4b6eff7868d97489af1 git.tgz

$ git log --oneline --branches -- git.tgz
dadf725 oops - removed large tarball
7b30847 add git tarball

# --index-filter: you’re modifying your staging area or index each time. 
$ git filter-branch --index-filter \
  'git rm --ignore-unmatch --cached git.tgz' -- 7b30847^..
Rewrite 7b30847d080183a1ab7d18fb202473b3096e9f34 (1/2)rm 'git.tgz'
Rewrite dadf7258d699da2c8d89b09ef6670edb7d5f91b4 (2/2)
Ref 'refs/heads/master' was rewritten

$ rm -Rf .git/refs/original
$ rm -Rf .git/logs/
$ git gc
Counting objects: 15, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (11/11), done.
Writing objects: 100% (15/15), done.
Total 15 (delta 1), reused 12 (delta 0)

$ git count-objects -v
count: 11
size: 4904
in-pack: 15
packs: 1
size-pack: 8
prune-packable: 0
garbage: 0
size-garbage: 0

# you could remove the object completely by running git prune with the --expire option:
$ git prune --expire now
$ git count-objects -v
count: 0
size: 0
in-pack: 15
packs: 1
size-pack: 8
prune-packable: 0
garbage: 0
size-garbage: 0
```

# Environment Variables

## Global Behavior

* `GIT_EXEC_PATH`
  * path for subprograms like git-commit, git-diff.
* `HOME`
  * Git looks for user configuration at `$HOME/.gitconfig`   
* `PREFIX`
  * Git looks for global configuration at `$PREFIX/etc/gitconfig`.
* `GIT_CONFIG_NOSYSTEM`
  * if set, disable global configuration.
* `GIT_PAGER`
  * If this is unset, no paing.
* `GIT_EDITOR`
  * Git will launch when the user needs to edit some text. if unset, `$EDITOR` will be used.

## Repository Locations

* `GIT_DIR`
  * the location of the `.git` folder
* `GIT_CEILING_DIRECTORIES`
  * controls the behavior of searching for a .git directory
* `GIT_WORK_TREE`
  * the location of the root of the working directory for a non-bare repository.
* `GIT_INDEX_FILE`
  * is the path to the index file for non-bare repository.
* `GIT_OBJECT_DIRECTORY`
  * `.git/objects`
* `GIT_ALTERNATE_OBJECT_DIRECTORIES`
  * a colon-separated list (formatted like /dir/one:/dir/two:…) which tells Git where to check for objects if they aren’t in `GIT_OBJECT_DIRECTORY`.

## Pathspecs

* `GIT_GLOB_PATHSPECS`
  * If `GIT_GLOB_PATHSPECS` is set to 1, wildcard characters act as wildcards (which is the default)
* `GIT_NOGLOB_PATHSPECS`
  * if `GIT_NOGLOB_PATHSPECS` is set to 1, wildcard characters only match themselves, meaning something like *.c would only match a file named “*.c”, rather than any file whose name ends with .c.
* `GIT_LITERAL_PATHSPECS`
  * disables both of the above behaviors.
* `GIT_ICASE_PATHSPECS`
*  sets all pathspecs to work in a case-insensitive manner

## Committing

* `GIT_AUTHOR_NAME`
  * name in the "author" field
* `GIT_AUTHOR_EMAIL`
  * email for the "author" field
* `GIT_AUTHOR_DATE`
  * the timestamp used for the "author" field
* `GIT_COMMITTER_NAME`
  * name for the "committer" field
* `GIT_COMMITTER_EMAIL`
  * email for the "committer" field
* `GIT_COMMITTER_DATE`
  * the timestamp in the "committer" field

## Networking

* `GIT_CURL_VERBOSE`
  * `curl -v`
* `GIT_SSL_NO_VERIFY`
  * This can sometimes be necessary if you’re using a self-signed certificate to serve Git repositories over HTTPS
* `GIT_HTTP_LOW_SPEED_LIMIT`
  * Git will abort that operation if the date rate of an HTTP operations is lower than `GIT_HTTP_LOW_SPEED_LIMIT`. It overrides `http.lowSpeedLimit`.
* `GIT_HTTP_LOW_SPEED_TIME`
  * Git will abort that operation if bytes for second of an HTTP operations is longer than `GIT_HTTP_LOW_SPEED_LIMIT`. It overrides `http.lowSpeedTime`.
* `GIT_HTTP_USER_AGENT`
  * The default is a value like `git/2.0.0`.

## Diffing and Merging

* `GIT_DIFF_OPTS`
  * The only valid values are `-u<n>` or `--unified=<n>`
* `GIT_EXTERNAL_DIFF`
  * external diff tool
* `GIT_DIFF_PATH_COUNTER, GIT_DIFF_PATH_TOTAL`
  * The former represents which file in a series is being diffed (starting with 1), and the latter is the total number of files in the batch.
* `GIT_MERGE_VERBOSITY`
  * 0: outputs nothing, except possibly a single error message.
  * 1: shows only conflicts.
  * **2: also shows file changes. (default)**
  * 3: shows when files are skipped because they haven’t changed.
  * 4: shows all paths as they are processed.
  * 5: and above show detailed debugging information.

## Debugging

* `GIT_TRACE`

```bash
$ GIT_TRACE=true git lga
20:12:49.877982 git.c:554               trace: exec: 'git-lga'
20:12:49.878369 run-command.c:341       trace: run_command: 'git-lga'
20:12:49.879529 git.c:282               trace: alias expansion: lga => 'log' '--graph' '--pretty=oneline' '--abbrev-commit' '--decorate' '--all'
20:12:49.879885 git.c:349               trace: built-in: git 'log' '--graph' '--pretty=oneline' '--abbrev-commit' '--decorate' '--all'
20:12:49.899217 run-command.c:341       trace: run_command: 'less'
20:12:49.899675 run-command.c:192       trace: exec: 'less'
```
   
* `GIT_TRACE_PACK_ACCESS`

```bash
$ GIT_TRACE_PACK_ACCESS=true git status
20:10:12.081397 sha1_file.c:2088        .git/objects/pack/pack-c3fa...291e.pack 12
20:10:12.081886 sha1_file.c:2088        .git/objects/pack/pack-c3fa...291e.pack 34662
20:10:12.082115 sha1_file.c:2088        .git/objects/pack/pack-c3fa...291e.pack 35175
# […]
20:10:12.087398 sha1_file.c:2088        .git/objects/pack/pack-e80e...e3d2.pack 56914983
20:10:12.087419 sha1_file.c:2088        .git/objects/pack/pack-e80e...e3d2.pack 14303666
On branch master
Your branch is up-to-date with 'origin/master'.
nothing to commit, working directory clean
```
  
* `GIT_TRACE_PACKET`

```bash
$ GIT_TRACE_PACKET=true git ls-remote origin
20:15:14.867043 pkt-line.c:46           packet:          git< # service=git-upload-pack
20:15:14.867071 pkt-line.c:46           packet:          git< 0000
20:15:14.867079 pkt-line.c:46           packet:          git< 97b8860c071898d9e162678ea1035a8ced2f8b1f HEAD\0multi_ack thin-pack side-band side-band-64k ofs-delta shallow no-progress include-tag multi_ack_detailed no-done symref=HEAD:refs/heads/master agent=git/2.0.4
20:15:14.867088 pkt-line.c:46           packet:          git< 0f20ae29889d61f2e93ae00fd34f1cdb53285702 refs/heads/ab/add-interactive-show-diff-func-name
20:15:14.867094 pkt-line.c:46           packet:          git< 36dc827bc9d17f80ed4f326de21247a5d1341fbc refs/heads/ah/doc-gitk-config
# […]
```

* `GIT_TRACE_PERFORMANCE`

```bash
$ GIT_TRACE_PERFORMANCE=true git gc
20:18:19.499676 trace.c:414             performance: 0.374835000 s: git command: 'git' 'pack-refs' '--all' '--prune'
20:18:19.845585 trace.c:414             performance: 0.343020000 s: git command: 'git' 'reflog' 'expire' '--all'
Counting objects: 170994, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (43413/43413), done.
Writing objects: 100% (170994/170994), done.
Total 170994 (delta 126176), reused 170524 (delta 125706)
20:18:23.567927 trace.c:414             performance: 3.715349000 s: git command: 'git' 'pack-objects' '--keep-true-parents' '--honor-pack-keep' '--non-empty' '--all' '--reflog' '--unpack-unreachable=2.weeks.ago' '--local' '--delta-base-offset' '.git/objects/pack/.tmp-49190-pack'
20:18:23.584728 trace.c:414             performance: 0.000910000 s: git command: 'git' 'prune-packed'
20:18:23.605218 trace.c:414             performance: 0.017972000 s: git command: 'git' 'update-server-info'
20:18:23.606342 trace.c:414             performance: 3.756312000 s: git command: 'git' 'repack' '-d' '-l' '-A' '--unpack-unreachable=2.weeks.ago'
Checking connectivity: 170994, done.
20:18:25.225424 trace.c:414             performance: 1.616423000 s: git command: 'git' 'prune' '--expire' '2.weeks.ago'
20:18:25.232403 trace.c:414             performance: 0.001051000 s: git command: 'git' 'rerere' 'gc'
20:18:25.233159 trace.c:414             performance: 6.112217000 s: git command: 'git' 'gc'
```

* `GIT_TRACE_SETUP`

```bash
$ GIT_TRACE_SETUP=true git status
20:19:47.086765 trace.c:315             setup: git_dir: .git
20:19:47.087184 trace.c:316             setup: worktree: /Users/ben/src/git
20:19:47.087191 trace.c:317             setup: cwd: /Users/ben/src/git
20:19:47.087194 trace.c:318             setup: prefix: (null)
On branch master
Your branch is up-to-date with 'origin/master'.
nothing to commit, working directory clean
```

## Miscellaneous

* `GIT_SSH`
  * a program that is invoked instead of ssh.
* `GIT_ASKPASS`
  * It overrides `core.askpass`.
* `GIT_NAMESPACE`
  * same with `--namespace`
* `GIT_FLUSH`
  * can be used to force Git to use non-buffered I/O when writing incrementally to stdout.
* `GIT_REFLOG_ACTION`
  * lets you specify the descriptive text written to the reflog.

```bash
$ GIT_REFLOG_ACTION="my action" git commit --allow-empty -m 'my message'
[master 9e3d55a] my message
$ git reflog -1
9e3d55a HEAD@{0}: my action: my message
```
