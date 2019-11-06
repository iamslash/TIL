# Materials

* [우린 Git-flow를 사용하고 있어요](http://woowabros.github.io/experience/2017/10/30/baemin-mobile-git-branch-strategy.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/) 
* [Git flow 사용해보기](https://boxfoxs.tistory.com/347) 
* [git-flow cheatsheet](https://danielkummer.github.io/git-flow-cheatsheet/) 

# Git branching model by Vincent Drissen

![](img/git-flow_overall_graph.png)

* There are 5 branches such as master, develop, feature/*, release/*, hotfix/*.
  * master : production deployment
  * develop : most recent
  * feature : for the specific feature
  * release : stage deployment
  * hotfix : bugs of production deployment

There are 2 remote repositories, upstream and origin. the origin is the forked one from the upstream.

```bash
$ cd ~/tmp
$ mkdir local origin upstream
$ mkdir local/david local/peter
$ mkdir origin/david origin/peter
$ git init --bare upstream/HelloWorld.git
$ cd origin/david && git clone ../../upstream/HelloWorld.git  && cd ~/tmp
$ cd origin/peter && git clone ../../upstream/HelloWorld.git && cd ~/tmp
$ cd local/david && git clone ../../origin/david/HelloWorld && git remote add upstream ../../origin/david/HelloWorld && cd ~/tmp
$ cd local/peter && git clone ../../origin/peter/HelloWorld && git remote add upstream ../../origin/peter/HelloWorld && cd ~/tmp
$ cd local/david && git remote -v && cd ~/tmp
$ cd local/peter && git remote -v && cd ~/tmp
```

This is a direcgtory structure what we did.


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

And There should be version convention, and 3 system environments, DEV (development environemnt), STG (stage environment), PRD (production environment).

When you need sync the origin with the upstream you need to fetch from upstream.

* [Configuring a remote for a fork](https://help.github.com/articles/configuring-a-remote-for-a-fork/)
* [Syncing a fork](https://help.github.com/articles/syncing-a-fork/)

```bash
$ git fetch upstream
$ git checkout develop
$ git merge upstream/develop
$ git checkout master
$ git merge upstream/master
```

[git-flow](https://danielkummer.github.io/git-flow-cheatsheet/) is a good utility for Git branching models by Vicent Drissen. This is about a `git-flow init`.

```bash
$ git flow init
Which branch should be used for bringing forth production releases?
   - master
Branch name for production releases: [master]
Branch name for "next release" development: [develop]

How to name your supporting branch prefixes?
Feature branches? [feature/]
Bugfix branches? [bugfix/]
Release branches? [release/]
Hotfix branches? [hotfix/]
Support branches? [support/]
Version tag prefix? []
Hooks and filters directory? [D:/tmp/HelloWorld/.git/hooks]
```

When you start a new **feature** `Foo` you need to do this. I am going to do this on `L/david/HelloWorld`.

```bash
## initial status
$ git log -5 --decorate --graph --oneline
* 2fee5b4 (HEAD -> master, origin/master) kick off

## start feature
$ git flow feature start Foo
Switched to a new branch 'feature/Foo'

Summary of actions:
- A new branch 'feature/Foo' was created, based on 'develop'
- You are now on branch 'feature/Foo'

Now, start committing on your feature. When done, use:

     git flow feature finish Foo

$ git log -5 --decorate --graph --oneline
* 2fee5b4 (HEAD -> feature/Foo, origin/master, master, develop) kick off

# edit commit
$ touch Foo.md
$ git add .
$ git commit -am "added Foo.md"

$ git log -5 --decorate --graph --oneline
* 29fb4b0 (HEAD -> feature/Foo) added Foo.md
* 2fee5b4 (origin/master, master, develop) kick off

## publish feature when you need to push local/feature/Foo to origin/feature/Foo
$ git flow feature publish Foo
Counting objects: 2, done.
Writing objects: 100% (2/2), 231 bytes | 231.00 KiB/s, done.
Total 2 (delta 0), reused 0 (delta 0)
To D:/tmp/bare\HelloWorld
 * [new branch]      feature/Foo -> feature/Foo
Branch 'feature/Foo' set up to track remote branch 'feature/Foo' from 'origin'.
Already on 'feature/Foo'
Your branch is up to date with 'origin/feature/Foo'.

Summary of actions:
- The remote branch 'feature/Foo' was created or updated
- The local branch 'feature/Foo' was configured to track the remote branch
- You are now on branch 'feature/Foo'

$ git log -5 --decorate --graph --oneline
* 29fb4b0 (HEAD -> feature/Foo, origin/feature/Foo) added Foo.md
* 2fee5b4 (origin/master, master, develop) kick off

## finish feature
$ git flow feature finish Foo
Switched to branch 'develop'
Updating 2fee5b4..29fb4b0
Fast-forward
 Foo.md | 0
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 Foo.md
To D:/tmp/bare\HelloWorld
 - [deleted]         feature/Foo
Deleted branch feature/Foo (was 29fb4b0).

Summary of actions:
- The feature branch 'feature/Foo' was merged into 'develop'
- Feature branch 'feature/Foo' has been locally deleted; it has been remotely deleted from 'origin'
- You are now on branch 'develop'

$ git log -5 --decorate --graph --oneline
* 29fb4b0 (HEAD -> develop) added Foo.md
* 2fee5b4 (origin/master, master) kick off

## push local/develop to origin/develop
$ git push origin develop

$ git log -5 --decorate --graph --oneline
* 29fb4b0 (HEAD -> develop, origin/develop) added Foo.md
* 2fee5b4 (origin/master, master) kick off

## create pull request origin/develop to upstream/develop
```

But This is very important. When you merge `feature/Foo` to `develop` You have to squash them. Then You can revert it easily and make the history simple.

If you need to pull another feature you need to do this.

```bash
## pull origin/feature/Foo
$ git flow feature pull origin Foo
$ git flow feature track Foo
```

When you start make a **release** `0.1` you need to do this.  I am going to do this on `L/david/HelloWorld`

```bash
## initial status
$ git remote -v
origin  D:/tmp/R/origin\HelloWorld (fetch)
origin  D:/tmp/R/origin\HelloWorld (push)
upstream        D:/tmp/R/upstream\HelloWorld (fetch)
upstream        D:/tmp/R/upstream\HelloWorld (push)

$ git log -5 --decorate --graph --oneline
* 29fb4b0 (HEAD -> develop, origin/develop) added Foo.md
* 2fee5b4 (origin/master, master) kick off

## start a release
$ git flow release start 0.1 develop
Switched to a new branch 'release/0.1'

Summary of actions:
- A new branch 'release/0.1' was created, based on 'develop'
- You are now on branch 'release/0.1'

Follow-up actions:
- Bump the version number now!
- Start committing last-minute fixes in preparing your release
- When done, run:

     git flow release finish '0.1'

$ git log -5 --decorate --graph --oneline
* 29fb4b0 (HEAD -> release/0.1, origin/develop, develop) added Foo.md
* 2fee5b4 (origin/master, master) kick off

## publish a release when you need to push local/release/0.1 to origin/release/0.1
$ git flow release publish 0.1
Total 0 (delta 0), reused 0 (delta 0)
To D:/tmp/R/origin\HelloWorld
 * [new branch]      release/0.1 -> release/0.1
Branch 'release/0.1' set up to track remote branch 'release/0.1' from 'origin'.
Already on 'release/0.1'
Your branch is up to date with 'origin/release/0.1'.

Summary of actions:
- The remote branch 'release/0.1' was created or updated
- The local branch 'release/0.1' was configured to track the remote branch
- You are now on branch 'release/0.1'

$ git log -5 --decorate --graph --oneline
* 29fb4b0 (HEAD -> release/0.1, origin/release/0.1, origin/develop, develop) added Foo.md
* 2fee5b4 (origin/master, master) kick off

## finish a release
$ git flow release finish 0.1
Switched to branch 'master'
Your branch is up to date with 'origin/master'.
Merge made by the 'recursive' strategy.
 Foo.md | 0
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 Foo.md
Already on 'master'
Your branch is ahead of 'origin/master' by 2 commits.
  (use "git push" to publish your local commits)
Switched to branch 'develop'
Already up to date!
Merge made by the 'recursive' strategy.
To D:/tmp/R/origin\HelloWorld
 - [deleted]         release/0.1
Deleted branch release/0.1 (was 29fb4b0).

Summary of actions:
- Release branch 'release/0.1' has been merged into 'master'
- The release was tagged '0.1'
- Release tag '0.1' has been back-merged into 'develop'
- Release branch 'release/0.1' has been locally deleted; it has been remotely deleted from 'origin'
- You are now on branch 'develop'

$ git log -5 --decorate --graph --oneline
*   99f9fd9 (HEAD -> develop) Merge tag '0.1' into develop
|\
| *   721d99f (tag: 0.1, master) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 (origin/develop) added Foo.md
|/
* 2fee5b4 (origin/master) kick off

## push local/master to origin/master with tags
$ git push origin master --tags

$ git log -10 --decorate --graph --oneline
*   99f9fd9 (HEAD -> develop) Merge tag '0.1' into develop
|\
| *   721d99f (tag: 0.1, origin/master, master) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 (origin/develop) added Foo.md
|/

## push local/develop to origin/develop
$ git push origin develop --tags

$ git log -10 --decorate --graph --oneline
*   99f9fd9 (HEAD -> develop, origin/develop) Merge tag '0.1' into develop
|\
| *   721d99f (tag: 0.1, origin/master, master) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

## create pull request origin/master to upstream/master with tags
## create pull request origin/develop to upstream/develop
```

But This is also very important. When you merge `release/0.1` to `master, develop` You have to squash them. Then You can revert it easily and make the history simple.

When you start make a **hofix** `0.2` you need to do this.  I am going to do this on `L/david/HelloWorld`

```bash
$ git remote -v
origin  D:/tmp/R/origin\HelloWorld (fetch)
origin  D:/tmp/R/origin\HelloWorld (push)
upstream        D:/tmp/R/upstream\HelloWorld (fetch)
upstream        D:/tmp/R/upstream\HelloWorld (push)

$ git push upstream master --tags
Counting objects: 6, done.
Writing objects: 100% (6/6), 585 bytes | 195.00 KiB/s, done.
Total 6 (delta 0), reused 0 (delta 0)
To D:/tmp/R/upstream\HelloWorld
 * [new branch]      master -> master

$ git push upstream develop --tags
Counting objects: 1, done.
Writing objects: 100% (1/1), 225 bytes | 112.00 KiB/s, done.
Total 1 (delta 0), reused 0 (delta 0)
To D:/tmp/R/upstream\HelloWorld
 * [new branch]      develop -> develop

$ git log -10 --decorate --graph --oneline
*   99f9fd9 (HEAD -> develop, upstream/develop, origin/develop) Merge tag '0.1' into develop
|\
| *   721d99f (tag: 0.1, upstream/master, origin/master, master) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

## start a hotfix
$ git flow hotfix start 0.2 master
Switched to a new branch 'hotfix/0.2'

Summary of actions:
- A new branch 'hotfix/0.2' was created, based on 'master'
- You are now on branch 'hotfix/0.2'

Follow-up actions:
- Start committing your hot fixes
- Bump the version number now!
- When done, run:

     git flow hotfix finish '0.2'

$ vim a.md
$ git commit -am "fixed blah blah" 

$ git log -10 --decorate --graph --oneline
* 51ba540 (HEAD -> hotfix/0.2) fixed blah blah
*   721d99f (tag: 0.1, upstream/master, origin/master, master) Merge branch 'release/0.1'
|\
| * 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

## finish a hoxfix
$ git flow hotfix finish 0.2
Switched to branch 'master'
Your branch is up to date with 'origin/master'.
Merge made by the 'recursive' strategy.
 a.md | 8 ++++++++
 1 file changed, 8 insertions(+)
Switched to branch 'develop'
Merge made by the 'recursive' strategy.
 a.md | 8 ++++++++
 1 file changed, 8 insertions(+)
Deleted branch hotfix/0.2 (was 51ba540).

Summary of actions:
- Hotfix branch 'hotfix/0.2' has been merged into 'master'
- The hotfix was tagged '0.2'
- Hotfix tag '0.2' has been back-merged into 'develop'
- Hotfix branch 'hotfix/0.2' has been locally deleted
- You are now on branch 'develop'

$ git log -10 --decorate --graph --oneline
*   998d92d (HEAD -> develop) Merge tag '0.2' into develop
|\
| *   c8755e4 (tag: 0.2, master) Merge branch 'hotfix/0.2'
| |\
| | * 51ba540 fixed blah blah
| |/
* |   99f9fd9 (upstream/develop, origin/develop) Merge tag '0.1' into develop
|\ \
| |/
| *   721d99f (tag: 0.1, upstream/master, origin/master) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

## push local/develop to origin/develop
## push local/master to origin/master with tags
$ git push origin master --tags
$ git push origin develop
$ git log -10 --decorate --graph --oneline
*   998d92d (HEAD -> develop, origin/develop) Merge tag '0.2' into develop
|\
| *   c8755e4 (tag: 0.2, origin/master, master) Merge branch 'hotfix/0.2'
| |\
| | * 51ba540 fixed blah blah
| |/
* |   99f9fd9 (upstream/develop) Merge tag '0.1' into develop
|\ \
| |/
| *   721d99f (tag: 0.1, upstream/master) Merge branch 'release/0.1'
| |\
| |/
|/|
* | 29fb4b0 added Foo.md
|/
* 2fee5b4 kick off

## create a pull request origin/master to upstream/master with tags
## create a pull request origin/develop to upstream/develop

$ git log -10 --decorate --graph --oneline
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
```

But This is also very very important. When you merge `hotfix/0.1` to `master, develop` You have to squash them. Then You can revert it easily and make the history simple.
