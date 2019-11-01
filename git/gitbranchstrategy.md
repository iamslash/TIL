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
$ git clone git@github.com:iamslash/TIL.git
$ git remote -v
origin	git@github.com:iamslash/TIL.git (fetch)
origin	git@github.com:iamslash/TIL.git (push)
$ git remote add upstream git@github.com:davidsun/TIL.git
$ git remote -v
origin	git@github.com:iamslash/TIL.git (fetch)
origin	git@github.com:iamslash/TIL.git (push)
upstream	git@github.com:davidsun/TIL.git (fetch)
upstream	git@github.com:davidsun/TIL.git (push)
```

When you need sync the origin with the upstream you need to fetch from upstream.

* [Configuring a remote for a fork](https://help.github.com/articles/configuring-a-remote-for-a-fork/)
* [Syncing a fork](https://help.github.com/articles/syncing-a-fork/)

```bash
$ git fetch upstream
$ git checkout master
$ git merge upstream/master
```

[git-flow](https://danielkummer.github.io/git-flow-cheatsheet/) is a good utility for Git branching models by Vicent Drissen.

When you start a new **feature** `Foo` you need to do this.

```bash

```

Make a release

Hotfixes



* 새로운 이슈 iss1 이 발급되었다.
* develop branch 로부터 feature/iss1 branch 를 만들고 기능을 구현하고 commit, push 한다. 
* develop branch 에서 feature/iss1 branch 를 merge 한다.
* 테스트를 위해 devlop branch 로부터 release/v0.0.1 branch 를 만들어 보자.
* 테스트 하면서 버그가 발생되면 release/v0.0.1 branch 에 commit, push 하고 develop branch 에 merge 한다.
* 테스트가 완료되면 release/v0.0.1 branch 를 master branch 에 merge 하고 tag v0.0.1 을 생성한다. 긜고 production zone 에 deploy 한다.
* deploy 후에 문제가 발생하면 hotfix/bug1 branch 를 만들어 commit, push 하고 develop, master branch 에 merge 한다.
* 다시 deploy 한다.

# git-flow

* 
  * git-flow 는 Vincent Driessen's branching model 을 지원하는 git extension 이다. sourcetree 역시 git-flow 를 지원한다. git-flow cheatsheet 으로 git branch strategy 의 큰 그림을 이해하는 것이 좋다.

-----

* Setup
  * `brew install git-flow-avh`
* Initialize
  * `git flow init`
* Featrues
  * Start a new feature
    * `git flow feature start MYFEATURE`
      * creates a new feature branch based on 'develop' and switches to it
  * Finish up a feature
    * `git flow feature finish MYFEATURE`
      * Merges MYFEATURE into 'develop'
      * Removes the feature branch
      * Switches back to 'develop' branch
  * Publish a feature
    * `git flow feature publish MYFEATURE`
      * Publish a feature to the remote server so it can be used by other users.
  * Getting a published feature
    * `git flow feature pull origin MYFEATURE`
      * Get a feature published by another user.
    * `git flow feature track MYFEATURE`
      * You can track a feature on origin by using
* Make a release
  * Start a release
    * `git flow release start RELEASE [BASE]`
    * `git flow release publish RELEASE`
    * `git flow release track RELEASE`
  * Finish up a release
    * `git flow release finish RELEASE`
      * Merges the release branch back into 'master'
      * Tags the release with its name
      * Back-merges the release into 'develop'
      * Removes the release branch
    * `git push origin --tags`
* Hotfixes
  * git flow hotfix start
    * `git flow hotfix start VERSION [BASENAME]`
  * Finish a hotfix
    * `git flow hotfix finish VERSION`
