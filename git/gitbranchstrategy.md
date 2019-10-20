# Materials

* [우린 Git-flow를 사용하고 있어요](http://woowabros.github.io/experience/2017/10/30/baemin-mobile-git-branch-strategy.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/) 
* [Git flow 사용해보기](https://boxfoxs.tistory.com/347) 
* [git-flow cheatsheet](https://danielkummer.github.io/git-flow-cheatsheet/) 

# Brainstorming

![](img/git-flow_overall_graph.png)


* repository 는 upstream remote repo, origin remote repo, local repo 와 같이 3 가지가 존재한다.
* branch 는 master, develop, feature-*, release-*, hotfix-* 와 같이 5 가지가 존재한다.
  * master : 출시 가능
  * develop : 다음 출시
  * feature : 기능 추가
  * release : 이번 출시
  * hotfix : 출시된 버전의 버그
* 새로운 이슈 iss1 이 발급되었다.
* 테스트를 위해 release-v.0.0.1 branch 를 만들어 보자.
* 테스트 하면서 버그가 발생되면 release-v.0.0.1 branch 에 commit 하고 develop 에 merge 한다.
* 테스트가 완료되면 release-v.0.0.1 branch 를 master branch 에 merge 하고 tag v.0.0.1 을 생성하고 production zone 에 deploy 한다.
* deploy 후에 문제가 발생하면 hotfix-bug1 branch 를 만들어 commit 하고 develop, master 에 merge 한다.
* 다시 deploy 한다.