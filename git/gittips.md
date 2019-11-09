# Materials

* [10.2 Git의 내부 - Git 개체 @ progit](https://git-scm.com/book/ko/v2/Git%EC%9D%98-%EB%82%B4%EB%B6%80-Git-%EA%B0%9C%EC%B2%B4)
* [Gerrit을 이용한 코드 리뷰 시스템 - Gerrit과 Git @ naverD2](https://d2.naver.com/helloworld/1859580)

# Deleting branches

Git is a contents addressable file system. Structures of Git objects (commit, tree, blob). The red box means a snapshot.

![](img/gitobjects.png)

The blob object saves the brand new version of a file not difference. So several blob objects of same file can be huge size. But Git provide packing to pack old objects and save just differences of blob bojects. After `$ git gc` Git will make files `./git/info/refs, ./git/objects/pack/*`

Recent commit objects can't be packed. So snapshos become huge even if they are useless. Finally, It's very important to delete old branches.

This is a command line which deletes all branches except master.

```bash
$ git branch | grep -v "master" | xargs git branch -D 
```

Even though you deleted branches you can restore using `$ git reflog`

```bash
$ git reflog
$ git branch feature/oldbranch xxxxxx
```

# Git LFS

[Git LFS @ TIL](/git#git-lfs)
