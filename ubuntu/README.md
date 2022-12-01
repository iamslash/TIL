# Ubuntu Dev Env

* [1분 만에 우분투 개발환경 만들기](https://bitlog.tistory.com/47)
  * [dev-ubuntu-docker](https://github.com/ikaruce/dev-ubuntu-docker)

----

```bash
$ docker run -d --name my-ubuntu --volume /Users/david.s/my:/root/my ikaruce/ubuntu:20.04
$ docker exec -it my-ubuntu bash
```
