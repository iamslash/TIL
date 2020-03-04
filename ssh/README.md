# Materials

* [ssh config 설정해서 서버마다 다른 ssh key 사용하기](https://www.lesstif.com/pages/viewpage.action?pageId=20776092)

# Usages

* `.ssh/config`

```
Host gitlab
    HostName gitlab.com
    User git
    PubkeyAcceptedKeyTypes +ssh-rsa
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/gitlab-key/id_rsa
    
Host github
    HostName github.com
    User git
    Port 22
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/github-key/id_rsa

Host my-dev
    HostName 192.168.0.2
    User iamslash
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/dev/id_rsa
    
Host *
    User iamslash
    PreferredAuthentications publickey, password
    IdentityFile ~/.ssh/id_rsa
```

* check

```bash
$ ssh -vvv -T github
```

* clone with host

```bash
$ git clone my-dev:iamslash/helloworld.git
```