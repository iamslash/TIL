# Install

## Install with docker on Windows10

```bash
> docker pull gitlab/gitlab-ce
> docker run --detach --hostname gitlab.example.com --publish 443:443 --publish 80:80 --publish 22:22 --name my-gitlab --restart always --volume d:\my\dockervolume\gitlab\config:/etc/gitlab --volume d:\my\dockervolume\gitlab\logs:/var/log/gitlab --volume d:\my\dockervolume\gitlab\data:/var/opt/gitlab gitlab/gitlab-ce:latest
> docker logs my-gitlab -f
```

**failed on windows because of chef error**

```bash
> docker pull gitlab/gitlab-ce
> docker run --detach --hostname gitlab.example.com --publish 443:443 --publish 80:80 --publish 22:22 --name my-gitlab --restart always --volume d:\my\dockervolume\gitlab\config:/etc/gitlab --volume d:\my\dockervolume\gitlab\logs:/var/log/gitlab --volume d:\my\dockervolume\gitlab\data:/var/opt/gitlab gitlab/gitlab-ce:latest
> docker logs my-gitlab -f
```

## Instal with docker on macOS

# System Architecture

