# Install

## Install gitlab-ce 12.3.5-ce.0 with docker on Windows10

```bash
> docker pull gitlab/gitlab-ce
> docker run -d --hostname gitlab.example.com -p 443:443 -p 80:80 -p 22:22 --name gitlab --volume d:\my\dockervolume\gitlab\config:/etc/gitlab --volume d:\my\dockervolume\gitlab\logs:/var/log/gitlab --volume d:\my\dockervolume\gitlab\data:/var/opt/gitlab gitlab/gitlab-ce:latest
> docker logs gitlab -f
```

failed on windows because of chef error. This is a error log.

```
Thank you for using GitLab Docker Image!
Current version: gitlab-ce=12.3.5-ce.0

Configure GitLab for your system by editing /etc/gitlab/gitlab.rb file
And restart this container to reload settings.
To do it use docker exec:

  docker exec -it gitlab vim /etc/gitlab/gitlab.rb
  docker restart gitlab

For a comprehensive list of configuration options please see the Omnibus GitLab readme
https://gitlab.com/gitlab-org/omnibus-gitlab/blob/master/README.md

If this container fails to start due to permission problems try to fix it by executing:

  docker exec -it gitlab update-permissions
  docker restart gitlab

Cleaning stale PIDs & sockets
Preparing services...
Starting services...
Configuring GitLab...
/opt/gitlab/embedded/bin/runsvdir-start: line 24: ulimit: pending signals: cannot modify limit: Operation not permitted
/opt/gitlab/embedded/bin/runsvdir-start: line 37: /proc/sys/fs/file-max: Read-only file system
Starting Chef Client, version 14.13.11
resolving cookbooks for run list: ["gitlab"]
Synchronizing Cookbooks:
  - gitlab (0.0.1)
  - package (0.1.0)
  - postgresql (0.1.0)
  - redis (0.1.0)
  - mattermost (0.1.0)
  - registry (0.1.0)
  - monitoring (0.1.0)
  - consul (0.1.0)
  - gitaly (0.1.0)
  - letsencrypt (0.1.0)
  - runit (4.3.0)
  - nginx (0.1.0)
  - acme (4.0.0)
  - crond (0.1.0)
Installing Cookbook Gems:
Compiling Cookbooks...
Recipe: gitlab::default
  * directory[/etc/gitlab] action create
    - change mode from '0777' to '0775'
  Converging 269 resources
  * directory[/etc/gitlab] action create (up to date)
  * directory[Create /var/opt/gitlab] action create
    - change mode from '0777' to '0755'
  * directory[Create /var/log/gitlab] action create
    - change mode from '0777' to '0755'
  * directory[/opt/gitlab/embedded/etc] action create
    - create new directory /opt/gitlab/embedded/etc
    - change mode from '' to '0755'
    - change owner from '' to 'root'
    - change group from '' to 'root'
  * template[/opt/gitlab/embedded/etc/gitconfig] action create
    - create new file /opt/gitlab/embedded/etc/gitconfig
    - update content in file /opt/gitlab/embedded/etc/gitconfig from none to fb60c1
    --- /opt/gitlab/embedded/etc/gitconfig      2019-10-17 19:38:09.936624800 +0000
    +++ /opt/gitlab/embedded/etc/.chef-gitconfig20191017-22-hvbrsa      2019-10-17 19:38:09.936624800 +0000
    @@ -1 +1,14 @@
    +[pack]
    +  threads = 1
    +[receive]
    +  fsckObjects = true
    +advertisePushOptions = true
    +[repack]
    +  writeBitmaps = true
    +[transfer]
    +  hideRefs=^refs/tmp/
    +hideRefs=^refs/keep-around/
    +hideRefs=^refs/remotes/
    +[core]
    +  alternateRefsCommand="exit 0 #"
    - change mode from '' to '0755'
Recipe: gitlab::web-server
  * account[Webserver user and group] action create (up to date)
Recipe: gitlab::users
  * directory[/var/opt/gitlab] action create (up to date)
  * account[GitLab user and group] action create (up to date)
  * template[/var/opt/gitlab/.gitconfig] action create
    - change mode from '0755' to '0644'
    - change owner from 'root' to 'git'
    - change group from 'root' to 'git'
  * directory[/var/opt/gitlab/.bundle] action create
    - change owner from 'root' to 'git'
    - change group from 'root' to 'git'
Recipe: gitlab::gitlab-shell
  * storage_directory[/var/opt/gitlab/.ssh] action create
    * ruby_block[directory resource: /var/opt/gitlab/.ssh] action run

      ================================================================================
      Error executing action `run` on resource 'ruby_block[directory resource: /var/opt/gitlab/.ssh]'
      ================================================================================

      Mixlib::ShellOut::ShellCommandFailed
      ------------------------------------
      Failed asserting that ownership of "/var/opt/gitlab/.ssh" was git:git
      ---- Begin output of set -x && [ "$(stat --printf='%U:%G' $(readlink -f /var/opt/gitlab/.ssh))" = 'git:git' ] ----
      STDOUT:
      STDERR: + readlink -f /var/opt/gitlab/.ssh
      + stat --printf=%U:%G /var/opt/gitlab/.ssh
      + [ root:root = git:git ]
      ---- End output of set -x && [ "$(stat --printf='%U:%G' $(readlink -f /var/opt/gitlab/.ssh))" = 'git:git' ] ----
      Ran set -x && [ "$(stat --printf='%U:%G' $(readlink -f /var/opt/gitlab/.ssh))" = 'git:git' ] returned 1

      Cookbook Trace:
      ---------------
      /opt/gitlab/embedded/cookbooks/cache/cookbooks/package/libraries/storage_directory_helper.rb:125:in `validate_command'
      /opt/gitlab/embedded/cookbooks/cache/cookbooks/package/libraries/storage_directory_helper.rb:113:in `block in validate'
      /opt/gitlab/embedded/cookbooks/cache/cookbooks/package/libraries/storage_directory_helper.rb:112:in `each_index'
      /opt/gitlab/embedded/cookbooks/cache/cookbooks/package/libraries/storage_directory_helper.rb:112:in `validate'
      /opt/gitlab/embedded/cookbooks/cache/cookbooks/package/libraries/storage_directory_helper.rb:87:in `validate!'
      /opt/gitlab/embedded/cookbooks/cache/cookbooks/package/resources/storage_directory.rb:43:in `block (3 levels) in class_from_file'

      Resource Declaration:
      ---------------------
      # In /opt/gitlab/embedded/cookbooks/cache/cookbooks/package/resources/storage_directory.rb

       34:   ruby_block "directory resource: #{new_resource.path}" do
       35:     block do
       36:       # Ensure the directory exists
       37:       storage_helper.ensure_directory_exists(new_resource.path)
       38:
       39:       # Ensure the permissions are set
       40:       storage_helper.ensure_permissions_set(new_resource.path)
       41:
       42:       # Error out if we have not achieved the target permissions
       43:       storage_helper.validate!(new_resource.path)
       44:     end
       45:     not_if { storage_helper.validate(new_resource.path) }
       46:   end
       47: end

      Compiled Resource:
      ------------------
      # Declared in /opt/gitlab/embedded/cookbooks/cache/cookbooks/package/resources/storage_directory.rb:34:in `block in class_from_file'

      ruby_block("directory resource: /var/opt/gitlab/.ssh") do
        action [:run]
        default_guard_interpreter :default
        declared_type :ruby_block
        cookbook_name "gitlab"
        block #<Proc:0x0000000005b7a4a8@/opt/gitlab/embedded/cookbooks/cache/cookbooks/package/resources/storage_directory.rb:35>
        block_name "directory resource: /var/opt/gitlab/.ssh"
        not_if { #code block }
      end

      System Info:
      ------------
      chef_version=14.13.11
      platform=ubuntu
      platform_version=16.04
      ruby=ruby 2.6.3p62 (2019-04-16 revision 67580) [x86_64-linux]
      program_name=/opt/gitlab/embedded/bin/chef-client
      executable=/opt/gitlab/embedded/bin/chef-client
```

I tried again.

```bash
> docker run -d -p 443:443 -p 80:80 -p 22:22 --name gitlab gitlab/gitlab-ce:latest
```

succeeded with volumes. Then open the browser with url `localhost`, change password, login with `root`. It's Done.

## Install gitlab-ce 12.3.5-ce.0 with docker on macOS

```bash
> docker pull gitlab/gitlab-ce
> sudo docker run -d --hostname gitlab.example.com -p 443:443 -p 80:80 -p 22:22 --name gitlab --volume /Users/iamslash/my/dockervolume/gitlab/config:/etc/gitlab --volume /Users/iamslash/my/dockervolume/gitlab/logs:/var/log/gitlab --volume /Users/iamslash/dockervolume/gitlab/data:/var/opt/gitlab gitlab/gitlab-ce:latest
> docker logs my-gitlab -f
```

# System Architecture

* [GitLab Architecture Overview](https://docs.gitlab.com/ee/development/architecture.html)

-----

![](https://docs.gitlab.com/ee/development/img/architecture_simplified.png)


# Brainstorming

* group 은 github 의 team 과 비슷하다.
* 