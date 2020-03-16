# Materials

* [17 useful rsync (remote sync) Command Examples in Linux](https://www.linuxtechi.com/rsync-command-examples-linux/)

# Basics

## Options

Some of the commonly used options in rsync command are listed below:

```
-v, –verbose                             Verbose output
-q, –quiet                                  suppress message output
-a, –archive                              archive files and directory while synchronizing ( -a equal to following options -rlptgoD)
-r, –recursive                           sync files and directories recursively
-b, –backup                              take the backup during synchronization
-u, –update                              don’t copy the files from source to destination if destination files are newer
-l, –links                                   copy symlinks as symlinks during the sync
-n, –dry-run                             perform a trial run without synchronization
-e, –rsh=COMMAND            mention the remote shell to use in rsync
-z, –compress                          compress file data during the transfer
-h, –human-readable            display the output numbers in a human-readable format
–progress                                 show the sync progress during transfer
```

## Sync locally

```bash
$ rsync -avh --proress /tmp/foo/ /tmp/bar
```
