<!-- # Materials -->

* [k9s](https://k9scli.io/)
  * [k9s | github](https://github.com/derailed/k9s)
* [k9s 설치 및 사용법](https://1minute-before6pm.tistory.com/18)

# Basic

## Basic Commands

```bash
# Show helps
k9s help
# Get info about K9s runtime (logs, configs, etc..)
k9s info
# Run K9s in a given namespace.
k9s -n myns
# Run K9s and launch in pod view via the pod command.
k9s -c pod
# Start K9s in a non default KubeConfig context
k9s --context myctx
# Start K9s in readonly mode - with all modification commands disabled
k9s --readonly
```
