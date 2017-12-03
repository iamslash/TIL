# Abstract

유용한 프로그램들들 정리한다.

# tmux

terminal multiplexer

```bash
tmux ls         : list session
tmux new -s <session-name> 
tmux attach -t <session-number or session-name>
```

```
CTRL + b, <key>
CTRL + b, ?     : help
CTRL + b, $     : rename session
CTRL + b, d     : detach session
CTRL + b, c     : create window
CTRL + b, ,     : rename window
CTRL + b, &     : destroy window
CTRL + b, 0-9   : move to numbered window
          n     : move to next window
          p     : move to prev window
          l     : move to last window
          w     : move to selected window
          f     : move to named window
CTRL + b, %     : split horizontally
CTRL + b, "     : split vertically
CTRL + b, q     : move to numbered pane
CTRL + b, o     : move to next pane
CTRL + b, <arrow-key> : move to arrowed pane
CTRL + b, x     : destroy pane
CTRL + b, :     : resize pane
             -L : 10
             -R : 10
             -D : 10
             -U : 10
CTRL + b, spacebar :
```
