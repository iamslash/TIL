- [Materials](#materials)
- [Basic](#basic)
  - [Modes](#modes)
  - [.vimrc](#vimrc)
  - [Major Commands](#major-commands)
  - [Split windows](#split-windows)
  - [Colors](#colors)

-----

# Materials

* [vim_robin @ github](https://github.com/iamslash/vim_robin)
* [밤앙개의 vim 강좌 33 - vim 플러그인 01 : vim script와 plugin이란, <Leader>란? | naver](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=nfwscho&logNo=220530572978)

# Basic

## Modes

```
Command mode: 'ESC'

Command line mode: ':'

Editor mode: 'i' 'a'

Visual mode: 'v' 'V' 'CTRL-v' `CTRL-q`
```

## .vimrc

```bash
# Echo my vimrc
:echo $MYVIMRC
/Users/david.s/.vimrc

# Source my vimrc
:source $MYVIMRC
```

## Major Commands

```bash
# Show help page
:help

# Show versions
:version

   system vimrc file: "$VIM/vimrc"
     user vimrc file: "$HOME/.vimrc"
 2nd user vimrc file: "~/.vim/vimrc"
      user exrc file: "$HOME/.exrc"
       defaults file: "$VIMRUNTIME/defaults.vim"
  fall-back for $VIM: "/usr/share/vim"
Compilation: gcc -c -I. -Iproto -DHAVE_CONFIG_H   -DMACOS_X_UNIX  -g -O2 -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=1
Linking: gcc   -L/usr/local/lib -o vim        -lm -lncurses  -liconv -framework Cocoa

# Show envs
:echo $HOME
:echo $VIM         # For .vimrc
:echo $VIMRUNTIME  # For vimrc

# Print working directory
:pwd

# Change directory
:cd $VIMRC

# Show directories
:!ls

# Help set
:help set
# Show options
:set
# Show all options
:set all

# Set noncompatible
# Activate VIM not compatible with old VI
:set noncompatible

# Set number
:set number

# Motion keys
h i j k w b

# Undo
u
# Redo
CTRL-r

# Search
/
# Next search
n
# Previous search
SHIFT-n

# Next word
w
# Previous word
b
# End of word
e
# End of line
$
# Begin of line
^

# Start of document
gg
# End of document
G 
# Page up
CTRL-u
# Page down
CTRL-d

# Delete character
x
# Delete word
dw
# Delete line
dd

# Help window
:help CTRL-w
# New window
Ctrl-wn
# Close window
Ctrl-wc
# Move window
Ctrl-w(h,j,k,l)
# Increase window one line
Ctrl-w+
# Decrease window one line
CTRL-w-
# Set windows same size
CTRL-w=

# Goto the tag on the cursor (links)
CTRL-]
# Goto the previous tag
CTRL-T
# Show tags
:tags

# Show buffers
:ls
:buffers
# Open the buffer 1
:buffer 1
# Open the next buffer
:bn
# Open the prev buffer
:bp
# Delete the buffer
:bd
# Go to the file on the cursor
:gf

# Show history
:history
:CTRL-f
q:
# Exit history
CTRL-C

# Set key map in normal mode, visual mode
:map
# Set key map in insert mode
:imap
# Set key map off
:unmap
# Set key map in insert mode off
:inoremap
# Map mode letters
# n: Normal mode
# v: Visual, select mode
# x: Visual mode only 
# s: Select mode only
# i: Insert mode 
# c: Command line mode
# l: Insert, cmd, RegEx mode
# o: Pending mode
# un: Cancel map
# re: Recursive mapping
# nore: No recursive mapping
# !: Insert and cmd-line mode
:nmap
:vmap
:xmap
:smap
:imap
:cmap
:lmap
:omap

```

## Split windows

[How To Use VIM Split Screen](https://linuxhint.com/how-to-use-vim-split-screen/)

| Short cut           | Description                   |
| ------------------- | ----------------------------- |
| `:sp <file-name>`   | split window with the file    |
| `^ws`               | split window horizontally     |
| `:open <file-name>` | open the file                 |
| `^wv`               | split window vertically       |
| `^ww`               | move window                   |
| `^wt`               | move top window               |
| `^wb`               | move bottom window            |
| `^wj`               | move left window              |
| `^wk`               | move right window             |
| `^wc`               | close window                  |
| `^w_`               | max window                    |
| `20^w_`             | set window size with 20 lines |
| `^w<`               | move window size to left      |
| `^w>`               | move window size to right     |
| `^w-`               | move window size to up        |
| `^w+`               | move window size to down      |
| `^w=`               | reset window size             |
| `:qa`               | quit all windows              |

## Colors

```bash
# Set color scheme
:colorscheme <TAB>

# Color vimscripts
$ ls /usr/share/vim/vim90/colors
README.txt     default.vim    elflord.vim    industry.vim   lunaperche.vim pablo.vim      ron.vim        tools
blue.vim       delek.vim      evening.vim    koehler.vim    morning.vim    peachpuff.vim  shine.vim      torte.vim
darkblue.vim   desert.vim     habamax.vim    lists          murphy.vim     quiet.vim      slate.vim      zellner.vim
```
