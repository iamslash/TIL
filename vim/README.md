- [Materials](#materials)
- [Basic](#basic)
  - [Modes](#modes)
  - [`.vimrc`](#vimrc)
  - [Basic Commands](#basic-commands)
  - [VIM Script](#vim-script)
  - [Plugins](#plugins)
  - [NERDTree Plugin](#nerdtree-plugin)
  - [Split Windows](#split-windows)
  - [Colors](#colors)

-----

# Materials

* [vim_robin @ github](https://github.com/iamslash/vim_robin)
* [밤앙개의 vim 강좌 33 - vim 플러그인 01 : vim script와 plugin이란, <Leader>란? | naver](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=nfwscho&logNo=220530572978)

# Basic

## Modes

* Command mode: `ESC`
* Command line mode: `:`
* Visual mode: `v` `V` `CTRL-v` `CTRL-q`
* Insert mode: `i` `a`

## `.vimrc`

```bash
# Echo my vimrc
:echo $MYVIMRC
/Users/david.s/.vimrc

# Source my vimrc
:source $MYVIMRC
```

## Basic Commands

```bash
# Show help page
:help

# Show versions
:version

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
# Remove hilight
:noh
,/  # When <Leader> is ,

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
# Set the window max
CTRL-w_

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

# VIM has two options such as on/off and value option.
# Show current options
:set
# Show all options
:set all
# Turn off option
:set no{option}
# Show specific option
# :set {option}?
:set number? 
# Show local options
:setlocal
# Set number option on
:set number
# Set number option off
:set nonumber
# Set options with value
:set clipboard=unnamed
:set tabstop=8
```

## VIM Script

`/usr/share/vim/vim90/vimrc_example.vim` 을 분석하자.

> Internal variables

```
v: vi predefined variable
g: global variable
l: local variable
b: buffer variable
w: window variable
t: tab page variable
s: script variable
a: function argument 
```

> Evaluation

| | ignorecase  | case sensitive |
|--|--|--|
| equal | `==?` | `==#` | 
| not equal | `!=?` | `!=#` | 
| great | `>?` | `>#` | 
| great equal | `>=?` | `>=#` | 
| less | `<?` | `<#` | 
| less equal | `<=?` | `<=#` | 
| **regex match** | `=~?` | `=~#` | 
| regex not match | `!~?` | `!~#` | 

```bash
:echo "evim" =~? "evim"
1
:echo "eVim" =~? "evim"
1
:echo "v:prograname"
VIM

# Has vms feature?
:echo has("vms")
0
:echo has("syntax")
1
```

> Options

```bash
# Set backspace option
# Delete indentation, end of line, start
:set backspace=indent,eol,start
# Create backup file which ends with ~
:set backup
# Set history lines
:set history=50
# Show line, column of the cursor
#:set noruler
:set ruler
# Show not completed command
:set showcmd
# Show search progress
#:set noincsearch
:set incsearch
# Set syntax hight on
:syntax on
# Show searched word
:set hlsearch
```

> Miscellaneous

```bash
# Has autocmd feature?
:has("autocmd")

# Show filetype options
# detections:ON means executing $VIMRUNTIME/filetype.vim
# plugin:ON means executing $VIMRUNTIME/ftplugin.vim
# indent:ON means executing $VIMRUNTIME/indent.vim
:filetype
filetype detection:ON  plugin:ON  indent:ON
# Turn off filetype detection
:filetype off
# Turn off filetype plugin
:filetype plugin off
# Turn off filetype indent
:filetype indent off

# Like ":set" but set only the value 
# local to the current buffer or window.
:setlocal

# autocmd <event-name> <pattern> <command>
:autocmd
# Show autocmd event-names
:help autocmd-events
|BufNewFile|		starting to edit a file that does not exist
|BufReadPre|		starting to edit a new buffer, before reading the file
|BufRead|		starting to edit a new buffer, after reading the file
|BufReadPost|		starting to edit a new buffer, after reading the file
|BufReadCmd|		before starting to edit a new buffer |Cmd-event|
# Show commands
:command

# Show augroup
# augroup means a group of autocmd
:augroup
# Delete previous autocmds in this augroup
:au!

# Copy indent from current line when starting a new line (typing <CR>
# in Insert mode or when using the "o" or "O" command).
:set autoindent
```

## Plugins

There two plugin types such as **filetype plugin**, **global plugin**.

**Filetype plugin** 

* Works just for the specific file type. 
* Saved in `$VIMRUNTIME/plugin` dir.
 
**Global plugin**
* works whole files. 
* Saved in `$VIMRUNTIME/ftplugin` dir.

Can execute plugin using `<Leader>` key. `<Leader>` is a defined key in a
`mapleader` variable. `,` is recommended.

`vundle` is a plugin which is a plugin manager.

This is how to install plugins. Paste these lines and Update them. This will
install `VundleVim`, `The-NERD-tree`, `OmniCppComplete`, `csharp.vim`.

```sh
" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
Plugin 'The-NERD-tree'
Plugin 'OmniCppComplete'
Plugin 'csharp.vim'

" All of your Plugins must be added before the following line
call vundle#end()            " required
```

`:PluginInstall` will install them.

Comment lines which start with `Plugin`. `:PluginClean` will uninstall them.

`:PluginList` will show installed plugins.

## NERDTree Plugin

This is the best explorer plugin.

```bash
# Execute NERDTree plugin
:NERDTree
# Move the cursor
h, j, k, l
# Open dirs, files
<Enter>
# Quit
q
# Help
?
# Change the root directory
C

# Show file menu
m

# Change Directory
:cd $HOME
# Change NERDTREE root to current working directory  
CD
# Show bookmarks
B
# Register bookmark on the files, dirs.
:Bookmark
# Delete bookmarks on the bookmarks
D

# Toggle hidden files
I
# Change working directory on the dir
cd
# Change NERDTree root to current working directory  
CD
# Toggle extracing all nodes
o
# Toggle NERDTree full screen
A
# Refresh
R
```

## Split Windows

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
