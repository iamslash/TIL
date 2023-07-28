- [Materials](#materials)
- [Basic](#basic)
  - [.vimrc](#vimrc)
- [Advanced](#advanced)
  - [Split windows](#split-windows)

-----

# Materials

* [vim_robin @ github](https://github.com/iamslash/vim_robin)
* [밤앙개의 vim 강좌 33 - vim 플러그인 01 : vim script와 plugin이란, <Leader>란? | naver](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=nfwscho&logNo=220530572978)

# Basic

## .vimrc

```vimrc
colorscheme desert
set paste
```

# Advanced

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
| `:qa`               | quit all windows             |
