# Abstract

zsh 에 대해 정리한다.

# Install

* [oh-my-zsh](https://github.com/robbyrussell/oh-my-zsh)

----

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```

# Features

## Auto completetion

* `git a` 하고 `tab` 하면 argument 까지도 auto completetion 이 된다.

## Shortcus

* [zsh shortcut @ gist](https://gist.github.com/acamino/2bc8df5e2ed0f99ddbe7a6fddb7773a6)

| shortcut                      | action                                                             |
| ----------------------------- | ------------------------------------------------------------------ |
| `CTRL + A`                    | move to the beginning of the line                                  |
| `CTRL + E`                    | move to the end of the line                                        |
| `CTRL + [left arrow], ALT-B`  | move one word backward                                             |
| `CTRL + [right arrow], ALT-F` | move one word forward                                              |
| `CTRL + U`                    | clear the entire line                                              |
| `CTRL + K`                    | clear the characters on the line after the current cursor position |
| `CTRL + W`                    | delete the word in front of the cursor                             |
| `Alt + D`                     | delete the word after the cursor                                   |
| `CTRL + R`                    | search history                                                     |
| `CTRL + G`                    | escape from search mode                                            |
| `CTRL + -`                    | undo the last change                                               |
| `CTRL + L`                    | clear the screen                                                   |
| `CTRL + S`                    | clear the screen                                                   |
| `CTRL + Q`                    | reenable the screen                                                |
| `CTRL + C`                    | terminate current foreground process                               |
| `CTRL + Z`                    | suspend  current foreground process                                |

