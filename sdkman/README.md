# Abstract

여러가지 SDK 의 version management application. 주로 JAVA 관련된 SDK 에
사용한다.

# Materials

* [sdkman.io](https://sdkman.io/)
* [SDK! 으로 Java 버전 관리하기](https://phoby.github.io/sdkman/)

# Basic

## Install

```bash
$ curl -s "https://get.sdkman.io" | bash

# Add this line to .zshrc
$ source "$HOME/.sdkman/bin/sdkman-init.sh"

# Uninstall sdkman
$ rm -rf ~/.sdkman
```

## Usages

```bash
$ sdk version

# Show available JDK
$ sdk list java

# Unstinall JDK
$ sdk uninstall java 8.0.192-zulu

# Install JDK
$ sdk install java 11.0.17-amzn

# Use JDK
$ sdk use java 11.0.17-amzn

# Show current JDK
$ sdk current
```
