# Materials

* [awesome ios @ github ](https://github.com/vsouza/awesome-ios)
* [awesome ios ui @ github](https://github.com/cjwirth/awesome-ios-ui)
* [iOS-Sample-Code @ github](https://github.com/xcatsan/iOS-Sample-Code)
* [ios-sample-apps @ github](https://github.com/ooyala/ios-sample-apps)

# Basic

## Install

Appstore 는 느리다. [xcodes](https://github.com/XcodesOrg/xcodes) command line 을
이용하면 빠르다.

```bash
$ brew install xcodesorg/made/xcodes
$ xcodes install --latest
```

## Shorcuts

| ShortCut | Description |
|:--------|:---------|
| `^ L Click` | Quick Help |
| `⇧ ⌘ O` | Open Quickly |
| `^ ⌥ ⌘ Enter` | Assistant View on storyboard |
| `⇧ ⌥ ⌘ Enter` | Enable Code Review on storyboard |
| `⇧ ⌘ L` | Show library window |
| `^ ⇧ ⌘ M` | Show/Hide Minimap |
| `⌘ B` | Build |
| `⌘ R` | Run |
| `^ ⌘ <-` | Back |
| `^ ⌘ ->` | Forward |
| `^ ⌘ J` | Jump to definition |
| `^ ⇧ ⌘ F` | Find selected symbol in workspace |
| `^ ⇧ ⌘ H` | Find call hierarchy |
| `⌘ 0` | Show/Hide Navigator |
| `⌥ ⌘ 0` | Show/Hide Inspector |

## Terms

* Workspace vs Project
  * an Xcode project is focused on building a specific product using
    a defined set of resources and settings, while a workspace allows you to
    work with multiple projects and their dependencies in a unified environment.
    Deciding which to use depends on the complexity of your application and the
    degree of modularity and collaboration required in your development process.

* Scheme
  * a collection of settings and configurations used to define how to build,
    run, test, profile, analyze, and archive your app.
  * The scheme content is stored in a `.xcscheme file`, which is located within
    your project's `.xcworkspace` or `.xcodeproj` directories. The `.xcscheme` file is
    an XML-based file that stores the information about your scheme, such as
    build targets, build configurations, and environment settings.
  * If your project uses an Xcode workspace (`.xcworkspace`): In this case, your
    scheme files are located inside the
    `<ProjectName>.xcworkspace/xcshareddata/xcschemes/` directory.
  * If your project uses an Xcode project file (`.xcodeproj`): Custom scheme
    files are located inside the
    `<ProjectName>.xcodeproj/xcuserdata/[username].xcuserdatad/xcschemes/`
    directory, while shared scheme files are in
    `<ProjectName>.xcodeproj/xcshareddata/xcschemes/`.

* Target
  * set of build configurations and settings used to create a specific product
    or output from your project, such as an app, a library, a framework, or a
    test bundle. Each target contains the necessary instructions on how to
    process your project's source code, resources, and dependencies to produce
    the desired outcome.
  * Saved in `project.pbxproj` and `.xcodeproj`

## File Extensions

Project View 의 File Extension 을 보이게 하자.

![](img/xcode_general_file_extension.png)

## Find References

검색을 원하는 함수에 cursor 를 놓는다. Editor View 왼쪽 위 아이콘을 클릭하고 Caller 를 확인한다.

![](img/editor_callers.png)

## How to make a iOS app with cocoapods 

[cocoapods create application](/cocoapods/README.md#using-pod-application-create)

## How to make a iOS library with cocoapods

[cocoapods create library](/cocoapods/README.md#using-pod-lib-create)
