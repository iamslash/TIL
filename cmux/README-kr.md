# Abstract

cmux 는 [Ghostty](https://ghostty.org) 터미널 엔진을 라이브러리(GhosttyKit)로 임베딩한
macOS 네이티브 터미널 앱이다. Swift/AppKit 으로 작성되어 Electron 앱보다 가볍고 빠르다.

Ghostty 가 제공하는 GPU 가속 터미널 렌더링 위에 세로 탭 사이드바, 알림 시스템, 내장
브라우저, SSH 원격 워크스페이스, 커스텀 명령어, Claude Code Teams 통합, CLI/소켓 API 등을
추가한다.

```
+-------------------------------------+
|          cmux (Swift/macOS)         |  <-- 앱 UI 레이어
|  세로탭, 사이드바, 알림, 브라우저,    |
|  워크스페이스, cmux.json, 소켓 API   |
+-------------------------------------+
|      GhosttyKit.xcframework         |  <-- 터미널 엔진 (Zig/C)
|  렌더링, 입력, 폰트, 색상, 테마      |
+-------------------------------------+
```

# Materials

- [cmux 공식 사이트](https://cmux.com)
- [cmux GitHub](https://github.com/manaflow-ai/cmux)
- [cmux 문서](https://cmux.com/docs/getting-started)
- [Ghostty 공식 사이트](https://ghostty.org)

# Install

## DMG

[cmux-macos.dmg](https://github.com/manaflow-ai/cmux/releases/latest/download/cmux-macos.dmg)
를 다운로드하고 응용 프로그램 폴더로 드래그한다. Sparkle 을 통해 자동 업데이트된다.

## Homebrew

```bash
brew tap manaflow-ai/cmux
brew install --cask cmux

# 업데이트
brew upgrade --cask cmux
```

# Configuration

cmux 에는 두 가지 설정 시스템이 있다.

## 터미널 외관 설정 (Ghostty Config)

Ghostty 설정 파일을 그대로 읽는다. `key = value` 형식이며 `#` 은 주석이다.

설정 파일 경로 (우선순위 순):

1. `~/.config/ghostty/config`
2. `~/.config/ghostty/config.ghostty`
3. `~/Library/Application Support/com.mitchellh.ghostty/config`
4. `~/Library/Application Support/com.cmuxterm.app/config`

### 주요 설정 항목

```bash
# 폰트
font-family = JetBrains Mono
font-size = 16

# 테마 (light/dark 자동 전환 가능)
theme = Catppuccin Mocha
# theme = light:Catppuccin Latte, dark:Catppuccin Mocha

# 색상
background = #1e1e2e
background-opacity = 0.95
foreground = #cdd6f4
cursor-color = #f5e0dc
cursor-text = #1e1e2e
selection-background = #45475a
selection-foreground = #cdd6f4

# 팔레트 (0~15)
palette = 0=#45475a
palette = 1=#f38ba8

# 스크롤백
scrollback-limit = 50000

# 분할 패널
unfocused-split-opacity = 0.85
unfocused-split-fill = #11111b
split-divider-color = #585b70

# 사이드바
sidebar-background = #1e1e2e
sidebar-tint-opacity = 0.8

# 작업 디렉토리
working-directory = ~/projects
```

설정 변경 후 적용 방법:

- 메뉴: **cmux > Reload Configuration**
- 단축키: `Cmd + Shift + ,`
- CLI: `cmux reload-config`

## 워크스페이스/커맨드 설정 (cmux.json)

프로젝트별 커스텀 명령어와 워크스페이스 레이아웃을 정의한다. JSON 형식이다.

설정 파일 경로:

- **로컬**: 프로젝트 디렉토리에서 위로 탐색하며 `cmux.json` 을 찾는다
- **글로벌**: `~/.config/cmux/cmux.json`
- 같은 이름의 명령어는 로컬이 우선한다

```json
{
  "commands": [
    {
      "name": "개발 서버",
      "description": "프론트엔드 개발 환경",
      "keywords": ["dev", "server"],
      "restart": "confirm",
      "workspace": {
        "name": "Dev",
        "cwd": "~/myapp",
        "color": "#FF6B6B",
        "layout": {
          "direction": "horizontal",
          "split": 0.6,
          "children": [
            {
              "pane": {
                "surfaces": [
                  { "type": "terminal", "command": "npm run dev" }
                ]
              }
            },
            {
              "pane": {
                "surfaces": [
                  { "type": "browser", "url": "http://localhost:3000" }
                ]
              }
            }
          ]
        }
      }
    },
    {
      "name": "배포",
      "description": "프로덕션 배포",
      "command": "make deploy",
      "confirm": true
    }
  ]
}
```

### 커맨드 필드

| 필드 | 설명 |
|---|---|
| `name` | 필수. 명령 팔레트에 표시되는 이름 |
| `description` | 설명 텍스트 |
| `keywords` | 명령 팔레트 검색 키워드 |
| `workspace` | 워크스페이스 생성 (`command` 와 배타적) |
| `command` | 셸 명령 실행 (`workspace` 와 배타적) |
| `restart` | 같은 이름 워크스페이스가 있을 때: `recreate`, `ignore`, `confirm` |
| `confirm` | `command` 실행 전 확인 다이얼로그 |

### 레이아웃 구조

- **pane**: `{ "pane": { "surfaces": [...] } }` -- 최소 1 개 surface 필요
- **split**: `{ "direction": "horizontal"|"vertical", "split": 0.5, "children": [노드, 노드] }` -- 정확히 2 개 children
- **surface type**: `terminal` 또는 `browser`
- `split` 값 범위: 0.1 ~ 0.9 (기본 0.5)

# Features

## 세로 탭 사이드바

Ghostty 는 가로 탭만 제공하지만 cmux 는 세로 사이드바에 워크스페이스 탭을 표시한다.
각 탭에서 다음 정보를 한눈에 볼 수 있다:

- git 브랜치 이름
- 연결된 PR 상태/번호
- 작업 디렉토리
- 수신 포트 (예: `localhost:3000`)
- 최근 알림 텍스트

## 알림 시스템

AI 코딩 에이전트(Claude Code 등)가 입력을 기다리면:

- 해당 패널에 파란색 링 표시
- 사이드바 탭 강조
- 알림 패널에서 대기 중인 알림 일괄 확인

```bash
# CLI 로 알림 보내기
cmux notify "빌드 완료"

# 터미널 시퀀스 OSC 9/99/777 자동 감지
# Claude Code 훅에 연결 가능
```

## 내장 브라우저

터미널 옆에 브라우저 패널을 분할로 띄울 수 있다.
[agent-browser](https://github.com/vercel-labs/agent-browser) 에서 포팅한 스크립팅
API 를 제공한다:

- 접근성 트리 스냅샷 가져오기
- 요소 클릭, 양식 채우기, JavaScript 실행
- 개발 서버 (`localhost:3000`) 직접 상호작용

Chrome, Firefox, Arc 등 20 개+ 브라우저에서 쿠키/세션을 가져와서 이미 로그인된 상태로
시작할 수 있다.

## SSH 원격 워크스페이스

```bash
cmux ssh user@remote-server
```

- 원격 머신 전용 워크스페이스 자동 생성
- 브라우저 패널이 원격 네트워크를 통해 라우팅되어 원격의 `localhost` 가 그대로 작동
- 이미지를 원격 세션에 드래그하면 scp 로 자동 업로드

## Claude Code Teams 통합

```bash
cmux claude-teams
```

Claude Code 의 팀원(teammate) 모드를 네이티브 분할 패널로 실행한다. tmux 없이
사이드바에 메타데이터와 알림이 표시된다.

## 분할 패널

패널 타입 3 종:

- **TerminalPanel** -- 터미널
- **BrowserPanel** -- 웹 브라우저
- **MarkdownPanel** -- 마크다운 렌더링

## CLI & 소켓 API

모든 것을 스크립트로 자동화할 수 있다:

```bash
cmux workspace create --name "API"
cmux split --direction right
cmux send-keys "npm test\n"
cmux notify "작업 완료"
cmux reload-config
```

## 세션 복원

앱 재시작 시 자동 복원:

- 창/워크스페이스/패널 레이아웃
- 작업 디렉토리
- 터미널 스크롤백
- 브라우저 URL 및 탐색 기록

활성 프로세스 상태(Claude Code 세션, vim 등)는 아직 복원되지 않는다.

# Ghostty vs cmux

| 영역 | Ghostty | cmux 가 추가한 것 |
|---|---|---|
| 터미널 렌더링 | O | (Ghostty 그대로 사용) |
| 탭 | 가로 탭 | 세로 사이드바 + git/PR/포트 표시 |
| 알림 | X | 알림 링 + 패널 + CLI |
| 브라우저 | X | 내장 브라우저 + 스크립팅 API |
| SSH | X | 원격 워크스페이스 + 브라우저 라우팅 |
| 커스텀 명령 | X | cmux.json + 명령 팔레트 |
| AI 에이전트 통합 | X | Claude Code Teams |
| 자동화 | X | CLI + 소켓 API |
| 세션 복원 | X | 레이아웃/URL/스크롤백 복원 |

# Keyboard Shortcuts

## 워크스페이스

| 단축키 | 동작 |
|---|---|
| `Cmd + N` | 새 워크스페이스 |
| `Cmd + 1~8` | 워크스페이스 1~8 로 이동 |
| `Cmd + 9` | 마지막 워크스페이스로 이동 |
| `Ctrl + Cmd + ]` | 다음 워크스페이스 |
| `Ctrl + Cmd + [` | 이전 워크스페이스 |
| `Cmd + Shift + W` | 워크스페이스 닫기 |
| `Cmd + Shift + R` | 워크스페이스 이름 변경 |
| `Cmd + B` | 사이드바 토글 |

## 서피스 (탭)

| 단축키 | 동작 |
|---|---|
| `Cmd + T` | 새 서피스 |
| `Cmd + Shift + ]` | 다음 서피스 |
| `Cmd + Shift + [` | 이전 서피스 |
| `Ctrl + Tab` | 다음 서피스 |
| `Ctrl + Shift + Tab` | 이전 서피스 |
| `Ctrl + 1~8` | 서피스 1~8 로 이동 |
| `Ctrl + 9` | 마지막 서피스로 이동 |
| `Cmd + W` | 서피스 닫기 |

## 분할 패널

| 단축키 | 동작 |
|---|---|
| `Cmd + D` | 오른쪽으로 분할 |
| `Cmd + Shift + D` | 아래로 분할 |
| `Alt + Cmd + Arrow` | 방향키로 패널 포커스 이동 |
| `Cmd + Shift + H` | 현재 패널 깜빡임 (위치 확인) |

## 브라우저

| 단축키 | 동작 |
|---|---|
| `Cmd + Shift + L` | 분할 패널로 브라우저 열기 |
| `Cmd + L` | 주소창 포커스 |
| `Cmd + [` | 뒤로 |
| `Cmd + ]` | 앞으로 |
| `Cmd + R` | 페이지 새로고침 |
| `Alt + Cmd + I` | 개발자 도구 열기 |
| `Alt + Cmd + C` | JavaScript 콘솔 표시 |

## 알림

| 단축키 | 동작 |
|---|---|
| `Cmd + I` | 알림 패널 표시 |
| `Cmd + Shift + U` | 최근 읽지 않은 알림으로 이동 |

## 찾기

| 단축키 | 동작 |
|---|---|
| `Cmd + F` | 찾기 |
| `Cmd + G` | 다음 찾기 |
| `Cmd + Shift + G` | 이전 찾기 |
| `Cmd + Shift + F` | 찾기 바 숨기기 |
| `Cmd + E` | 선택한 텍스트로 찾기 |

## 터미널

| 단축키 | 동작 |
|---|---|
| `Cmd + K` | 스크롤백 지우기 |
| `Cmd + C` | 복사 (선택 시) |
| `Cmd + V` | 붙여넣기 |
| `Cmd + +` | 글꼴 크기 확대 |
| `Cmd + -` | 글꼴 크기 축소 |
| `Cmd + 0` | 글꼴 크기 초기화 |

## 창

| 단축키 | 동작 |
|---|---|
| `Cmd + Shift + N` | 새 창 |
| `Cmd + ,` | 설정 |
| `Cmd + Shift + ,` | 설정 다시 불러오기 |
| `Cmd + Q` | 종료 |
