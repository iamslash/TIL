- [Abstract](#abstract)
- [Materials](#materials)
- [Basic Usages](#basic-usages)
  - [Commands](#commands)
  - [Variables](#variables)
    - [Commandline Arguments](#commandline-arguments)
    - [Set Command](#set-command)
    - [Working with Numeric Values](#working-with-numeric-values)
    - [Local vs Global Variables](#local-vs-global-variables)
    - [Working with Environment Variables](#working-with-environment-variables)
  - [String](#string)
    - [Create String](#create-string)
    - [Empty String](#empty-string)
    - [String Interpolation](#string-interpolation)
    - [String Concatenation](#string-concatenation)
    - [String length](#string-length)
    - [toInt](#toint)
    - [Align Right](#align-right)
    - [Left String](#left-string)
    - [Mid String](#mid-string)
    - [Right String](#right-string)
    - [Remove](#remove)
    - [Remove Both Ends](#remove-both-ends)
    - [Remove All Spaces](#remove-all-spaces)
    - [Replace a String](#replace-a-string)
    - [Expansion](#expansion)

---

# Abstract

cmd.exe 에 대해 정리한다.

# Materials

* [batch script @ tutorialpoint](https://www.tutorialspoint.com/batch_script/batch_script_environment.htm)
* [An A-Z Index of the Windows CMD command line](https://ss64.com/nt/)
* [Windows CMD Shell Command Line Syntax](https://ss64.com/nt/syntax.html)

# Basic Usages

## Commands

1	**VER**
This batch command shows the version of MS-DOS you are using.

2	**ASSOC**
This is a batch command that associates an extension with a file type (FTYPE), displays existing associations, or deletes an association.

3	**CD**
This batch command helps in making changes to a different directory, or displays the current directory.

4	**CLS**
This batch command clears the screen.

5	**COPY**
This batch command is used for copying files from one location to the other.

6	**DEL**
This batch command deletes files and not directories.

7	DIR
This batch command lists the contents of a directory.

8	DATE
This batch command help to find the system date.

9	ECHO
This batch command displays messages, or turns command echoing on or off.

10	EXIT
This batch command exits the DOS console.

11	MD
This batch command creates a new directory in the current location.

12	MOVE
This batch command moves files or directories between directories.

13	PATH
This batch command displays or sets the path variable.

14	PAUSE
This batch command prompts the user and waits for a line of input to be entered.

15	PROMPT
This batch command can be used to change or reset the cmd.exe prompt.

16	RD
This batch command removes directories, but the directories need to be empty before they can be removed.

17	REN
Renames files and directories

18	REM
This batch command is used for remarks in batch files, preventing the content of the remark from being executed.

19	START
This batch command starts a program in new window, or opens a document.

20	TIME
This batch command sets or displays the time.

21	TYPE
This batch command prints the content of a file or files to the output.

22	VOL
This batch command displays the volume labels.

23	ATTRIB
Displays or sets the attributes of the files in the curret directory

24	CHKDSK
This batch command checks the disk for any problems.

25	CHOICE
This batch command provides a list of options to the user.

26	CMD
This batch command invokes another instance of command prompt.

27	COMP
This batch command compares 2 files based on the file size.

28	CONVERT
This batch command converts a volume from FAT16 or FAT32 file system to NTFS file system.

29	DRIVERQUERY
This batch command shows all installed device drivers and their properties.

30	EXPAND
This batch command extracts files from compressed .cab cabinet files.

31	FIND
This batch command searches for a string in files or input, outputting matching lines.

32	FORMAT
This batch command formats a disk to use Windows-supported file system such as FAT, FAT32 or NTFS, thereby overwriting the previous content of the disk.

33	HELP
This batch command shows the list of Windows-supplied commands.

34	IPCONFIG
This batch command displays Windows IP Configuration. Shows configuration by connection and the name of that connection.

35	LABEL
This batch command adds, sets or removes a disk label.

36	MORE
This batch command displays the contents of a file or files, one screen at a time.

37	NET
Provides various network services, depending on the command used.

38	PING
This batch command sends ICMP/IP "echo" packets over the network to the designated address.

39	SHUTDOWN
This batch command shuts down a computer, or logs off the current user.

40	SORT
This batch command takes the input from a source file and sorts its contents alphabetically, from A to Z or Z to A. It prints the output on the console.

41	SUBST
This batch command assigns a drive letter to a local folder, displays current assignments, or removes an assignment.

42	SYSTEMINFO
This batch command shows configuration of a computer and its operating system.

43	TASKKILL
This batch command ends one or more tasks.

44	TASKLIST
This batch command lists tasks, including task name and process id (PID).

45	XCOPY
This batch command copies files and directories in a more advanced way.

46	TREE
This batch command displays a tree of all subdirectories of the current directory to any level of recursion or depth.

47	FC
This batch command lists the actual differences between two files.

48	DISKPART
This batch command shows and configures the properties of disk partitions.

49	TITLE
This batch command sets the title displayed in the console window.

50	SET
Displays the list of environment variables on the current system.

## Variables


### Commandline Arguments

```cmd
@echo off 
echo %1 
echo %2 
echo %3
REM Test.bat 1 2 3
```

### Set Command

```cmd
set /A variable-name=value
REM /A – This switch is used if the value needs to be numeric in nature.

@echo off 
set message=Hello World 
echo %message%
```

### Working with Numeric Values

```cmd
@echo off 
SET /A a=5 
SET /A b=10 
SET /A c=%a% + %b% 
echo %c%
REM 15

@echo off 
SET /A a=5 
SET /A b=10 
SET /A c=%a% + %b% 
echo %c% 
SET /A c=%a% - %b% 
echo %c% 
SET /A c=%b% / %a% 
echo %c% 
SET /A c=%b% * %a% 
echo %c%
REM 15 
REM -5 
REM 2 
REM 50
```

### Local vs Global Variables

* Local Variable 을 선언하기 위해서는 `set` 을 `SETLOCAL` 과 `ENDLOCAL` 사이에 선언한다.

```cmd
@echo off 
set globalvar=5
SETLOCAL
set var=13145
set /A var=%var% + 5
echo %var%
echo %globalvar%
ENDLOCAL
```

### Working with Environment Variables

```cmd
@echo off 
echo %PATH%
REM D:\local\bin
```

## String

### Create String

```cmd
@echo off 
:: This program just displays Hello World 
set message=Hello World 
echo %message%
REM Hello World
```

### Empty String

```cmd
@echo off 
SET a= 
SET b=Hello 
if [%a%]==[] echo "String A is empty" 
if [%b%]==[] echo "String B is empty "
REM String A is empty
```

### String Interpolation

```cmd
@echo off 
SET a=Hello 
SET b=World 
SET /A d=50 
SET c=%a% and %b% %d%
echo %c%
REM Hello and World 50
```

### String Concatenation

```cmd
@echo off 
SET a=Hello 
SET b=World 
SET c=%a% and %b% 
echo %c%
REM Hello and World
```

### String length

* [EnableDelayedExpansion](https://ss64.com/nt/delayedexpansion.html)

```cmd
@echo off
set str=Hello World
call :strLen str strlen
echo String is %strlen% characters long
exit /b

:strLen
setlocal enabledelayedexpansion

:strLen_Loop
   if not "!%1:~%len%!"=="" set /A len+=1 & goto :strLen_Loop
(endlocal & set %2=%len%)
goto :eof
REM 1 2 3
```

### toInt

```cmd
@echo off
set var=13145
set /A var=%var% + 5
echo %var%
REM 13150
```

### Align Right

```cmd
@echo off 
set x=1000 
set y=1 
set y=%y% 
echo %x% 

set y=%y:~-4% 
echo %y%
REM 1000
REM 1
```

### Left String

```cmd
@echo off 
set str=Helloworld 
echo %str% 

set str=%str:~0,5% 
echo %str%
REM Helloworld 
REM Hello
```

### Mid String

```cmd
@echo off 
set str=Helloworld 
echo %str%

set str=%str:~5,10% 
echo %str%
REM Helloworld 
REM world
```

### Right String

```cmd
@echo off 
set str=This message needs changed. 
echo %str% 

set str=%str:~-8% 
echo %str%
REM This message needs changed. 
REM changed.
```

### Remove 

```cmd
@echo off 
set str=Batch scripts is easy. It is really easy. 
echo %str% 

set str=%str:is=% 
echo %str%
REM Batch scripts is easy. It is really easy. 
REM Batch scripts easy. It really easy.
```

### Remove Both Ends

```cmd
@echo off 
set str=Batch scripts is easy. It is really easy 
echo %str% 

set str=%str:~1,-1% 
echo %str%
REM Batch scripts is easy. It is really easy 
REM atch scripts is easy. It is really eas
```

### Remove All Spaces

```cmd
@echo off 
set str=This string    has    a  lot  of spaces 
echo %str% 

set str=%str:=% 
echo %str%
REM This string    has    a  lot  of spaces
REM Thisstringhasalotofspaces
```

### Replace a String

```cmd
@echo off 
set str=This message needs changed. 
echo %str% 

set str=%str:needs=has% 
echo %str%
REM This message needs changed. 
REM This message has changed.
```

### Expansion

```cmd
> for /?
    %~I         - 따옴표(")를 제거하는 %I을 확장합니다.
    %~fI        - %I을 정규화된 경로 이름으로 확장합니다.
    %~dI        - %I을 드라이브 문자로만 확장합니다.
    %~pI        - %I을 경로로만 확장합니다.
    %~nI        - %I을 파일 이름으로만 확장합니다.
    %~xI        - %I을 파일 확장명으로만 확장합니다.
    %~sI        - 확장된 경로가 짧은 이름만 가지고 있습니다.
    %~aI        - %I이 파일의 파일 속성으로만 확장합니다.
    %~tI        - %I을 파일의 날짜/시간으로만 확장합니다.
    %~zI        - %I을 파일 크기로만 확장합니다.
    %~$PATH:I   - PATH 환경 변수 목록에 있는
                   디렉터리를 찾고 %I을 처음으로 찾은
                   정규화된 이름으로 확장합니다.
                   환경 변수 이름이 정의되지 않았거나
                   찾기에서 파일을 찾지 못하면
                   이 구문에서 빈 문자열로
                   확장합니다

위의 구문은 여러 가지 결과를 얻기 위해 결합될 수 있습니다.

    %~dpI       - %I을 드라이브 문자와 경로로만 확장합니다.
    %~nxI       - %I을 파일 이름과 확장명으로만 확장합니다.
    %~fsI       - %I 을 짧은 이름을 가진 전체 경로 이름으로만 확장합니다.
    %~dp$PATH:i - %I에 대한 PATH 환경 변수 목록에 있는
                   디렉터리를 찾고 처음 찾은 것의
                   드라이브 문자와 경로로 확장합니다.
    %~ftzaI     - %I을 출력줄과 같은 DIR로 확장합니다.
```