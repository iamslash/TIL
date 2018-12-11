- [Abstract](#abstract)
- [Materials](#materials)
- [Basic Usages](#basic-usages)
  - [Commands](#commands)
  - [Variables](#variables)
    - [Parameter Expansion](#parameter-expansion)
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
  - [Decision Making](#decision-making)
    - [If Statement](#if-statement)
    - [If, Else Statement](#if-else-statement)
    - [Nested If Statements](#nested-if-statements)
  - [Return Code](#return-code)
  - [Loops](#loops)
    - [While Statement Implementation](#while-statement-implementation)
    - [For Statement - List Implementation](#for-statement---list-implementation)
    - [Looping through Ranges](#looping-through-ranges)
    - [Classic for Loop Implementation](#classic-for-loop-implementation)
    - [Looping through Command Line Arguments](#looping-through-command-line-arguments)
    - [Break Statement Implementation](#break-statement-implementation)
  - [Functions](#functions)
    - [Calling a Function](#calling-a-function)
    - [Functions with Parameters](#functions-with-parameters)
    - [Functions with Return Values](#functions-with-return-values)
    - [Local Variables in Functions](#local-variables-in-functions)
    - [Recursive Functions](#recursive-functions)

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

7	**DIR**
This batch command lists the contents of a directory.

8	**DATE**
This batch command help to find the system date.

9	**ECHO**
This batch command displays messages, or turns command echoing on or off.

10	**EXIT**
This batch command exits the DOS console.

11	**MD**
This batch command creates a new directory in the current location.

12	**MOVE**
This batch command moves files or directories between directories.

13	**PATH**
This batch command displays or sets the path variable.

14	**PAUSE**
This batch command prompts the user and waits for a line of input to be entered.

15	**PROMPT**
This batch command can be used to change or reset the cmd.exe prompt.

16	**RD**
This batch command removes directories, but the directories need to be empty before they can be removed.

17	**REN**
Renames files and directories

18	**REM**
This batch command is used for remarks in batch files, preventing the content of the remark from being executed.

19	**START**
This batch command starts a program in new window, or opens a document.

20	**TIME**
This batch command sets or displays the time.

21	**TYPE**
This batch command prints the content of a file or files to the output.

22	**VOL**
This batch command displays the volume labels.

23	**ATTRIB**
Displays or sets the attributes of the files in the curret directory

24	**CHKDSK**
This batch command checks the disk for any problems.

25	**CHOICE**
This batch command provides a list of options to the user.

26	**CMD**
This batch command invokes another instance of command prompt.

27	**COMP**
This batch command compares 2 files based on the file size.

28	**CONVERT**
This batch command converts a volume from FAT16 or FAT32 file system to NTFS file system.

29	**DRIVERQUERY**
This batch command shows all installed device drivers and their properties.

30	**EXPAND**
This batch command extracts files from compressed .cab cabinet files.

31	**FIND**
This batch command searches for a string in files or input, outputting matching lines.

32	**FORMAT**
This batch command formats a disk to use Windows-supported file system such as FAT, FAT32 or NTFS, thereby overwriting the previous content of the disk.

33	**HELP**
This batch command shows the list of Windows-supplied commands.

34	**IPCONFIG**
This batch command displays Windows IP Configuration. Shows configuration by connection and the name of that connection.

35	**LABEL**
This batch command adds, sets or removes a disk label.

36	**MORE**
This batch command displays the contents of a file or files, one screen at a time.

37	**NET**
Provides various network services, depending on the command used.

38	**PING**
This batch command sends ICMP/IP "echo" packets over the network to the designated address.

39	**SHUTDOWN**
This batch command shuts down a computer, or logs off the current user.

40	**SORT**
This batch command takes the input from a source file and sorts its contents alphabetically, from A to Z or Z to A. It prints the output on the console.

41	**SUBST**
This batch command assigns a drive letter to a local folder, displays current assignments, or removes an assignment.

42	**SYSTEMINFO**
This batch command shows configuration of a computer and its operating system.

43	**TASKKILL**
This batch command ends one or more tasks.

44	**TASKLIST**
This batch command lists tasks, including task name and process id (PID).

45	**XCOPY**
This batch command copies files and directories in a more advanced way.

46	**TREE**
This batch command displays a tree of all subdirectories of the current directory to any level of recursion or depth.

47	**FC**
This batch command lists the actual differences between two files.

48	**DISKPART**
This batch command shows and configures the properties of disk partitions.

49	**TITLE**
This batch command sets the title displayed in the console window.

50	**SET**
Displays the list of environment variables on the current system.

## Variables

### Parameter Expansion

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

* [Windows Environment Variables](https://ss64.com/nt/syntax-variables.html)

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

## Decision Making

### If Statement

* Checking Integer Variables

```cmd
@echo off 
SET /A a=5 
SET /A b=10 
SET /A c=%a% + %b% 
if %c%==15 echo "The value of variable c is 15" 
if %c%==10 echo "The value of variable c is 10"
REM 15
```

* Checking String Variables

```cmd
@echo off 
SET str1=String1 
SET str2=String2 
if %str1%==String1 echo "The value of variable String1" 
if %str2%==String3 echo "The value of variable c is String3"
REM "The value of variable String1"
```

* Checking Command Line Arguments

```cmd
@echo off 
echo %1 
echo %2 
echo %3 
if %1%==1 echo "The value is 1" 
if %2%==2 echo "The value is 2" 
if %3%==3 echo "The value is 3"
REM test.bat 1 2 3
REM
REM 1 
REM 2 
REM 3 
REM "The value is 1" 
REM "The value is 2" 
REM "The value is 3"
```

### If, Else Statement

* Checking Integer Variables

```cmd
@echo off 
SET /A a=5 
SET /A b=10
SET /A c=%a% + %b% 
if %c%==15 (echo "The value of variable c is 15") else (echo "Unknown value") 
if %c%==10 (echo "The value of variable c is 10") else (echo "Unknown value")
REM "The value of variable c is 15" 
REM "Unknown value"
```

* Checking String Variables

```cmd
@echo off 
SET str1=String1 
SET str2=String2 

if %str1%==String1 (echo "The value of variable String1") else (echo "Unknown value") 

if %str2%==String3 (echo "The value of variable c is String3") else (echo "Unknown value")

REM "The value of variable String1" 
REM "Unknown value"
```

* Checking Command Line Arguments

```cmd
@echo off 
echo %1 
echo %2 
echo %3 
if %1%==1 (echo "The value is 1") else (echo "Unknown value") 
if %2%==2 (echo "The value is 2") else (echo "Unknown value") 
if %3%==3 (echo "The value is 3") else (echo "Unknown value")
```

* if defined

```cmd
@echo off 
SET str1=String1 
SET str2=String2 
if defined str1 echo "Variable str1 is defined"

if defined str3 (echo "Variable str3 is defined") else (echo "Variable str3 is not defined")

REM "Variable str1 is defined" 
REM "Variable str3 is not defined"
```

* if exists

```cmd
@echo off 
if exist C:\set2.txt echo "File exists" 
if exist C:\set3.txt (echo "File exists") else (echo "File does not exist")

REM "File exists"
REM "File does not exist"
```

### Nested If Statements

```cmd
@echo off
SET /A a=5
SET /A b=10
if %a%==5 if %b%==10 echo "The value of the variables are correct"

REM "The value of the variables are correct"

@echo off 
SET /A a=5 

if %a%==5 goto :labela 
if %a%==10 goto :labelb

:labela 
echo "The value of a is 5" 

exit /b 0

:labelb 
echo "The value of a is 10"

REM "The value of a is 5" 
```

## Return Code

```cmd
IF %ERRORLEVEL% NEQ 0 ( 
   DO_Something 
)

if not exist c:\lists.txt exit 7 
if not defined userprofile exit 9 
exit 0

Call Find.cmd

if errorlevel gtr 0 exit 
echo “Successful completion”
```

## Loops

### While Statement Implementation

```cmd
REM Set counters
REM :label
REM If (expression) (
REM    Do_something
REM    Increment counter
REM    Go back to :label
REM )

@echo off
SET /A "index=1"
SET /A "count=5"
:while
if %index% leq %count% (
   echo The value of index is %index%
   SET /A "index=index + 1"
   goto :while
)

REM The value of index is 1
REM The value of index is 2
REM The value of index is 3
REM The value of index is 4
REM The value of index is 5
```

### For Statement - List Implementation

```cmd
REM FOR %%variable IN list DO do_something

@echo off 
FOR %%F IN (1 2 3 4 5) DO echo %%F

REM 1 
REM 2 
REM 3 
REM 4 
REM 5
```

### Looping through Ranges

* `/L` is for range option
  
```cmd
REM FOR /L %%variable IN (lowerlimit,Increment,Upperlimit) DO do_something

@ECHO OFF 
FOR /L %%X IN (0,1,5) DO ECHO %%X

REM 0 
REM 1 
REM 2 
REM 3 
REM 4 
REM 5
```

### Classic for Loop Implementation

```cmd
@echo off 
SET /A i=1 
:loop 

IF %i%==5 GOTO END 
echo The value of i is %i% 
SET /a i=%i%+1 
GOTO :LOOP 
:END

REM The value of i is 1 
REM The value of i is 2 
REM The value of i is 3 
REM The value of i is 4
```

### Looping through Command Line Arguments

```cmd
@ECHO OFF 
:Loop 

IF "%1"=="" GOTO completed 
FOR %%F IN (%1) DO echo %%F 
SHIFT 
GOTO Loop 
:completed

REM 1 
REM 2 
REM 3
```

### Break Statement Implementation

```cmd
@echo off 
SET /A "index=1" 
SET /A "count=5" 
:while 
if %index% leq %count% ( 
   if %index%==2 goto :Increment 
      echo The value of index is %index% 
:Increment 
   SET /A "index=index + 1" 
   goto :while 
)

REM The value of index is 1 
REM The value of index is 3 
REM The value of index is 4 
REM The value of index is 5
```

## Functions

### Calling a Function

```cmd
REM call :function_name

@echo off 
SETLOCAL 
CALL :Display 
EXIT /B %ERRORLEVEL% 
:Display 
SET /A index=2 
echo The value of index is %index% 
EXIT /B 0

REM The value of index is 2
```


### Functions with Parameters

```cmd
REM Call :function_name parameter1, parameter2… parametern

@echo off
SETLOCAL
CALL :Display 5, 10
EXIT /B %ERRORLEVEL%
:Display
echo The value of parameter 1 is %~1
echo The value of parameter 2 is %~2
EXIT /B 0

REM The value of parameter 1 is 5
REM The value of parameter 2 is 10
```


### Functions with Return Values

```cmd
@echo off
SETLOCAL
CALL :SetValue value1,value2
echo %value1%
echo %value2%
EXIT /B %ERRORLEVEL%
:SetValue
set "%~1=5"
set "%~2=10"
EXIT /B 0

REM 5 
REM 10
```

### Local Variables in Functions

```cmd
@echo off
set str=Outer
echo %str%
CALL :SetValue str
echo %str%
EXIT /B %ERRORLEVEL%
:SetValue
SETLOCAL
set str=Inner
set "%~1=%str%"
ENDLOCAL
EXIT /B 0

REM Outer
REM Outer
```

### Recursive Functions

```cmd
@echo off
set "fst=0"
set "fib=1"
set "limit=1000000000"
call:myFibo fib,%fst%,%limit%
echo.The next Fibonacci number greater or equal %limit% is %fib%.
echo.&pause&goto:eof
:myFibo -- calculate recursively
:myFibo -- calculate recursively the next Fibonacci number greater or equal to a limit
SETLOCAL
set /a "Number1=%~1"
set /a "Number2=%~2"
set /a "Limit=%~3"
set /a "NumberN=Number1 + Number2"

if /i %NumberN% LSS %Limit% call:myFibo NumberN,%Number1%,%Limit%
(ENDLOCAL
   IF "%~1" NEQ "" SET "%~1=%NumberN%"
)goto:eof

REM The next Fibonacci number greater or equal 1000000000 is 1134903170.
```

