# Materials

* [[8장 빌드 자동화와 Ant] 1. Ant 환경 설정과 매뉴얼](https://www.youtube.com/watch?v=26EGMpcQgvA&list=PLo18BfwYMpEIurLDszlgC4OiQdhxykgKr&index=44)
* [Apache Ant™ 1.10.7 Manual @ apache](https://ant.apache.org/manual/)

# Basic

## Configure

```
set ANT_HOME=C:\NVPACK\apache-ant-1.8.2
set JAVA_HOME=C:\Program Files\Java\jdk1.8.0_31
```

## Structure of a project

```
- build - classes  - iamslash - HelloWorld.class
        - jar      - HelloWorld.jar
- src   - iamslash - HelloWorld.java  
  build.xml
```

## Hello World

```bash
# make a structure of a project
$ cd ~/my/java
$ mkdir Hello
$ mkdir Hello/src
$ mkdir Hello/src/iamslash
$ cd Hello
$ code src/iamslash/HelloWorld.java
$ mkdir build/classes
$ javac -sourcepath src -d build\classes src\iamslash\HelloWorld.java
$ java -cp build\classes iamslash.HelloWorld
# build and run using ant
$ ant compile
$ ant jar
$ ant run
$ ant compile jar run
```

* HelloWorld.java

```java
package iamslash;

public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
```

* build.xml

```xml
<project>

    <target name="clean">
        <delete dir="build"/>
    </target>

    <target name="compile">
        <mkdir dir="build/classes"/>
        <javac srcdir="src" destdir="build/classes"/>
    </target>

    <target name="jar">
        <mkdir dir="build/jar"/>
        <jar destfile="build/jar/HelloWorld.jar" basedir="build/classes">
            <manifest>
                <attribute name="Main-Class" value="iamslash.HelloWorld"/>
            </manifest>
        </jar>
    </target>

    <target name="run">
        <java jar="build/jar/HelloWorld.jar" fork="true"/>
    </target>

</project>
```

