# Abstract

두가지 Java Object 를 쉽게 converting 할 수 있게 해준다.

# Materials

* [Quick Guide to MapStruct](https://www.baeldung.com/mapstruct)


# Basic

`@Mapper` interface 를 선언하기만 해도 `FooSrc, FooDst` object 를 converting 할 수 있다.

```java
public class FooSrc {
    private String name;
    private String email; 
}

public class FooDst {
    private String name;
    private String email; 
}

@Mapper
public interface FooMapper {
    FooDst of(FooSrc src);
    FooSrc of(FooDst dst); 
}
```

`FooSrc, FooDst` 의 field 이름들이 서로 다르다면 다음과 같이 한다.

```java
public class FooSrc {
    private String fooName;
    private String fooEmail; 
}

public class FooDst {
    private String name;
    private String email; 
}

@Mapper
public interface FooMapper {
    @Mappings({
        @Mapping(target="fooName", source="src.name"),
        @Mapping(target="fooEmail", source="src.email"),
    })
    FooDst of(FooSrc src);
    @Mappings({
        @Mapping(target="name", source="dst.fooName"),
        @Mapping(target="email", source="dst.fooEmail"),
    })    
    FooSrc of(FooDst dst); 
}
```
