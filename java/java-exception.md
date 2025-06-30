# Java Exception 계층 구조 및 상세 설명

## 📊 **Java Exception 계층 구조**

```
Throwable (모든 예외의 최상위 클래스)
├── Error (시스템 레벨 오류)
│   ├── OutOfMemoryError
│   ├── StackOverflowError
│   ├── VirtualMachineError
│   ├── NoClassDefFoundError
│   └── ...
│
└── Exception (애플리케이션 레벨 예외)
    ├── RuntimeException (Unchecked Exception)
    │   ├── NullPointerException
    │   ├── IllegalArgumentException
    │   ├── ArrayIndexOutOfBoundsException
    │   ├── ClassCastException
    │   ├── ArithmeticException
    │   ├── NumberFormatException
    │   └── ...
    │
    └── Checked Exception
        ├── IOException
        │   ├── FileNotFoundException
        │   ├── SocketException
        │   └── ...
        ├── SQLException
        ├── ClassNotFoundException
        ├── InterruptedException
        ├── ParseException
        └── ...
```

## 🎯 **세 가지 주요 카테고리 비교**

### **1. Error (시스템 레벨 오류)**

**특징:**
- JVM이나 시스템 레벨에서 발생하는 심각한 오류
- 애플리케이션에서 복구 불가능
- 프로그램 종료가 일반적
- try-catch 권장하지 않음

**예시:**
```java
// OutOfMemoryError - 메모리 부족
public void memoryLeak() {
    List<byte[]> list = new ArrayList<>();
    while (true) {
        list.add(new byte[1024 * 1024]); // 결국 OutOfMemoryError 발생
    }
}

// StackOverflowError - 스택 오버플로우
public void infiniteRecursion() {
    infiniteRecursion(); // 무한 재귀로 StackOverflowError 발생
}
```

**처리 방식:**
```java
// Error는 일반적으로 catch하지 않음
public void badPractice() {
    try {
        // 일부 코드
    } catch (OutOfMemoryError e) {
        // 이렇게 하면 안됨 - 복구 불가능
    }
}

// 올바른 방식: 예방에 집중
public void goodPractice() {
    // 메모리 사용량 모니터링
    // 적절한 JVM 옵션 설정 (-Xmx, -Xms)
    // 메모리 누수 방지
}
```

### **2. Checked Exception (컴파일 타임 체크)**

**특징:**
- 컴파일러가 강제로 처리하도록 요구
- 예상 가능하고 복구 가능한 상황
- 외부 리소스나 시스템과의 상호작용에서 주로 발생
- 비즈니스 로직의 정상적인 흐름의 일부

**강제되는 이유:**
1. **예상 가능성**: 파일이 없거나, 네트워크가 끊어질 수 있음을 미리 알 수 있음
2. **복구 가능성**: 의미있는 대안 로직을 작성할 수 있음
3. **비즈니스 중요성**: 애플리케이션의 핵심 기능과 관련

**예시:**
```java
// 파일 처리 - 파일이 없을 수 있음을 예상
public String readConfig() throws IOException {
    return Files.readString(Paths.get("config.txt"));
}

// 의미있는 복구 로직
public String getConfiguration() {
    try {
        return readConfig();
    } catch (IOException e) {
        // 복구: 기본 설정 사용
        logger.warn("Config file not found, using defaults", e);
        return "default.property=value";
    }
}

// 네트워크 통신 - 연결 실패 가능성 예상
public void sendData(String data) throws SocketException {
    socket.send(data.getBytes());
}

// 재시도 로직으로 복구
public void reliableSendData(String data) {
    int retries = 3;
    while (retries > 0) {
        try {
            sendData(data);
            return; // 성공
        } catch (SocketException e) {
            retries--;
            if (retries == 0) {
                throw new RuntimeException("Failed after 3 retries", e);
            }
            // 잠시 대기 후 재시도
            Thread.sleep(1000);
        }
    }
}
```

### **3. RuntimeException (Unchecked Exception)**

**특징:**
- 컴파일러가 처리를 강제하지 않음
- 주로 프로그래밍 오류나 로직 실수
- 어디서든 발생할 수 있음
- 복구보다는 코드 수정으로 해결

**try-catch가 강제되지 않는 이유:**

#### **이유 1: 프로그래밍 오류**
```java
// 이런 오류들은 개발자가 코드를 잘못 작성한 것
String str = null;
int length = str.length();  // NullPointerException

int[] arr = new int[5];
int value = arr[10];        // ArrayIndexOutOfBoundsException

// 이걸 try-catch로 감싸는 것보다
try {
    int value = arr[10];
} catch (ArrayIndexOutOfBoundsException e) {
    // 무의미한 복구
}

// 이렇게 예방하는 것이 올바름
if (index >= 0 && index < arr.length) {
    int value = arr[index];
}
```

#### **이유 2: 어디서든 발생 가능**
```java
// 만약 RuntimeException도 강제된다면...
public void simpleMethod(String name, List<Item> items) {
    try {
        try {
            try {
                try {
                    String upper = name.toUpperCase();  // NPE 가능
                    Item first = items.get(0);          // IndexOutOfBounds 가능
                    String result = first.toString();   // NPE 가능
                    Integer.parseInt(result);           // NumberFormat 가능
                } catch (NumberFormatException e) { }
            } catch (NullPointerException e) { }
        } catch (IndexOutOfBoundsException e) { }
    } catch (NullPointerException e) { }
    // 코드가 읽을 수 없게 됨
}

// 실제로는 이렇게 예방
public void simpleMethod(String name, List<Item> items) {
    if (name != null && !items.isEmpty()) {
        String upper = name.toUpperCase();
        Item first = items.get(0);
        if (first != null) {
            String result = first.toString();
            try {
                Integer.parseInt(result);  // 이것만 선택적으로 처리
            } catch (NumberFormatException e) {
                // 실제로 복구 가능한 경우만
            }
        }
    }
}
```

#### **이유 3: 성능과 실용성**
```java
// 모든 메서드 호출에서 예외 체크가 필요하다면
public int calculate(int a, int b) {
    // 이런 간단한 계산도
    return a + b;  // 오버플로우로 ArithmeticException 가능
}

// 모든 호출에서 try-catch 필요
try {
    int result = calculate(x, y);
} catch (ArithmeticException e) { }

// JVM 성능 저하 및 코드 복잡성 급증
```

#### **이유 4: 개발자의 선택권**
```java
// 필요한 경우에만 선택적으로 처리
public void robustMethod() {
    try {
        // 중요한 로직
        processData();
    } catch (RuntimeException e) {
        // 예상치 못한 오류에 대한 전반적 처리
        logger.error("Unexpected error", e);
        notifyAdmin(e);
        // graceful degradation
    }
}
```

## 🔄 **실제 사용 패턴**

### **Error 처리**
```java
// Error는 일반적으로 처리하지 않음
public void systemLevelCode() {
    // 예방에 집중
    // - 적절한 JVM 설정
    // - 메모리 누수 방지
    // - 무한 재귀 방지
}
```

### **Checked Exception 처리**
```java
// 반드시 처리해야 함
public void businessLogic() {
    try {
        String data = readFile("important.txt");
        sendToServer(data);
    } catch (IOException e) {
        // 의미있는 복구 로직
        useBackupData();
    } catch (NetworkException e) {
        // 재시도 로직
        scheduleRetry();
    }
}
```

### **RuntimeException 처리**
```java
// 선택적 처리
public void applicationCode() {
    // 방어적 프로그래밍으로 예방
    if (input != null && input.length() > 0) {
        processInput(input);
    }
    
    // 필요시에만 catch
    try {
        riskyOperation();
    } catch (RuntimeException e) {
        // 로깅 및 graceful handling
        handleUnexpectedError(e);
    }
}
```

## 🎯 **설계 철학 요약**

**Java 예외 시스템의 핵심 원칙:**

1. **Error**: "시스템이 복구 불가능한 상태" → 처리하지 말고 예방하라
2. **Checked Exception**: "예상 가능하고 복구 가능한 상황" → 반드시 처리하라
3. **RuntimeException**: "프로그래밍 오류" → 예방하되, 필요시 선택적 처리

**RuntimeException이 강제되지 않는 핵심 이유:**
- **코드 가독성 유지**: 모든 곳에 try-catch가 있으면 본질적 로직이 묻힘
- **성능 최적화**: 모든 메서드 호출에서 예외 체크는 비효율적
- **실용성**: 프로그래밍 오류는 복구보다 수정이 우선
- **개발자 판단**: 상황에 따라 선택적으로 처리할 수 있는 유연성 제공

이러한 설계를 통해 Java는 **안전성과 실용성의 균형**을 맞추고 있습니다.
