# Rust 스타일 가이드 (Style Guide)

## Rust 명명 규칙

* **모듈**: snake_case 사용 (예: `my_module`)
* **타입**: UpperCamelCase 사용 (예: `MyStruct`)
* **함수**: snake_case 사용 (예: `my_function`)
* **상수**: SCREAMING_SNAKE_CASE 사용 (예: `MY_CONSTANT`)
* **라이프타임**: 짧은 소문자 이름 사용 (예: `'a`, `'b`)

## 코드 구성

* 모듈을 집중적이고 응집력 있게 유지
* 조직을 위해 `mod.rs` 또는 모듈 파일 사용
* `pub`으로 공개 API를 신중하게 내보내기
* `///` 문서 주석으로 공개 API 문서화

## Best Practices

* 기본적으로 불변성 선호
* 패닉 대신 `Result`와 `Option` 사용
* `?` 연산자로 명시적으로 에러 처리
* 적절한 곳에 반복자와 함수형 패턴 사용
* 불필요한 clone 피하기
