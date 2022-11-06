- [Abstract](#abstract)
- [Materials](#materials)
- [Overview](#overview)
- [Scanner](#scanner)
- [Parser](#parser)
- [Generator](#generator)
- [Interpreter](#interpreter)
- [Machine](#machine)

----

# Abstract

Compiler 에 대해 정리한다.

# Materials

* [컴파일러 만들기 컴퓨터 프로그램의 구조와 원리 | yes24](http://www.yes24.com/Product/Goods/103153057)
  * [src](https://github.com/AcornPublishing/crafting-compiler) 
* [Crafting Interpreters](https://craftinginterpreters.com/)
  * 짱짱맨
* [컴파일러 구조와 원리 | yes24](http://www.yes24.com/Product/Goods/4189980)
  * linker 의 분량이 적다는게 흠이다.

# Overview

Compiler 는 다음과 같은 과정을 수행하며 Object Code 를 생성한다.

```
소스코드 -> 어휘분석 -> 구문분석 -> 코드생성 -> 목적코드
```

Interpreter 는 다음과 같은 과정을 수행하며 실행결과를 생성한다. Compiler
와 어휘분석, 구문분석은 동일하다.

```
소스코드 -> 어휘분석 -> 구문분석 -> 코드실행 -> 실행결과
```

Compiler 에 의해 생성된 Object Code 는 다음과 같이 Machine 에 의해
실행된다.

```
목적코드 -> Machine -> 실행결과
```

# Scanner

아래와 같은 code 를 `vector<Token> tokens` 에 저장한다.

```
    function main() {
      print 'Hello, World!';
    }
```

아래는 `Token` 의 정의이다.

```cpp
struct Token {
  Kind kind = Kind::Unknown;
  string string;
};

enum class Kind {
  Unknown, EndOfToken,

  NullLiteral,
  TrueLiteral, FalseLiteral,
  NumberLiteral, StringLiteral,
  Identifier,

  Function, Return,
  Variable,
  For, Break, Continue,
  If, Elif, Else,
  Print, PrintLine,

  LogicalAnd, LogicalOr,
  Assignment,
  Add, Subtract,
  Multiply, Divide, Modulo,
  Equal, NotEqual,
  LessThan, GreaterThan,
  LessOrEqual, GreaterOrEqual,

  Comma, Colon, Semicolon,
  LeftParen, RightParen,
  LeftBrace, RightBrace,
  LeftBraket, RightBraket,
};
```

# Parser

Scanner 에 의해 생성한 `vector<Token> tokens` 를 읽고 `Program* program` 을
리턴한다. 즉, Abstract Syntax Tree 를 생성한다.

```cpp
struct Program {
  vector<struct Function*> functions;
};

struct Statement {
  virtual auto print(int)->void = 0;
};

struct Expression {
  virtual auto print(int)->void = 0;
};

struct Function: Statement {
  string name;
  vector<string> parameters;
  vector<Statement*> block;
  auto print(int)->void;
};

struct Variable: Statement {
  string name;
  Expression* expression;
  auto print(int)->void;
};

struct Return: Statement {
  Expression* expression;
  auto print(int)->void;
};

struct For: Statement {
  Variable* variable;
  Expression* condition;
  Expression* expression;
  vector<Statement*> block;
  auto print(int)->void;
};

struct Break: Statement {
  auto print(int)->void;
};

struct Continue: Statement {
  auto print(int)->void;
};

struct If: Statement {
  vector<Expression*> conditions;
  vector<vector<Statement*>> blocks;
  vector<Statement*> elseBlock;
  auto print(int)->void;
};

struct Print: Statement {
  bool lineFeed = false;
  vector<Expression*> arguments;
  auto print(int)->void;
};

struct ExpressionStatement: Statement {
  Expression* expression;
  auto print(int)->void;
};

struct Or: Expression {
  Expression* lhs;
  Expression* rhs;
  auto print(int)->void;
};

struct And: Expression {
  Expression* lhs;
  Expression* rhs;
  auto print(int)->void;
};

struct Relational: Expression {
  Kind kind;
  Expression* lhs;
  Expression* rhs;
  auto print(int)->void;
};

struct Arithmetic: Expression {
  Kind kind;
  Expression* lhs;
  Expression* rhs;
  auto print(int)->void;
};

struct Unary: Expression {
  Kind kind;
  Expression* sub;
  auto print(int)->void;
};

struct Call: Expression {
  Expression* sub;
  vector<Expression*> arguments;
  auto print(int)->void;
};

struct GetElement: Expression {
  Expression* sub;
  Expression* index;
  auto print(int)->void;
};

struct SetElement: Expression {
  Expression* sub;
  Expression* index;
  Expression* value;
  auto print(int)->void;
};

struct GetVariable: Expression {
  string name;
  auto print(int)->void;
};

struct SetVariable: Expression {
  string name;
  Expression* value;
  auto print(int)->void;
};

struct NullLiteral: Expression {
  auto print(int)->void;
};

struct BooleanLiteral: Expression {
  bool value = false;
  auto print(int)->void;
};

struct NumberLiteral: Expression {
  double value = 0.0;
  auto print(int)->void;
};

struct StringLiteral: Expression {
  string value;
  auto print(int)->void;
};

struct ArrayLiteral: Expression {
  vector<Expression*> values;
  auto print(int)->void;
};

struct MapLiteral: Expression {
  map<string, Expression*> values;
  auto print(int)->void;
};
```

# Generator

Parser 에 의해 생성된 Abstract Syntax Tree 인 `Program* program` 를 읽고
`tuple<vector<Code>, map<string, size_t>> objCodes` 를 리턴한다. 즉, Object Code
와 Function Table 을 리턴한다.

```cpp
struct Code {
  Instruction instruction;
  any operand;
};

enum class Instruction {
  Exit,
  Call, Alloca, Return,
  Jump, ConditionJump,
  Print, PrintLine,

  LogicalOr, LogicalAnd,
  Add, Subtract,
  Multiply, Divide, Modulo,
  Equal, NotEqual,
  LessThan, GreaterThan,
  LessOrEqual, GreaterOrEqual,
  Absolute, ReverseSign,

  GetElement, SetElement,
  GetGlobal, SetGlobal,
  GetLocal, SetLocal,

  PushNull, PushBoolean,
  PushNumber, PushString,
  PushArray, PushMap,
  PopOperand,
};
```

Abstract Syntax Tree 를 구성하는 Node 들의 종류는 다음과 같다. `generate()` 는 Object Code
를 생성하는 함수이다.

```cpp
struct Program {
  vector<struct Function*> functions;
};

struct Statement {
  virtual auto generate()->void = 0;
};

struct Expression {
  virtual auto generate()->void = 0;
};
```

# Interpreter

Abstract Syntax Tree 를 구성하는 Node 들의 종류는 다음과 같다. `interpret()` 는 Node 를 실행하는 함수이다.

```cpp
struct Program {
  vector<struct Function*> functions;
};

struct Statement {
  virtual auto interpret()->void = 0;
};

struct Expression {
  virtual auto interpret()->any = 0;
};
```

# Machine

Generator 가 생성한 Object Codes `tuple<vector<Code>, map<string, size_t>> objCodes` 를 실행합니다.

이때 Stack 을 이용한 Virtual Machine 을 Stack Machine 이라고 합니다. 다음과 같이
`struct StackFrame` 을 정의하여 Stack 으로 이용합니다.

```cpp
struct StackFrame {
  vector<any> variables;
  vector<any> operandStack;
  size_t instructionPointer = 0;
};

auto execute(tuple<vector<Code>, map<string, size_t>> objectCode)->void {
...
  while (true) {
    auto code = codeList[callStack.back().instructionPointer];
    switch (code.instruction) {
    case Instruction::Exit: ...
    case Instruction::Call: ...
    case Instruction::Alloca: ...
    case Instruction::Return: ...
    case Instruction::Jump: ...
    case Instruction::ConditionJump: ...
    case Instruction::Print: ...
    case Instruction::PrintLine: ...
    case Instruction::LogicalOr: ...
    case Instruction::LogicalAnd: ...
    case Instruction::Equal: ...
    case Instruction::NotEqual: ...
    case Instruction::LessThan: ...
    case Instruction::GreaterThan: ...
    case Instruction::LessOrEqual: ...
    case Instruction::GreaterOrEqual: ...
    case Instruction::Add: ...
    case Instruction::Subtract: ...
    case Instruction::Multiply: ...
    case Instruction::Divide: ...
    case Instruction::Modulo: ...
    case Instruction::Absolute: ...
    case Instruction::ReverseSign: ...
    case Instruction::GetElement: ...
    case Instruction::SetElement: ...
    case Instruction::GetGlobal: ...
    case Instruction::SetGlobal: ...
    case Instruction::GetLocal: ...
    case Instruction::SetLocal: ...
    case Instruction::PushNull: ...
    case Instruction::PushBoolean: ...
    case Instruction::PushNumber: ...
    case Instruction::PushString: ...
    case Instruction::PushArray: ...
    case Instruction::PushMap: ...
    case Instruction::PopOperand: ...
  }
}
```
