- [The Basics (Handbook)](#the-basics-handbook)
- [Everyday Types (Handbook)](#everyday-types-handbook)
- [Narrowing (Handbook)](#narrowing-handbook)
- [More on Functions (Handbook)](#more-on-functions-handbook)
- [Object Types (Handbook)](#object-types-handbook)
- [Type Manipulation (Handbook)](#type-manipulation-handbook)
  - [Creating Types from Types](#creating-types-from-types)
  - [Generics](#generics)
  - [Keyof Type Operator](#keyof-type-operator)
  - [Typeof Type Operator](#typeof-type-operator)
  - [Indexed Access Types](#indexed-access-types)
  - [Conditional Types](#conditional-types)
  - [Mapped Types](#mapped-types)
  - [Template Literal Types](#template-literal-types)
- [Classes (Handbook)](#classes-handbook)
- [Modules (Handbook)](#modules-handbook)

----

## The Basics (Handbook)

```typescript
//////////////////////////////////////////////////////////////////////
// Boolean
{
    let isDone: Boolean = false;
}

//////////////////////////////////////////////////////////////////////
// Number
{
    let decimal: number = 6;
    let hex: number = 0xf00d;       // 0x
    let binary: number = 0b1010;    // 0b
    let octal: Number = 0o744;      // 0o
}

//////////////////////////////////////////////////////////////////////
// String
{
    let color: String = "blue";
    color = "red";

    let fullName: String = `David Sun`;
    let age: number = 18;
    let sentence: String = `Hello, my name is ${ fullName }.
    I'll be ${ age + 1 } years old next month.`;
}

//////////////////////////////////////////////////////////////////////
// Array
{
    let list: number[] = [1, 2, 3];
    let listAgain: Array<number> = [1, 2, 3];
}

//////////////////////////////////////////////////////////////////////
// Tuple
{
    let x: [string, number];
    x = ["Hello", 10];
    // x = [10, "Hello"]; // ERROR
    
    console.log(x[0].substring(1));
    // console.log(x[1].substring(1)); // ERROR

    // x[3] = "world" // ERROR
    // console.log(x[5].toString()); // ERROR
}

//////////////////////////////////////////////////////////////////////
// Enum
{
    {
        enum Color {
            Red,
            Green,
            Blue,
        }
        let c: Color = Color.Green;
    }
    {
        enum Color {
            Red = 1,
            Green,
            Blue,
        }
        let c: Color = Color.Green;
    }
    {
        enum Color {Red = 1, Green, Blue}
        let colorName: string = Color[2];
        console.log(colorName); // Green
    }
}

//////////////////////////////////////////////////////////////////////
// Any
{
    {
        let notSure: any = 4;
        notSure = "Maybe a string instead";
        notSure = false;
    }
    {
        let notSure: any = 4;
        notSure.ifItExists;
        notSure.toFixed();

        let prettySure: Object = 4;
        // prettySure.toFixed();  // ERROR

        let list: any[] = [1, true, "free"];
        list[1] = 100;
        console.log(list);
    }
}

//////////////////////////////////////////////////////////////////////
// Void
{
    function warnUser(): void {
        console.log("This is my warnUser.");
    }
    
    let unusable: void = undefined;
    // unusable = null;  // ERROR, just OK without '--strictNullChecks'
}

//////////////////////////////////////////////////////////////////////
// Null and Undefined
{
    let u: undefined = undefined;
    let n: null = null;
}

//////////////////////////////////////////////////////////////////////
// Never
{
    // Cannot reach end of function with never return.
    function error(message: string): never {
        throw new Error(message);
    }
    // Infer never return
    function fail() {
        return error("Something failed!!!");
    }
    function infiniteLoop(): never {
        while (true) {            
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Object
declare function create(o: object | null): void;
{
    create({ prop: 0 });
    create(null);
    // create(42);         // ERROR
    // create("string");   // ERROR
    // create(false);      // ERROR
    // create(undefined);  // ERROR    
}

//////////////////////////////////////////////////////////////////////
// Type assertions (Type casting)
// Trust me I know what I am doing.
{
    let someValue: any = "This is a string.";
    // 0. angle-bracket
    let strLen: number = (<string>someValue).length;
    // 1. as
    let strLen2: number = (someValue as string).length;
}

//////////////////////////////////////////////////////////////////////
// let
// Prefer let than var
{
    let a: any = 3;
    console.log(a);
}
```

## Everyday Types (Handbook)

## Narrowing (Handbook)

## More on Functions (Handbook)

## Object Types (Handbook)

```ts
//////////////////////////////////////////////////////////////////////
// Object Types
// object type
{
    function greet(person: { name: string; age: number }) {
        return "Hello " + person.name;
    }
}
// interface
{
    interface Person {
        name: string;
        age: number;
    }
    function greet(person: Person) {
        return "Hello " + person.name;
    }
}
// type alias
{
    type Person = {
        name: String;
        age: number;
    }
    function greet(person: Person) {
        return "Helo " + person.name;
    }
}

//////////////////////////////////////////////////////////////////////
// Property Modifiers
//////////////////////////////////////////////////////////////////////
// ..Optional Properties
// {
//     interface PaintOptions {
//         shape: Shape;
//         xPos?: number;
//         yPos?: number;
//     }
//     function paintShape(opts: PaintOptions) {
//     }
//     const shape = getShape();
//     paintShape({ shape });
//     paintShape({ shape, xPos: 100 });
//     paintShape({ shape, yPos: 100 });
//     paintShape({ shape, xPos: 100, yPos: 100 });
// }
// {
//     function paintShape(opts: PaintOptions) {
//         let xPos = opts.xPos === undefined ? 0 : opts.xPos;
//         let yPos = opts.yPos === undefined ? 0 : opts.yPos;
//     }
// }

//////////////////////////////////////////////////////////////////////
// ..readonly Properties
{
    interface SomeType {
        readonly prop: string;
    }
    function doSomething(obj: SomeType) {
        console.log(`prop has the value '${obj.prop}'.`);
        // obj.prop = "hello"; // ERROR
    }
}
{
    interface Home {
        readonly resident: { name: string; age: number };
    }
    function visitForBirthday(home: Home) {
        console.log(`Happy birthday ${home.resident.name}!`);
        home.resident.age++;
    }
    function evict(home: Home) {
        // home.resident = {  // ERROR
        //     name: "Victor the Evictor",
        //     age: 42,
        // }
    };
}
{
    interface Person {
        name: string;
        age: number;
    }
    interface ReadonlyPerson {
        readonly name: string;
        readonly age: number;
    }
    let writablePerson: Person = {
        name: "Person David Sun",
        age: 42,
    }
    let readonlyPerson: ReadonlyPerson = writablePerson;
    console.log(readonlyPerson.age);  // 42
    writablePerson.age++;
    console.log(readonlyPerson.age);  // 43
}
//////////////////////////////////////////////////////////////////////
// ..Index Signatures
// declare function getStringArray();
{
    // An index signature property type must be either ‘string’ or ‘number’.
    interface StringArray {
        [index: number]: string;
    }
    // const myArray: StringArray = getStringArray();
    // const secondItem = myArray[1];
}
{
    interface Animal {
        name: string;
    }
    interface Dog extends Animal {
        breed: string;
    }
    interface NotOkay {
        // ERROR: 'number' index type 'Animal' is not assignable to 'string'
        // index type 'Dog'.
        // [x: number]: Animal;
        [x: string]: Dog;
    }
}
{
    interface NumberDictionary {
        [index: string]: number;
        length: number;
        // ERROR: Property 'name' of type 'string' is not assignable to 'string'
        // index type 'number'.
        // name: string;
    }
}
{
    // union of the property types:
    interface NumberOrStringDictionary {
        [index: string]: number | string;
        length: number;
        name: string;
    }
}
// declare function geReadOnlyStringArray(): any;
{
    interface ReadonlyStringArray {
        readonly [index: number]: string;
    }
    // let myArray: ReadonlyStringArray = geReadOnlyStringArray();
    // myArray[2] = "David";  // ERROR
}

//////////////////////////////////////////////////////////////////////
// Extending Types
{
    interface BasicAddress {
        name?: string;
        street: string;
        city: string;
        country: string;
        postalCode: string;
    }
    interface AddressWithUnit {
        name?: string;
        unit: string;
        street: string;
        city: string;
        country: string;
        postalCode: string;
    }   
}
{
    interface BasicAddress {
        name?: string;
        street: string;
        city: string;
        country: string;
        postalCode: string;
    }
    interface AddressWithUnit extends BasicAddress {
        unit: string;
    }
}
{
    interface Colorful {
        color: string;
    }
    interface Circle {
        radius: number;
    }
    interface ColorfulCircle extends Colorful, Circle {}
    const cc: ColorfulCircle = {
        color: "red",
        radius: 42,
    }
}
//////////////////////////////////////////////////////////////////////
// Intersection Types
{
    interface Colorful {
        color: string;
    }
    interface Circle {
        radius: number;
    }
    type ColorfulCircle = Colorful & Circle;
    function draw(circle: Colorful & Circle) {
        console.log(`Color was ${circle.color}`);
        console.log(`Radius was ${circle.radius}`);
    }
    draw({ color: "blue", radius: 42 });
    // ERROR: Argument of type '{ color: string; raidus: number; }' is not
    // assignable to parameter of type 'Colorful & Circle'. Object literal may
    // only specify known properties, but 'raidus' does not exist in type
    // 'Colorful & Circle'. Did you mean to write 'radius'?
    // draw({ color: "red", raidus: 42 });

}
//////////////////////////////////////////////////////////////////////
// Interfaces vs. Intersections
// extends
{
    interface Point {
        x: number;
        y: number;
    }
    interface PointColor extends Point {
        c: number;
    }
    const pointColor = {
        x: 3,
        y: 3,
        c: 3,
    }
    console.log(pointColor);
}
{
    type Point = {
        x: number;
        y: number;
    }
    interface PointColor extends Point {
        c: number;
    }
    const pointColor: PointColor = { x: 3, y: 3, c: 3 };
    console.log(pointColor);
}
{
    // extends does not work for type
    type Point = {
        x: number;
        y: number;
    }
    // // ERROR: Could not use type with extends
    // type PointColor extends Point {
    //     c: number;
    // }
}
// merged declaration
{
    // merged declaration works for interface
    interface PointColor {
        x: number;
        y: number;
    }
    interface PointColor {
        c: number;
    }
    const pointColor: PointColor = { x: 3, y: 3, c: 3 };
    console.log(pointColor);
}
{
    // // ERROR: mergedd declaration does not work for type
    // type PointColor = {
    //     x: number;
    //     y: number;
    // }
    // type PointColor = {
    //     c: number;
    // }
}
// computed value
{
    // computed value does not work for interface
    type coords = 'x' | 'y';
    interface CoordTypes {
        // // ERROR
        // [key in coords]: string
    }
}
{
    // computed value works for type
    type coords = 'x' | 'y';
    type CoordTypes = {
        [CoordTypes in coords]: string;
    }
    const point: CoordTypes = { x: '3', y: '3' };
    console.log(point);
}
// type could be resolved to never type
// You should be careful
{
    type goodType = { a: 1 } & { b: 2 } // good
    type neverType = { a: 1; b: 2 } & { b: 3 } // resolved to `never`

    const foo: goodType = { a: 1, b: 2 } // good
    // // ERROR: Type 'number' is not assignable to type 'never'.(2322)
    // const bar: neverType = { a: 1, b: 3 } 
    // // ERROR: Type 'number' is not assignable to type 'never'.(2322)
    // const baz: neverType = { a: 1, b: 2 } 
}
{
    type t1 = {
        a: number
    }
    type t2 = t1 & {
        b: string
    }
    // // ERROR
    // const foo: t2 = { a: 1, b: 2 }
}
//////////////////////////////////////////////////////////////////////
// Generic Object Types
{
    interface Box {
        contents: any;
    }
}
{
    interface Box {
        contents: unknown;
    }
    let x: Box = {
        contents: "Hello World",
    }
    if (typeof x.contents === "string") {
        console.log(x.contents.toLowerCase());
    }
    console.log((x.contents as string).toLowerCase());
}
{
    interface NumberBox {
        contents: number;
    }

    interface StringBox {
        contents: string;
    }

    interface BooleanBox {
        contents: boolean;
    }
    // boiler plates
    function setContents(box: StringBox, newContents: string): void;
    function setContents(box: NumberBox, newContents: number): void;
    function setContents(box: BooleanBox, newContents: boolean): void;
    function setContents(box: { contents: any }, newContents: any) {
            box.contents = newContents;
    }
}
{
    interface Box<Type> {
        contents: Type;
    }
    let box: Box<string> = { contents: "Hello Box" };
    function setContents<Type>(box: Box<Type>, newContents: Type) {
        box.contents = newContents;
    }
    setContents(box, "Bye World");
    console.log(box);
    console.log(typeof(box));
}
{
    type Box<Type> = {
        contents: Type;
    }
    let box: Box<string> = { contents: "Hello World" };
    console.log(box);
    console.log(typeof(box));
}
{
    // type is useful for generic helper types.
    type OrNull<Type> = Type | null;
    type OneOrMany<Type> = Type | Type[];
    type OneOrManyOrNull<Type> = OrNull<OneOrMany<Type>>;
    type OneOrManuOrNullStrings = OneOrManyOrNull<string>;
}
//////////////////////////////////////////////////////////////////////
// ..The Array Type
{
    function doSomething(value: Array<string>) {        
    }
    let myArray: string[] = ["hello", "world"];
    doSomething(myArray);
    doSomething(new Array("hello", "world"));
    interface Array<Type> {
        length: number;
        pop(): Type | undefined;
        push(...items: Type[]): number;
    }
}

//////////////////////////////////////////////////////////////////////
// ..The ReadonlyArray Type
{
    function doStuff(values: ReadonlyArray<string>) {
        const copy = values.slice();
        console.log(`The first value is ${values[0]}`);
        
        // // ERROR:
        // // Property 'push' does not exist on type 'readonly string[]'.
        // values.push("hello!");
        
        // // ERROR:
        // // Property 'push' does not exist on type 'readonly string[]'.
        // new ReadonlyArray("red", "green", "blue");
    }
}
{
    let x: readonly string[] = [];
    let y: string[] = [];

    x = y;
    // // ERROR:
    // // The type 'readonly string[]' is 'readonly' and cannot be assigned to the
    // // mutable type 'string[]'.
    // y = x;
}

//////////////////////////////////////////////////////////////////////
// ..Tuple Types
{
    type StringNumberPair = [string, number];
    function doSomething(pair: [string, number]) {
        const a = pair[0];
        const b = pair[1];
    }
    doSomething(["hello", 43]);
}
{
    function doSomething(pair: [string, number]) {
        // // ERROR:
        // // Tuple type '[string, number]' of length '2' has no element at index
        // // '2'.
        // const c = pair[2];
    }
}
{
    function doSomething(stringHash: [string, number]) {
        const [inputString, hash] = stringHash;
        console.log(inputString);
        console.log(hash);
    }
}
{
    interface StringNumberPair {
        length: 2;
        0: string;
        1: number;
        slide(start?: number, end?: number): Array<string | number>;
    }
}
{
    type Either2dOr3d = [number, number, number?];
    function setCoord(coord: Either2dOr3d) {
        const [x, y, z] = coord;
        console.log(`Provided coordinates had ${coord.length} dimensions`);
    }
}
{
    type StringNumberBoolenas = [string, number, ...boolean[]];
    type StringBooleansNumber = [string, ...boolean[], number];
    type BooleansStringNumber = [...boolean[], string, number];
    const a: StringNumberBoolenas = ["hello", 1];
    const b: StringNumberBoolenas = ["world", 2, true];
    const c: StringNumberBoolenas = ["beutiful", 3, true, false];
}
{
    function readButtonInput(...args: [string, number, ...boolean[]]) {
        const [name, version, ...input] = args;
    }
    function readButtonInputSame(name: string, version: number, ...input: boolean[]) {
    }
}

//////////////////////////////////////////////////////////////////////
// ..readonly Tuple Types
{
    function foo(pair: readonly [string, number]) {        
    }
    function bar(pair: readonly [string, number]) {
        // // ERROR:
        // // Cannot assign to '0' because it is a read-only property.
        // pair[0] = "hello!";
    }
}
{
    let point = [3, 4] as const;
    function distanceFromOrigin([x, y]: [number, number]) {
        return Math.sqrt(x ** 2 + y ** 2);
    }
    // // ERROR: 
    // // Argument of type 'readonly [3, 4]' is not assignable to parameter of type
    // // '[number, number]'.
    // distanceFromOrigin(point);
}
```

## Type Manipulation (Handbook)

### Creating Types from Types

### Generics

### Keyof Type Operator

### Typeof Type Operator

### Indexed Access Types

### Conditional Types

### Mapped Types

### Template Literal Types

## Classes (Handbook)

## Modules (Handbook)
