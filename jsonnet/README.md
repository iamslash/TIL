# Abstract

jsonnet 은 json, yaml file 을 생성하는 template engine 이다. k8s 에서 주로 사용한다.

# Materials

* [Jsonnet @ youtube](https://www.youtube.com/watch?v=i5PVp92tAmE)

# Install

## Install on maxOS

```bash
$ brew install jsonnet
$ jsonnet a.jsonnet
```

# Basics

## Simple one

```bash
$ jsonnet a.jsonnet
{
   "person1": {
      "name": "Alice",
      "welcome": "Hello Alice!"
   },
   "person2": {
      "name": "Bob",
      "welcome": "Hello Bob!"
   }
}
$ jsonnet -e '{ x: 1, y: self.x + 1 } {x: 10}'
{
    "x": 10,
    "y": 11
}
```

## Multiple File Output

* f.jsonnet

```jsonnet
{
  "a.json": {
    x: 1,
    y: $["b.json"].y,
  },
  "b.json": {
    x: $["a.json"].x,
    y: 2,
  },
}
```

* run

```bash
$ jsonnet -m . f.jsonnet
a.json
b.json
$ cat a.json 
{
   "x": 1,
   "y": 2
}
$ cat b.json 
{
   "x": 1,
   "y": 2
}
```

## YAML stream output

YAML can represent several objects in the same file, separated by `---`.

* f.jsonnet

```
// yaml_stream.jsonnet
local
  a = {
    x: 1,
    y: b.y,
  },
  b = {
    x: a.x,
    y: 2,
  };

[a, b]
```

* run

```bash
$ jsonnet -y f.jsonnet
---
{
   "x": 1,
   "y": 2
}
---
{
   "x": 1,
   "y": 2
}
```

## Syntax

* C-style, Python-style comments
* `|||` means text blocks
* commas at the end of arrays or objects
* `"` and `'` is same.
  * `"Farmer's Gin`, `'Farmer\'s Fin'`
* `@` means verbatim strings.
 
```jsonnet
/* A C-style comment. */
# A Python-style comment.
{
  cocktails: {
    // Ingredient quantities are in fl oz.
    'Tom Collins': {
      ingredients: [
        { kind: "Farmer's Gin", qty: 1.5 },
        { kind: 'Lemon', qty: 1 },
        { kind: 'Simple Syrup', qty: 0.5 },
        { kind: 'Soda', qty: 2 },
        { kind: 'Angostura', qty: 'dash' },
      ],
      garnish: 'Maraschino Cherry',
      served: 'Tall',
      description: |||
        The Tom Collins is essentially gin and
        lemonade.  The bitters add complexity.
      |||,
    },
    Manhattan: {
      ingredients: [
        { kind: 'Rye', qty: 2.5 },
        { kind: 'Sweet Red Vermouth', qty: 1 },
        { kind: 'Angostura', qty: 'dash' },
      ],
      garnish: 'Maraschino Cherry',
      served: 'Straight Up',
      description: @'A clear \ red drink.',
    },
  },
}
```

* run

```bash
{
   "cocktails": {
      "Manhattan": {
         "description": "A clear \\ red drink.",
         "garnish": "Maraschino Cherry",
         "ingredients": [
            {
               "kind": "Rye",
               "qty": 2.5
            },
            {
               "kind": "Sweet Red Vermouth",
               "qty": 1
            },
            {
               "kind": "Angostura",
               "qty": "dash"
            }
         ],
         "served": "Straight Up"
      },
      "Tom Collins": {
         "description": "The Tom Collins is essentially gin and\nlemonade.  The bitters add complexity.\n",
         "garnish": "Maraschino Cherry",
         "ingredients": [
            {
               "kind": "Farmer's Gin",
               "qty": 1.5
            },
            {
               "kind": "Lemon",
               "qty": 1
            },
            {
               "kind": "Simple Syrup",
               "qty": 0.5
            },
            {
               "kind": "Soda",
               "qty": 2
            },
            {
               "kind": "Angostura",
               "qty": "dash"
            }
         ],
         "served": "Tall"
      }
   }
}
```

## Variables

* f.jsonnet

```js
// A regular definition.
local house_rum = 'Banks Rum';

{
  // A definition next to fields.
  local pour = 1.5,

  Daiquiri: {
    ingredients: [
      { kind: house_rum, qty: pour },
      { kind: 'Lime', qty: 1 },
      { kind: 'Simple Syrup', qty: 0.5 },
    ],
    served: 'Straight Up',
  },
  Mojito: {
    ingredients: [
      {
        kind: 'Mint',
        action: 'muddle',
        qty: 6,
        unit: 'leaves',
      },
      { kind: house_rum, qty: pour },
      { kind: 'Lime', qty: 0.5 },
      { kind: 'Simple Syrup', qty: 0.5 },
      { kind: 'Soda', qty: 3 },
    ],
    garnish: 'Lime wedge',
    served: 'Over crushed ice',
  },
}
```

* output.json

```json
{
  "Daiquiri": {
    "ingredients": [
      {
        "kind": "Banks Rum",
        "qty": 1.5
      },
      {
        "kind": "Lime",
        "qty": 1
      },
      {
        "kind": "Simple Syrup",
        "qty": 0.5
      }
    ],
    "served": "Straight Up"
  },
  "Mojito": {
    "garnish": "Lime wedge",
    "ingredients": [
      {
        "action": "muddle",
        "kind": "Mint",
        "qty": 6,
        "unit": "leaves"
      },
      {
        "kind": "Banks Rum",
        "qty": 1.5
      },
      {
        "kind": "Lime",
        "qty": 0.5
      },
      {
        "kind": "Simple Syrup",
        "qty": 0.5
      },
      {
        "kind": "Soda",
        "qty": 3
      }
    ],
    "served": "Over crushed ice"
  }
}
```

## References

* f.jsonnet

```js
{
  concat_array: [1, 2, 3] + [4],
  concat_string: '123' + 4,
  equality1: 1 == '1',
  equality2: [{}, { x: 3 - 1 }]
             == [{}, { x: 2 }],
  ex1: 1 + 2 * 3 / (4 + 5),
  // Bitwise operations first cast to int.
  ex2: self.ex1 | 3,
  // Modulo operator.
  ex3: self.ex1 % 2,
  // Boolean logic
  ex4: (4 > 3) && (1 <= 3) || false,
  // Mixing objects together
  obj: { a: 1, b: 2 } + { b: 3, c: 4 },
  // Test if a field is in an object
  obj_member: 'foo' in { foo: 1 },
  // String formatting
  str1: 'The value of self.ex2 is '
        + self.ex2 + '.',
  str2: 'The value of self.ex2 is %g.'
        % self.ex2,
  str3: 'ex1=%0.2f, ex2=%0.2f'
        % [self.ex1, self.ex2],
  // By passing self, we allow ex1 and ex2 to
  // be extracted internally.
  str4: 'ex1=%(ex1)0.2f, ex2=%(ex2)0.2f'
        % self,
  // Do textual templating of entire files:
  str5: |||
    ex1=%(ex1)0.2f
    ex2=%(ex2)0.2f
  ||| % self,
}
```

* output.json

```json
{
  "concat_array": [
    1,
    2,
    3,
    4
  ],
  "concat_string": "1234",
  "equality1": false,
  "equality2": true,
  "ex1": 1.6666666666666665,
  "ex2": 3,
  "ex3": 1.6666666666666665,
  "ex4": true,
  "obj": {
    "a": 1,
    "b": 3,
    "c": 4
  },
  "obj_member": true,
  "str1": "The value of self.ex2 is 3.",
  "str2": "The value of self.ex2 is 3.",
  "str3": "ex1=1.67, ex2=3.00",
  "str4": "ex1=1.67, ex2=3.00",
  "str5": "ex1=1.67\nex2=3.00\n"
}
```

## Arithmetic

* f.jsonnet

```
// Define a local function.
// Default arguments are like Python:
local my_function(x, y=10) = x + y;

local object = {
  // A method
  my_method(x): x * x,
};

{
  // Functions are first class citizens.
  call_inline_function:
    (function(x) x * x)(5),

  // Using the variable fetches the function,
  // the parens call the function.
  call: my_function(2),

  // Like python, parameters can be named at
  // call time.
  named_params: my_function(x=2),
  // This allows changing their order
  named_params2: my_function(y=3, x=2),

  // object.my_method returns the function,
  // which is then called like any other.
  call_method1: object.my_method(3),

  standard_lib:
    std.join(' ', std.split('foo/bar', '/')),
  len: [
    std.length('hello'),
    std.length([1, 2, 3]),
  ],
}
```

* output.json

```json
{
  "call": 12,
  "call_inline_function": 25,
  "call_method1": 9,
  "len": [
    5,
    3
  ],
  "named_params": 12,
  "named_params2": 5,
  "standard_lib": "foo bar"
}
```

## Functions

* f.jsonnet

```
// This function returns an object. Although
// the braces look like Java or C++ they do
// not mean a statement block, they are instead
// the value being returned.
local Sour(spirit, garnish='Lemon twist') = {
  ingredients: [
    { kind: spirit, qty: 2 },
    { kind: 'Egg white', qty: 1 },
    { kind: 'Lemon Juice', qty: 1 },
    { kind: 'Simple Syrup', qty: 1 },
  ],
  garnish: garnish,
  served: 'Straight Up',
};

{
  'Whiskey Sour': Sour('Bulleit Bourbon',
                       'Orange bitters'),
  'Pisco Sour': Sour('Machu Pisco',
                     'Angostura bitters'),
}
```

* output.json

```json
{
  "Pisco Sour": {
    "garnish": "Angostura bitters",
    "ingredients": [
      {
        "kind": "Machu Pisco",
        "qty": 2
      },
      {
        "kind": "Egg white",
        "qty": 1
      },
      {
        "kind": "Lemon Juice",
        "qty": 1
      },
      {
        "kind": "Simple Syrup",
        "qty": 1
      }
    ],
    "served": "Straight Up"
  },
  "Whiskey Sour": {
    "garnish": "Orange bitters",
    "ingredients": [
      {
        "kind": "Bulleit Bourbon",
        "qty": 2
      },
      {
        "kind": "Egg white",
        "qty": 1
      },
      {
        "kind": "Lemon Juice",
        "qty": 1
      },
      {
        "kind": "Simple Syrup",
        "qty": 1
      }
    ],
    "served": "Straight Up"
  }
}
```


## Conditionals

* f.jsonnet

```
local Mojito(virgin=false, large=false) = {
  // A local next to fields ends with ','.
  local factor = if large then 2 else 1,
  // The ingredients are split into 3 arrays,
  // the middle one is either length 1 or 0.
  ingredients: [
    {
      kind: 'Mint',
      action: 'muddle',
      qty: 6 * factor,
      unit: 'leaves',
    },
  ] + (
    if virgin then [] else [
      { kind: 'Banks', qty: 1.5 * factor },
    ]
  ) + [
    { kind: 'Lime', qty: 0.5 * factor },
    { kind: 'Simple Syrup', qty: 0.5 * factor },
    { kind: 'Soda', qty: 3 * factor },
  ],
  // Returns null if not large.
  garnish: if large then 'Lime wedge',
  served: 'Over crushed ice',
};

{
  Mojito: Mojito(),
  'Virgin Mojito': Mojito(virgin=true),
  'Large Mojito': Mojito(large=true),
}
```

* run

```json
{
  "Large Mojito": {
    "garnish": "Lime wedge",
    "ingredients": [
      {
        "action": "muddle",
        "kind": "Mint",
        "qty": 12,
        "unit": "leaves"
      },
      {
        "kind": "Banks",
        "qty": 3
      },
      {
        "kind": "Lime",
        "qty": 1
      },
      {
        "kind": "Simple Syrup",
        "qty": 1
      },
      {
        "kind": "Soda",
        "qty": 6
      }
    ],
    "served": "Over crushed ice"
  },
  "Mojito": {
    "garnish": null,
    "ingredients": [
      {
        "action": "muddle",
        "kind": "Mint",
        "qty": 6,
        "unit": "leaves"
      },
      {
        "kind": "Banks",
        "qty": 1.5
      },
      {
        "kind": "Lime",
        "qty": 0.5
      },
      {
        "kind": "Simple Syrup",
        "qty": 0.5
      },
      {
        "kind": "Soda",
        "qty": 3
      }
    ],
    "served": "Over crushed ice"
  },
  "Virgin Mojito": {
    "garnish": null,
    "ingredients": [
      {
        "action": "muddle",
        "kind": "Mint",
        "qty": 6,
        "unit": "leaves"
      },
      {
         "kind": "Lime",
         "qty": 0.5
      },
      {
         "kind": "Simple Syrup",
         "qty": 0.5
      },
      {
         "kind": "Soda",
         "qty": 3
      }
    ],
    "served": "Over crushed ice"
  }
}
```

## Computed Field Names


## Array and Object Comprehension


## Imports


## Errors


## Parameterize Entire Config


## Object-Orientation

