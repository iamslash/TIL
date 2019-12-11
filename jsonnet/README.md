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

* Fields do not need quotes.
* Trailing commas at the end of arrays or objects
* C-style, Python-style comments
* String literals use `"` and `'`.
  * `"Farmer's Gin`, `'Farmer\'s Fin'`
* `|||` means text blocks across multiple lines
* `@` means verbatim strings.
 
```js
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

* output.json

```json
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

The local keyword defines a variable.
Variables defined next to fields end with a comma (,).
All other cases end with a semicolon (;).

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

self refers to the current object.
`$` refers to the outer-most object.
`['foo']` looks up a field.
`.f` can be used if the field name is an identifier.
`[10]` looks up an array element.
Arbitrarily long paths are allowed.
Array slices like arr[10:20:2] are allowed, like in Python.
Strings can be looked up / sliced too, by unicode codepoint.

* references.jsonnet

```js
{
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
  },
  Martini: {
    ingredients: [
      {
        // Use the same gin as the Tom Collins.
        kind:
          $['Tom Collins'].ingredients[0].kind,
        qty: 2,
      },
      { kind: 'Dry White Vermouth', qty: 1 },
    ],
    garnish: 'Olive',
    served: 'Straight Up',
  },
  // Create an alias.
  'Gin Martini': self.Martini,
}
```

```json
{
  "Gin Martini": {
    "garnish": "Olive",
    "ingredients": [
      {
        "kind": "Farmer's Gin",
        "qty": 2
      },
      {
        "kind": "Dry White Vermouth",
        "qty": 1
      }
    ],
    "served": "Straight Up"
  },
  "Martini": {
    "garnish": "Olive",
    "ingredients": [
      {
        "kind": "Farmer's Gin",
        "qty": 2
      },
      {
        "kind": "Dry White Vermouth",
        "qty": 1
      }
    ],
    "served": "Straight Up"
  },
  "Tom Collins": {
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
```

* inner-reference.jsonnet

```js
{
  Martini: {
    local drink = self,
    ingredients: [
      { kind: "Farmer's Gin", qty: 1 },
      {
        kind: 'Dry White Vermouth',
        qty: drink.ingredients[0].qty,
      },
    ],
    garnish: 'Olive',
    served: 'Straight Up',
  },
}
```

```json
{
  "Martini": {
    "garnish": "Olive",
    "ingredients": [
      {
        "kind": "Farmer's Gin",
        "qty": 1
      },
      {
        "kind": "Dry White Vermouth",
        "qty": 1
      }
    ],
    "served": "Straight Up"
  }
}
```

## Arithmetic

Use floating point arithmetic, bitwise ops, boolean logic.
Strings may be concatenated with `+`, which implicitly converts one operand to string if needed.
Two strings can be compared with `<` (unicode codepoint order).
Objects may be combined with `+` where the right-hand side wins field conflicts.
Test if a field is in an object with in.
`==` is deep value equality.
Python-compatible string formatting is available via `%`. When combined with `|||` this can be used for templating text files.

* arith.jsonnet

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

## Functions

* functions.jsonnet

```js
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

* sours.jsonnet

```js
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

* conditionals.jsonnet

```js
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

Recall that a field lookup can be computed with `obj[e]`
The definition equivalent is `{[e]: ... }`
`self` or `object locals` cannot be accessed when field names are being computed, since the object is not yet constructed.
If a field name evaluates to null during object construction, the field is omitted. This works nicely with the default false branch of a conditional.

* computed-fields.jsonnet

```js
local Margarita(salted) = {
  ingredients: [
    { kind: 'Tequila Blanco', qty: 2 },
    { kind: 'Lime', qty: 1 },
    { kind: 'Cointreau', qty: 1 },
  ],
  [if salted then 'garnish']: 'Salt',
};
{
  Margarita: Margarita(true),
  'Margarita Unsalted': Margarita(false),
}
```

```json
{
  "Margarita": {
    "garnish": "Salt",
    "ingredients": [
      {
        "kind": "Tequila Blanco",
        "qty": 2
      },
      {
        "kind": "Lime",
        "qty": 1
      },
      {
        "kind": "Cointreau",
        "qty": 1
      }
    ]
  },
  "Margarita Unsalted": {
    "ingredients": [
      {
        "kind": "Tequila Blanco",
        "qty": 2
      },
      {
        "kind": "Lime",
        "qty": 1
      },
      {
        "kind": "Cointreau",
        "qty": 1
      }
    ]
  }
}
```

## Array and Object Comprehension

Any nesting of `for` and `if` can be used.
The nest behaves like a loop nest, although the body is written first.

```js
local arr = std.range(5, 8);
{
  array_comprehensions: {
    higher: [x + 3 for x in arr],
    lower: [x - 3 for x in arr],
    evens: [x for x in arr if x % 2 == 0],
    odds: [x for x in arr if x % 2 == 1],
    evens_and_odds: [
      '%d-%d' % [x, y]
      for x in arr
      if x % 2 == 0
      for y in arr
      if y % 2 == 1
    ],
  },
  object_comprehensions: {
    evens: {
      ['f' + x]: true
      for x in arr
      if x % 2 == 0
    },
    // Use object composition (+) to add in
    // static fields:
    mixture: {
      f: 1,
      g: 2,
    } + {
      [x]: 0
      for x in ['a', 'b', 'c']
    },
  },
}
```

```json
{
  "array_comprehensions": {
    "evens": [
      6,
      8
    ],
    "evens_and_odds": [
      "6-5",
      "6-7",
      "8-5",
      "8-7"
    ],
    "higher": [
      8,
      9,
      10,
      11
    ],
    "lower": [
      2,
      3,
      4,
      5
    ],
    "odds": [
      5,
      7
    ]
  },
  "object_comprehensions": {
    "evens": {
      "f6": true,
      "f8": true
    },
    "mixture": {
      "a": 0,
      "b": 0,
      "c": 0,
      "f": 1,
      "g": 2
    }
  }
}
```

* cocktail-comprehensions.jsonnet

```js
{
  cocktails: {
    "Bee's Knees": {
      // Construct the ingredients by using
      // 4/3 oz of each element in the given
      // list.
      ingredients: [  // Array comprehension.
        { kind: kind, qty: 4 / 3 }
        for kind in [
          'Honey Syrup',
          'Lemon Juice',
          'Farmers Gin',
        ]
      ],
      garnish: 'Lemon Twist',
      served: 'Straight Up',
    },
  } + {  // Object comprehension.
    [sd.name + 'Screwdriver']: {
      ingredients: [
        { kind: 'Vodka', qty: 1.5 },
        { kind: sd.fruit, qty: 3 },
      ],
      served: 'On The Rocks',
    }
    for sd in [
      { name: 'Yellow ', fruit: 'Lemonade' },
      { name: '', fruit: 'Orange Juice' },
    ]
  },
}
```

```json
{
  "cocktails": {
    "Bee's Knees": {
      "garnish": "Lemon Twist",
      "ingredients": [
        {
          "kind": "Honey Syrup",
          "qty": 1.3333333333333333
        },
        {
          "kind": "Lemon Juice",
          "qty": 1.3333333333333333
        },
        {
          "kind": "Farmers Gin",
          "qty": 1.3333333333333333
        }
      ],
      "served": "Straight Up"
    },
    "Screwdriver": {
      "ingredients": [
        {
          "kind": "Vodka",
          "qty": 1.5
        },
        {
          "kind": "Orange Juice",
          "qty": 3
        }
      ],
      "served": "On The Rocks"
    },
    "Yellow Screwdriver": {
      "ingredients": [
        {
          "kind": "Vodka",
          "qty": 1.5
        },
        {
          "kind": "Lemonade",
          "qty": 3
        }
      ],
      "served": "On The Rocks"
    }
  }
}
```

## Imports

The import construct is like copy/pasting Jsonnet code.
Files designed for import by convention end with .libsonnet
Raw JSON can be imported this way too.
The importstr construct is for verbatim UTF-8 text.

* martinis.libsonnet

```js
{
  'Vodka Martini': {
    ingredients: [
      { kind: 'Vodka', qty: 2 },
      { kind: 'Dry White Vermouth', qty: 1 },
    ],
    garnish: 'Olive',
    served: 'Straight Up',
  },
  Cosmopolitan: {
    ingredients: [
      { kind: 'Vodka', qty: 2 },
      { kind: 'Triple Sec', qty: 0.5 },
      { kind: 'Cranberry Juice', qty: 0.75 },
      { kind: 'Lime Juice', qty: 0.5 },
    ],
    garnish: 'Orange Peel',
    served: 'Straight Up',
  },
}
```

* garnish.txt

```
Maraschino Cherry
```

* imports.jsonnet

```js
local martinis = import 'martinis.libsonnet';

{
  'Vodka Martini': martinis['Vodka Martini'],
  Manhattan: {
    ingredients: [
      { kind: 'Rye', qty: 2.5 },
      { kind: 'Sweet Red Vermouth', qty: 1 },
      { kind: 'Angostura', qty: 'dash' },
    ],
    garnish: importstr 'garnish.txt',
    served: 'Straight Up',
  },
}
```

```js
{
  "Manhattan": {
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
  "Vodka Martini": {
    "garnish": "Olive",
    "ingredients": [
      {
        "kind": "Vodka",
        "qty": 2
      },
      {
        "kind": "Dry White Vermouth",
        "qty": 1
      }
    ],
    "served": "Straight Up"
  }
}
```

* utils.libsonnet

```js
{
  equal_parts(size, ingredients)::
    // Define a function-scoped variable.
    local qty = size / std.length(ingredients);
    // Return an array.
    [
      { kind: i, qty: qty }
      for i in ingredients
    ],
}
```

* negroni.jsonnet

```js
local utils = import 'utils.libsonnet';
{
  Negroni: {
    // Divide 3oz among the 3 ingredients.
    ingredients: utils.equal_parts(3, [
      'Farmers Gin',
      'Sweet Red Vermouth',
      'Campari',
    ]),
    garnish: 'Orange Peel',
    served: 'On The Rocks',
  },
}
```

```json
{
  "Negroni": {
    "garnish": "Orange Peel",
    "ingredients": [
      {
        "kind": "Farmers Gin",
        "qty": 1
      },
      {
        "kind": "Sweet Red Vermouth",
        "qty": 1
      },
      {
        "kind": "Campari",
        "qty": 1
      }
    ],
    "served": "On The Rocks"
  }
}
```

## Errors

To raise an error: `error "foo"`
To assert a condition before an expression: assert "foo";
A custom failure message: assert "foo" : "message";
Assert fields have a property: assert self.f == 10,
With custom failure message: assert "foo" : "message",

* error-examples.jsonnet

```js
// Extend above example to sanity check input.
local equal_parts(size, ingredients) =
  local qty = size / std.length(ingredients);
  // Check a pre-condition
  if std.length(ingredients) == 0 then
    error 'Empty ingredients.'
  else [
    { kind: i, qty: qty }
    for i in ingredients
  ];

local subtract(a, b) =
  assert a > b : 'a must be bigger than b';
  a - b;

assert std.isFunction(subtract);

{
  test1: equal_parts(1, ['Whiskey']),
  test2: subtract(10, 3),
  object: {
    assert self.f < self.g : 'wat',
    f: 1,
    g: 2,
  },
  assert std.isObject(self.object),
}
```

```json
{
  "object": {
    "f": 1,
    "g": 2
  },
  "test1": [
    {
      "kind": "Whiskey",
      "qty": 1
    }
  ],
  "test2": 7
}
```


## Parameterize Entire Config

**External variables**, which are accessible anywhere in the config, or any file, using std.extVar("foo").
**Top-level arguments**, where the whole config is expressed as a function.

### External variables

* library-ext.libsonnet

```js
local fizz = if std.extVar('brunch') then
  'Cheap Sparkling Wine'
else
  'Champagne';
{
  Mimosa: {
    ingredients: [
      { kind: fizz, qty: 3 },
      { kind: 'Orange Juice', qty: 3 },
    ],
    garnish: 'Orange Slice',
    served: 'Champagne Flute',
  },
}
```

* top-level-ext.jsonnet

```js
local lib = import 'library-ext.libsonnet';
{
  [std.extVar('prefix') + 'Pina Colada']: {
    ingredients: [
      { kind: 'Rum', qty: 3 },
      { kind: 'Pineapple Juice', qty: 6 },
      { kind: 'Coconut Cream', qty: 2 },
      { kind: 'Ice', qty: 12 },
    ],
    garnish: 'Pineapple slice',
    served: 'Frozen',
  },

  [if std.extVar('brunch') then
    std.extVar('prefix') + 'Bloody Mary'
  ]: {
    ingredients: [
      { kind: 'Vodka', qty: 1.5 },
      { kind: 'Tomato Juice', qty: 3 },
      { kind: 'Lemon Juice', qty: 1.5 },
      { kind: 'Worcestershire', qty: 0.25 },
      { kind: 'Tobasco Sauce', qty: 0.15 },
    ],
    garnish: 'Celery salt & pepper',
    served: 'Tall',
  },

  [std.extVar('prefix') + 'Mimosa']:
    lib.Mimosa,
}
```

```bash
$ jsonnet --ext-str prefix="Happy Hour " \
        --ext-code brunch=true top-level-ext.jsonnet
```

prefix is bound to the string "Happy Hour "
brunch is bound to true

```json
{
  "Happy Hour Bloody Mary": {
    "garnish": "Celery salt & pepper",
    "ingredients": [
      {
        "kind": "Vodka",
        "qty": 1.5
      },
      {
        "kind": "Tomato Juice",
        "qty": 3
      },
      {
        "kind": "Lemon Juice",
        "qty": 1.5
      },
      {
        "kind": "Worcestershire",
        "qty": 0.25
      },
      {
        "kind": "Tobasco Sauce",
        "qty": 0.15
      }
    ],
    "served": "Tall"
  },
  "Happy Hour Mimosa": {
    "garnish": "Orange Slice",
    "ingredients": [
      {
        "kind": "Cheap Sparkling Wine",
        "qty": 3
      },
      {
        "kind": "Orange Juice",
        "qty": 3
      }
    ],
    "served": "Champagne Flute"
  },
  "Happy Hour Pina Colada": {
    "garnish": "Pineapple slice",
    "ingredients": [
      {
        "kind": "Rum",
        "qty": 3
      },
      {
        "kind": "Pineapple Juice",
        "qty": 6
      },
      {
        "kind": "Coconut Cream",
        "qty": 2
      },
      {
        "kind": "Ice",
        "qty": 12
      }
    ],
    "served": "Frozen"
  }
}
```

### Top-level arguments

Values must be explicitly threaded through files
Default values can be provided
The config can be imported as a library and called as a function

* library-tla.libsonnet

```js
{
  // Note that the Mimosa is now
  // parameterized.
  Mimosa(brunch): {
    local fizz = if brunch then
      'Cheap Sparkling Wine'
    else
      'Champagne',
    ingredients: [
      { kind: fizz, qty: 3 },
      { kind: 'Orange Juice', qty: 3 },
    ],
    garnish: 'Orange Slice',
    served: 'Champagne Flute',
  },
}
```

* top-level-tla.jsonnet

```js
local lib = import 'library-tla.libsonnet';

// Here is the top-level function, note brunch
// now has a default value.
function(prefix, brunch=false) {

  [prefix + 'Pina Colada']: {
    ingredients: [
      { kind: 'Rum', qty: 3 },
      { kind: 'Pineapple Juice', qty: 6 },
      { kind: 'Coconut Cream', qty: 2 },
      { kind: 'Ice', qty: 12 },
    ],
    garnish: 'Pineapple slice',
    served: 'Frozen',
  },

  [if brunch then prefix + 'Bloody Mary']: {
    ingredients: [
      { kind: 'Vodka', qty: 1.5 },
      { kind: 'Tomato Juice', qty: 3 },
      { kind: 'Lemon Juice', qty: 1.5 },
      { kind: 'Worcestershire', qty: 0.25 },
      { kind: 'Tobasco Sauce', qty: 0.15 },
    ],
    garnish: 'Celery salt & pepper',
    served: 'Tall',
  },

  [prefix + 'Mimosa']: lib.Mimosa(brunch),
}
```

```bash
$ jsonnet --tla-str prefix="Happy Hour " \
        --tla-code brunch=true top-level-tla.jsonnet
```

```json
{
  "Happy Hour Bloody Mary": {
    "garnish": "Celery salt & pepper",
    "ingredients": [
      {
        "kind": "Vodka",
        "qty": 1.5
      },
      {
        "kind": "Tomato Juice",
        "qty": 3
      },
      {
        "kind": "Lemon Juice",
        "qty": 1.5
      },
      {
        "kind": "Worcestershire",
        "qty": 0.25
      },
      {
        "kind": "Tobasco Sauce",
        "qty": 0.15
      }
    ],
    "served": "Tall"
  },
  "Happy Hour Mimosa": {
    "garnish": "Orange Slice",
    "ingredients": [
      {
        "kind": "Cheap Sparkling Wine",
        "qty": 3
      },
      {
        "kind": "Orange Juice",
        "qty": 3
      }
    ],
    "served": "Champagne Flute"
  },
  "Happy Hour Pina Colada": {
    "garnish": "Pineapple slice",
    "ingredients": [
      {
        "kind": "Rum",
        "qty": 3
      },
      {
        "kind": "Pineapple Juice",
        "qty": 6
      },
      {
        "kind": "Coconut Cream",
        "qty": 2
      },
      {
        "kind": "Ice",
        "qty": 12
      }
    ],
    "served": "Frozen"
  }
}
```

## Object-Orientation

Objects (which we inherit from JSON)
The object composition operator `+`, which merges two objects, choosing the right hand side when fields collide
The `self` keyword, a reference to the current object

Hidden fields, defined with `::`, which do not appear in generated JSON
The `super` keyword, which has its usual meaning
The `+:` field syntax for overriding deeply nested fields

* oo-contrived.jsonnet

```js
local Base = {
  f: 2,
  g: self.f + 100,
};

local WrapperBase = {
  Base: Base,
};

{
  Derived: Base + {
    f: 5,
    old_f: super.f,
    old_g: super.g,
  },
  WrapperDerived: WrapperBase + {
    Base+: { f: 5 },
  },
}
```

```json
{
  "Derived": {
    "f": 5,
    "g": 105,
    "old_f": 2,
    "old_g": 105
  },
  "WrapperDerived": {
    "Base": {
      "f": 5,
      "g": 105
    }
  }
}
```

----

* templates.libsonnet

```js
{
  // Abstract template of a "sour" cocktail.
  Sour: {
    local drink = self,

    // Hidden fields can be referred to
    // and overrridden, but do not appear
    // in the JSON output.
    citrus:: {
      kind: 'Lemon Juice',
      qty: 1,
    },
    sweetener:: {
      kind: 'Simple Syrup',
      qty: 0.5,
    },

    // A field that must be overridden.
    spirit:: error 'Must override "spirit"',

    ingredients: [
      { kind: drink.spirit, qty: 2 },
      drink.citrus,
      drink.sweetener,
    ],
    garnish: self.citrus.kind + ' twist',
    served: 'Straight Up',
  },
}
```

* sours-oo.jsonnet

```js
local templates = import 'templates.libsonnet';

{
  // The template requires us to override
  // the 'spirit'.
  'Whiskey Sour': templates.Sour {
    spirit: 'Whiskey',
  },

  // Specialize it further.
  'Deluxe Sour': self['Whiskey Sour'] {
    // Don't replace the whole sweetner,
    // just change 'kind' within it.
    sweetener+: { kind: 'Gomme Syrup' },
  },

  Daiquiri: templates.Sour {
    spirit: 'Banks 7 Rum',
    citrus+: { kind: 'Lime' },
    // Any field can be overridden.
    garnish: 'Lime wedge',
  },

  "Nor'Easter": templates.Sour {
    spirit: 'Whiskey',
    citrus: { kind: 'Lime', qty: 0.5 },
    sweetener+: { kind: 'Maple Syrup' },
    // +: Can also add to a list.
    ingredients+: [
      { kind: 'Ginger Beer', qty: 1 },
    ],
  },
}
```

----

* sours-oo.jsonnet

```js
local templates = import 'templates.libsonnet';

{
  // The template requires us to override
  // the 'spirit'.
  'Whiskey Sour': templates.Sour {
    spirit: 'Whiskey',
  },

  // Specialize it further.
  'Deluxe Sour': self['Whiskey Sour'] {
    // Don't replace the whole sweetner,
    // just change 'kind' within it.
    sweetener+: { kind: 'Gomme Syrup' },
  },

  Daiquiri: templates.Sour {
    spirit: 'Banks 7 Rum',
    citrus+: { kind: 'Lime' },
    // Any field can be overridden.
    garnish: 'Lime wedge',
  },

  "Nor'Easter": templates.Sour {
    spirit: 'Whiskey',
    citrus: { kind: 'Lime', qty: 0.5 },
    sweetener+: { kind: 'Maple Syrup' },
    // +: Can also add to a list.
    ingredients+: [
      { kind: 'Ginger Beer', qty: 1 },
    ],
  },
}
```

* templaets.libssont

```js
{
  // Abstract template of a "sour" cocktail.
  Sour: {
    local drink = self,

    // Hidden fields can be referred to
    // and overrridden, but do not appear
    // in the JSON output.
    citrus:: {
      kind: 'Lemon Juice',
      qty: 1,
    },
    sweetener:: {
      kind: 'Simple Syrup',
      qty: 0.5,
    },

    // A field that must be overridden.
    spirit:: error 'Must override "spirit"',

    ingredients: [
      { kind: drink.spirit, qty: 2 },
      drink.citrus,
      drink.sweetener,
    ],
    garnish: self.citrus.kind + ' twist',
    served: 'Straight Up',
  },
}
```

* mixins.jsonnet

```js
local sours = import 'sours-oo.jsonnet';

local RemoveGarnish = {
  // Not technically removed, but made hidden.
  garnish:: super.garnish,
};

// Make virgin cocktails
local NoAlcohol = {
  local Substitute(ingredient) =
    local k = ingredient.kind;
    local bitters = 'Angustura Bitters';
    if k == 'Whiskey' then [
      { kind: 'Water', qty: ingredient.qty },
      { kind: bitters, qty: 'tsp' },
    ] else if k == 'Banks 7 Rum' then [
      { kind: 'Water', qty: ingredient.qty },
      { kind: 'Vanilla Essence', qty: 'dash' },
      { kind: bitters, qty: 'dash' },
    ] else [
      ingredient,
    ],
  ingredients: std.flattenArrays([
    Substitute(i)
    for i in super.ingredients
  ]),
};

local PartyMode = {
  served: 'In a plastic cup',
};

{
  'Whiskey Sour':
    sours['Whiskey Sour']
    + RemoveGarnish + PartyMode,

  'Virgin Whiskey Sour':
    sours['Whiskey Sour'] + NoAlcohol,

  'Virgin Daiquiri':
    sours.Daiquiri + NoAlcohol,

}
```

```
{
  "Virgin Daiquiri": {
    "garnish": "Lime wedge",
    "ingredients": [
      {
        "kind": "Water",
        "qty": 2
      },
      {
        "kind": "Vanilla Essence",
        "qty": "dash"
      },
      {
        "kind": "Angustura Bitters",
        "qty": "dash"
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
  "Virgin Whiskey Sour": {
    "garnish": "Lemon Juice twist",
    "ingredients": [
      {
        "kind": "Water",
        "qty": 2
      },
      {
        "kind": "Angustura Bitters",
        "qty": "tsp"
      },
      {
        "kind": "Lemon Juice",
        "qty": 1
      },
      {
        "kind": "Simple Syrup",
        "qty": 0.5
      }
    ],
    "served": "Straight Up"
  },
  "Whiskey Sour": {
    "ingredients": [
      {
        "kind": "Whiskey",
        "qty": 2
      },
      {
        "kind": "Lemon Juice",
        "qty": 1
      },
      {
        "kind": "Simple Syrup",
        "qty": 0.5
      }
    ],
    "served": "In a plastic cup"
  }
}
```