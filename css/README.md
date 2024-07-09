- [Abstract](#abstract)
- [CSS Syntax](#css-syntax)
- [CSS Selectors](#css-selectors)
- [Simple Selectors](#simple-selectors)
  - [The CSS element Selector](#the-css-element-selector)
  - [The CSS id Selector](#the-css-id-selector)
  - [The CSS class Selector](#the-css-class-selector)
  - [The CSS Universal Selector](#the-css-universal-selector)
  - [The CSS Grouping Selector](#the-css-grouping-selector)
- [Combinator selectors](#combinator-selectors)
  - [The CSS Descendant Selector](#the-css-descendant-selector)
  - [The CSS Child Selector](#the-css-child-selector)
  - [The CSS Adjacent Sibling Selector](#the-css-adjacent-sibling-selector)
  - [The CSS General Sibling Selector](#the-css-general-sibling-selector)
- [Pseudo-class selectors](#pseudo-class-selectors)
  - [The CSS :hover Selector](#the-css-hover-selector)
  - [The CSS :focus Selector](#the-css-focus-selector)
  - [The CSS :nth-child Selector](#the-css-nth-child-selector)
- [Pseudo-elements selectors](#pseudo-elements-selectors)
  - [The CSS ::after Selector](#the-css-after-selector)
  - [The CSS ::before Selector](#the-css-before-selector)
  - [The CSS ::first-line Selector](#the-css-first-line-selector)
- [Attribute selectors](#attribute-selectors)
  - [The CSS \[attribute\] Selector](#the-css-attribute-selector)
  - [The CSS \[attribute=value\] Selector](#the-css-attributevalue-selector)
  - [The CSS \[attribute~=value\] Selector](#the-css-attributevalue-selector-1)
  - [The CSS \[attribute|=value\] Selector](#the-css-attributevalue-selector-2)

-----

# Abstract

css 정리

# CSS Syntax

> [CSS Syntax | w3schools](https://www.w3schools.com/css/css_syntax.asp)

![](https://www.w3schools.com/css/img_selector.gif)

# CSS Selectors

> [CSS Selectors | w3schools](https://www.w3schools.com/css/css_selectors.asp)

다음과 같은 Selectors 가 있음.

* **Simple selectors** (select elements based on name, id, class)
* **Combinator selectors** (select elements based on a specific relationship between them)
* **Pseudo-class selectors** (select elements based on a certain state)
* **Pseudo-elements selectors** (select and style a part of an element)
* **Attribute selectors** (select elements based on an attribute or attribute value)

# Simple Selectors

## The CSS element Selector

모든 element 에 적용하라.

```css
p {
  text-align: center;
  color: red;
}
```

## The CSS id Selector

특정 id attribute 를 갖는 element 에 적용하라.

```css
#para1 {
  text-align: center;
  color: red;
}
```

## The CSS class Selector

특정 class attribute 를 갖는 element 에게 적용하라.

```css
.center {
  text-align: center;
  color: red;
}

p.center {
  text-align: center;
  color: red;
}

<p class="center large">This paragraph refers to two classes.</p>
```

## The CSS Universal Selector

몽땅 적용하라.

```css
* {
  text-align: center;
  color: blue;
}
```

## The CSS Grouping Selector

특정 element 들에 적용하라. 

```css
h1, h2, p {
  text-align: center;
  color: red;
}
```

# Combinator selectors

## The CSS Descendant Selector

특정 요소의 자손 요소에 적용하라.

```css
div p {
  color: blue;
}
```

## The CSS Child Selector

특정 요소의 자식 요소에 적용하라.

```css
div > p {
  color: red;
}
```

## The CSS Adjacent Sibling Selector

특정 요소의 인접 형제 요소에 적용하라.

```css
h1 + p {
  color: green;
}
```

## The CSS General Sibling Selector

특정 요소의 모든 형제 요소에 적용하라.

```css
h1 ~ p {
  color: yellow;
}
```

# Pseudo-class selectors

## The CSS :hover Selector

마우스를 올렸을 때 적용하라.

```css
a:hover {
  color: red;
}
```

## The CSS :focus Selector

포커스가 맞춰졌을 때 적용하라.

```css
input:focus {
  background-color: yellow;
}
```

## The CSS :nth-child Selector

n 번째 자식 요소에 적용하라.

```css
p:nth-child(2) {
  color: blue;
}
```

# Pseudo-elements selectors

## The CSS ::after Selector

특정 요소의 마지막에 내용을 추가하라.

```css
p::after {
  content: " - Read more";
}
```

## The CSS ::before Selector

특정 요소의 앞에 내용을 추가하라.

```css
p::before {
  content: "Note: ";
}
```

## The CSS ::first-line Selector

특정 요소의 첫 번째 줄에 스타일을 적용하라.

```css
p::first-line {
  color: red;
}
```

# Attribute selectors

## The CSS [attribute] Selector

특정 속성을 가진 요소에 적용하라.

```css
a[target] {
  background-color: yellow;
}
```

## The CSS [attribute=value] Selector

특정 속성과 값을 가진 요소에 적용하라.

```css
a[target="_blank"] {
  color: red;
}
```

## The CSS [attribute~=value] Selector

특정 속성의 값이 포함된 요소에 적용하라.

```css
[title~="flower"] {
  color: blue;
}
```

## The CSS [attribute|=value] Selector

특정 속성의 값이 특정 문자열로 시작하는 요소에 적용하라.

```css
[lang|="en"] {
  color: green;
}
```
