# Abstract

css 정리

# Materials

* []()

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

# Pseudo-class selectors

# Pseudo-elements selectors

# Attribute selectors
