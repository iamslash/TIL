# Abstract

styled component 에 대해 정리한다.

# Materials

* [Styled Components로 React 컴포넌트 스타일하기](https://www.daleseo.com/react-styled-components/)
* [Styled Components Basic | styled-components](https://styled-components.com/docs/basics)

# Basic

## Fixed Styled Button

ES6 는 [Tagged Template Literals](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Template_literals) 를 지원한다. 즉, backtick 을 이용하여 문자열을 만들어 낼 수 있다.

```js
////////////////////////////////////////
// StyledComponent.js
const FixButton = styled.button`
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 1rem;
  line-height: 1.5;
  border: 1px solid lightgray;
  color: gray;
  background: white;
`;
export function FixStyledButton({children}) {
    return <FixButton>{children}</FixButton>;
}
export default {FixStyledButton};

////////////////////////////////////////
// App.js
import {FixStyledButton} from './StyledComponent';

function App() {
  return <div>
    <div>
      <FixStyledButton>
        Hello World
      </FixStyledButton>
    </div>
  </div>;
}

export default App;
```

## Variable Styled Button

변수를 [Tagged Template Literals](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Template_literals) 삽입할 수도 있다.

```js
////////////////////////////////////////
// StyledComponent.js
const VarButton = styled.button`
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 1rem;
  line-height: 1.5;
  border: 1px solid lightgray;

  color: ${(props) => props.color || "gray"};
  background: ${(props) => props.background || "white"};
`;

export function VarStyledButton({ children, color, background }) {
    return (
        <VarButton color={color} background={background} Î>
        {children}
        </VarButton>
    );
}

export default {VarStyledButton};

////////////////////////////////////////
// App.js
import {VarStyledButton} from './StyledComponent';

function App() {
  return <div>
    <div>
      <VarStyledButton color='green' background='light gray'>
        Hello World
      </VarStyledButton>
    </div>
  </div>;
}

export default App;
```

## Css Styled Button

css tag 를 [Tagged Template Literals](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Template_literals) 에 삽입할 수도 있다. tag 는 곧 function 이다.

```js
////////////////////////////////////////
// StyledComponent.js
const CssButton = styled.button`
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 1rem;
  line-height: 1.5;
  border: 1px solid lightgray;

  ${(props) =>
    props.primary &&
    css`
      color: white;
      background: navy;
      border-color: navy;
    `}
`;

export function CssStyledButton({ children, ...props }) {
    return <CssButton {...props}>{children}</CssButton>;
}

export default {CssStyledButton};

////////////////////////////////////////
// App.js
import {CssStyledButton} from './StyledComponent';

function App() {
  return <div>
    <div>
      <CssStyledButton>
        Hello World
      </CssStyledButton>
    </div>
  </div>;
}

export default App;
```
