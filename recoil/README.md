# Abstract

recoil 은 state management library 이다.

# Materials

* [[Recoil][기초] | Recoil 사용법](https://velog.io/@myway_7/Recoil%EA%B8%B0%EC%B4%88-Recoil-%EC%82%AC%EC%9A%A9%EB%B2%95)
* [Recoil 시작하기 | recoil.js](https://recoiljs.org/ko/docs/introduction/getting-started)

# Basic

## Tutorial

`RecoilRoot` Component 를 배치한다.

```js
ReactDOM.render(
  <>
    <GlobalStyle />
    <ThemeProvider theme={{ ...theme, ...mixin }}>
      <RecoilRoot> 
        <Router /> 
      </RecoilRoot>
    </ThemeProvider>
  </>,
  document.getElementById('root')
);
```

`atom()` 호출하여 Global State 를 만든다.

```js
export const filterSelect = atom({
  key: 'filterSelect',
  default: {
    'start-date': dateConverter(new Date()),
    'end-date': dateConverter(new Date()),
    'weekly-date': dateConverter(new Date()),
    categories: '',
    subcategories: new Set(),
    seasons: new Set(),
    'serial-number': '',
    limit: 200,
    'deadline-week': dateConverter(setRecentSunday(new Date())),
  },
});
```

`useRecoilState()` 를 호출한다. 사용법은 `userState()` 과 같다.

```js
import filterSelect from 'file path' // atom으로 만든 전역상태
import {useRecoilState} from 'recoil' // 훅 import

const [state, setState] = useRecoilState(filterSelect); // 전역상태를 state로 만듦
```

`setState()` 를 호출하여 `state` 를 변경해 보자.

```js
setState(prev => {
	const variable = {...prev};

	variable.property1 = ....;
	.
	.
	return {...variable}
}
)
```
