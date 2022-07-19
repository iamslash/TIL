# Abstract

react-query 에 대해 정리한다. 

react-query 는 fetching, caching, synchronizing, updating server state 를 수월히
하게 해주는 library 이다.

# Materials

* [TanStack Query v4](https://tanstack.com/query/v4/docs/overview)
* [[React Query] 리액트 쿼리 시작하기 (useQuery) | velog](https://velog.io/@kimhyo_0218/React-Query-%EB%A6%AC%EC%95%A1%ED%8A%B8-%EC%BF%BC%EB%A6%AC-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0-useQuery)

# Basic

## Tutorial

`QueryClientProvider` Component 를 배치한다.

```js
import { QueryClient, QueryClientProvider } from "react-query";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Home />
    </QueryClientProvider>
  );
}
```

`useQuery()` 를 실행한다. [userQuery | react-query](https://tanstack.com/query/v4/docs/reference/useQuery?from=reactQueryV3&original=https://react-query-v3.tanstack.com/reference/useQuery)

```js
import { useQuery } from "react-query";
const { data, isLoading, error } = useQuery(queryKey, queryFn, options)
```
