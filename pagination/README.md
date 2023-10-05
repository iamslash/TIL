# Materials

* [[Spring Boot] QueryDSL 커서 기반 페이지네이션 구현해보기 | velog](https://velog.io/@ohjinseo/Spring-querydsl-%EC%BB%A4%EC%84%9C-%EA%B8%B0%EB%B0%98-%ED%8E%98%EC%9D%B4%EC%A7%80%EB%84%A4%EC%9D%B4%EC%85%98-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EA%B8%B0)
* [[Paging] Offset 페이징과 Cursor 페이징 차이](https://devlog-wjdrbs96.tistory.com/440)
* [JPA를 이용하여 cursor 기반 페이징 구현 | tistory](https://alwayspr.tistory.com/45)

----

# Abstract

Pagination has two options including **offset-based**, **cursor-based**.

# Offset-Based

Pagination with **offset, limit**. We need to get all data from RDBMS because of offset. It is suitable with navigation page UI.

```sql
-- n: page number, m: size
-- When n is big, read performance is bad.
SELECT * FROM salaries ORDER BY salary LIMIT n, m;
```

* Pros
  * Easy to implement.
  * Easy to move to any page.
* Cons:
  * Need to get all data for offset, not suitable for large data.

# Cursor-Based

Pagination with **curosr, limit**. It is suitable with previous, next page UI.

```sql
  SELECT s.* 
    FROM salaries s 
   WHERE s.id > salary_id
ORDER BY s.id DESC
   LIMIT 10
```

* Pros
  * Just get some data after cursor, suitable for large data.
* Cons:
  * Difficult to implement. Especially, with many search conditions. 
  * Easy to move just to previous, next page.

JPA examples

```java
@Entity
public class Board {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;
    
    private String title;
    
    private String contents;
    
    private LocalDateTime createAt;
	...
}

public class CursorResult<T> {

    private List<T> values;
    
    private Boolean hasNext;

    public CursorResult(List<T> values, Boolean hasNext) {
        this.values = values;
        this.hasNext = hasNext;
    }
    ...
}

@RestController
@RequestMapping("/boards")
public class BoardController {

    private static final int DEFAULT_SIZE = 10;

    private final BoardService boardService;

    public BoardController(BoardService boardService) {
        this.boardService = boardService;
    }

    @GetMapping
    public CursorResult<Board> getBoards(Long cursorId, Integer size) {
        if (size == null) {
            size = DEFAULT_SIZE;
        }
        return this.boardService.get(cursorId, PageRequest.of(0, size));
    }
}

@Service
public class BoardService {

    private final BoardRepository boardRepository;

    public BoardService(BoardRepository boardRepository) {
        this.boardRepository = boardRepository;
    }

    CursorResult<Board> get(Long cursorId, Pageable page) {
        final List<Board> boards = getBoards(cursorId, page);
        final Long lastIdOfList = boards.isEmpty() ?
                null : boards.get(boards.size() - 1).getId();

        return new CursorResult<>(boards, hasNext(lastIdOfList));
    }

    private List<Board> getBoards(Long id, Pageable page) {
        return id == null ?
                this.boardRepository.findAllByOrderByIdDesc(page) :
                this.boardRepository.findByIdLessThanOrderByIdDesc(id, page);
    }

    private Boolean hasNext(Long id) {
        if (id == null) {
            return false;
        }
        return this.boardRepository.existsByIdLessThan(id);
    }
}
```
