- [Materials](#materials)
- [Data Manipulation](#data-manipulation)
  - [Prevent Lock Contention For Updates On Hot Rows](#prevent-lock-contention-for-updates-on-hot-rows)
  - [Updates Based On A Select Query](#updates-based-on-a-select-query)
  - [Return The Values Of Modified Rows](#return-the-values-of-modified-rows)
  - [Delete Duplicate Rows](#delete-duplicate-rows)
  - [Table Maintenance After Bulk Modifications](#table-maintenance-after-bulk-modifications)
- [Querying Data](#querying-data)
  - [Reduce The Amount Of Group By Columns](#reduce-the-amount-of-group-by-columns)
  - [Fill Tables With Large Amounts Of Test Data](#fill-tables-with-large-amounts-of-test-data)
  - [Simplified Inequality Checks With Nullable Columns](#simplified-inequality-checks-with-nullable-columns)
  - [Prevent Division By Zero Errors](#prevent-division-by-zero-errors)
  - [Sorting Order With Nullable Columns](#sorting-order-with-nullable-columns)
  - [Deterministic Ordering for Pagination](#deterministic-ordering-for-pagination)
  - [More Efficient Pagination Than LIMIT OFFSET](#more-efficient-pagination-than-limit-offset)
  - [Database-Backed Locks With Safety Guarantees](#database-backed-locks-with-safety-guarantees)
  - [Refinement Of Data With Common Table Expressions](#refinement-of-data-with-common-table-expressions)
  - [First Row Of Many Similar Ones](#first-row-of-many-similar-ones)
  - [Multiple Aggregates In One Query](#multiple-aggregates-in-one-query)
  - [Limit Rows Also Including Ties](#limit-rows-also-including-ties)
  - [Fast Row Count Estimates](#fast-row-count-estimates)
  - [Date-Based Statistical Queries With Gap-Filling](#date-based-statistical-queries-with-gap-filling)
  - [Table Joins With A For-Each Loop](#table-joins-with-a-for-each-loop)
- [Schema](#schema)
  - [Rows Without Overlapping Dates](#rows-without-overlapping-dates)
  - [Store Trees As Materialized Paths](#store-trees-as-materialized-paths)
  - [JSON Columns to Combine NoSQL and Relational Databases](#json-columns-to-combine-nosql-and-relational-databases)
  - [Alternative Tag Storage With JSON Arrays](#alternative-tag-storage-with-json-arrays)
  - [Constraints for Improved Data Strictness](#constraints-for-improved-data-strictness)
  - [Validation Of JSON Colums Against A Schema](#validation-of-json-colums-against-a-schema)
  - [UUID Keys Against Enumeration Attacks](#uuid-keys-against-enumeration-attacks)
  - [Fast Delete Of Big Data With Partitions](#fast-delete-of-big-data-with-partitions)
  - [Pre-Sorted Tables For Faster Access](#pre-sorted-tables-for-faster-access)
  - [Pre-Aggregation of Values for Faster Queres](#pre-aggregation-of-values-for-faster-queres)
- [Indexes](#indexes)
  - [Indexes On Functions And Expressions](#indexes-on-functions-and-expressions)
  - [Find Unused Indexes](#find-unused-indexes)
  - [Safely Deleting Unused Indexes](#safely-deleting-unused-indexes)
  - [Index-Only Operations By Including More Columns](#index-only-operations-by-including-more-columns)
  - [Partial Indexes To Reduce Index Size](#partial-indexes-to-reduce-index-size)
  - [Partial Indexes For Uniqueness Constraints](#partial-indexes-for-uniqueness-constraints)
  - [Index Support For Wildcard Searches](#index-support-for-wildcard-searches)
  - [Rules For Multi-Column Indexes](#rules-for-multi-column-indexes)
  - [Hash Indexes To Descrease Index Size](#hash-indexes-to-descrease-index-size)
  - [Descending Indexes For Order By](#descending-indexes-for-order-by)
  - [Ghost Conditions Against Unindexed Columns](#ghost-conditions-against-unindexed-columns)

----

# Materials

- [The Database Cookbook For Developers](https://sqlfordevs.com/ebook)

# Data Manipulation	

## Prevent Lock Contention For Updates On Hot Rows	

```sql
-- MySQL
INSERT INTO tweet_statistics (
    tweet_id, fanout, likes_count
) VALUES (
    1475870220422107137, FLOOR(RAND() * 10), 1
) ON DUPLICATE KEY UPDATE likes_count = 
likes_count + VALUES(likes_count);

-- PostgreSQL
INSERT INTO tweet_statistics (
  tweet_id, fanout, likes_count
) VALUES (
  1475870220422107137, FLOOR(RANDOM() * 10), 1
) ON CONFLICT (tweet_id, fanout) DO UPDATE SET likes_count =
tweet_statistics.likes_count + excluded.likes_count;
```

## Updates Based On A Select Query	
## Return The Values Of Modified Rows	
## Delete Duplicate Rows	
## Table Maintenance After Bulk Modifications	

# Querying Data	

## Reduce The Amount Of Group By Columns	
## Fill Tables With Large Amounts Of Test Data	
## Simplified Inequality Checks With Nullable Columns	
## Prevent Division By Zero Errors	
## Sorting Order With Nullable Columns	
## Deterministic Ordering for Pagination	
## More Efficient Pagination Than LIMIT OFFSET	
## Database-Backed Locks With Safety Guarantees	
## Refinement Of Data With Common Table Expressions	
## First Row Of Many Similar Ones	
## Multiple Aggregates In One Query	
## Limit Rows Also Including Ties	
## Fast Row Count Estimates	
## Date-Based Statistical Queries With Gap-Filling	
## Table Joins With A For-Each Loop	

# Schema	

## Rows Without Overlapping Dates	
## Store Trees As Materialized Paths	
## JSON Columns to Combine NoSQL and Relational Databases	
## Alternative Tag Storage With JSON Arrays	
## Constraints for Improved Data Strictness	
## Validation Of JSON Colums Against A Schema	
## UUID Keys Against Enumeration Attacks	
## Fast Delete Of Big Data With Partitions	
## Pre-Sorted Tables For Faster Access	
## Pre-Aggregation of Values for Faster Queres	

# Indexes	

## Indexes On Functions And Expressions	
## Find Unused Indexes	
## Safely Deleting Unused Indexes	
## Index-Only Operations By Including More Columns	
## Partial Indexes To Reduce Index Size	
## Partial Indexes For Uniqueness Constraints	
## Index Support For Wildcard Searches	
## Rules For Multi-Column Indexes	
## Hash Indexes To Descrease Index Size	
## Descending Indexes For Order By	
## Ghost Conditions Against Unindexed Columns	
