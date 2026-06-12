param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("DSA", "SQL")]
    [string]$Subject
)

$ErrorActionPreference = "Stop"

$topic = "utsav-study-7e2bc39fd6af49f0b7871ad8e6a018b0fa81b4fd138ab7d8"
$startDate = [datetime]"2026-06-12"
$today = [datetime]::Today
$elapsedDays = [int]($today - $startDate).TotalDays
$dayNumber = (($elapsedDays % 30) + 30) % 30 + 1

$dsaFocus = @(
    "Trees, Union Find, 1D DP",
    "Trees, Graph DFS/BFS, 1D DP",
    "BST, Advanced Graphs, 1D DP",
    "Heaps, Backtracking, 2D DP",
    "Heaps, Backtracking, 2D DP",
    "Tries, Greedy, 2D DP",
    "Tries, Greedy, Knapsack DP",
    "Segment Trees, Binary Search, Knapsack",
    "Segment Trees, Binary Search, Knapsack",
    "Bit Manipulation, Sliding Window, Knapsack",
    "Bit Manipulation, Sliding Window, String DP",
    "Monotonic Stack, Intervals, String DP",
    "Monotonic Stack, Intervals, String DP",
    "Shortest Paths, Linked Lists, Stock DP",
    "MST, Linked Lists, Stock DP",
    "Two Pointers, Math, Stock DP",
    "Two Pointers, Math, LIS DP",
    "Strings, Matrices, LIS DP",
    "Strings, Matrices, Partition DP",
    "Segment Trees, Tries, Tree DP",
    "Backtracking, DFS, Tree DP",
    "BFS, Topological Sort, Bitmask DP",
    "Design, Heaps, Bitmask DP",
    "Greedy, Two Pointers, Digit DP",
    "Segment Trees, Advanced Graphs, String DP",
    "Union Find, Monotonic Stack, Mixed DP",
    "Tries, Bit Manipulation, Hard String DP",
    "Sliding Window, BST, Mixed DP",
    "Implementations, Math, Mixed DP",
    "Mixed OA Finale"
)

$dsaProblems = @(
    "LC 746, 236, 987, 684, 547",
    "LC 198, 105, 199, 130, 127",
    "LC 213, 98, 230, 1631, 1192",
    "LC 62, 347, 295, 46, 78",
    "LC 64, 621, 973, 39, 51",
    "LC 174, 208, 211, 55, 134",
    "LC 322, 648, 212, 45, 452",
    "LC 416, 307, 315, 34, 74",
    "LC 494, 729, 699, 875, 1482",
    "LC 518, 137, 201, 424, 239",
    "LC 1143, 421, 1611, 567, 904",
    "LC 72, 739, 503, 56, 57",
    "LC 516, 901, 84, 435, 731",
    "LC 309, 743, 787, 146, 25",
    "LC 714, 1584, 778, 138, 143",
    "LC 123, 15, 11, 50, 12",
    "LC 300, 18, 42, 43, 48",
    "LC 354, 5, 8, 73, 54",
    "LC 312, 394, 438, 36, 289",
    "LC 337, 2381, 327, 677, 1268",
    "LC 124, 79, 93, 417, 200",
    "LC 473, 994, 1091, 210, 1203",
    "LC 1879, 355, 981, 767, 264",
    "LC 902, 763, 1647, 75, 16",
    "LC 97, 1109, 673, 785, 126",
    "LC 1312, 721, 990, 402, 85",
    "LC 10, 1023, 1707, 90, 318",
    "LC 1035, 209, 76, 450, 99",
    "LC 1911, 380, 59, 166, 224",
    "LC 221, 560, 41, 207, 399"
)

$sqlAssignments = @(
    "SELECT, WHERE, AND/OR; solve Q1 and Q3",
    "DISTINCT, ORDER BY, LIMIT; solve Q2 and Q4",
    "COUNT, SUM, AVG, MIN, MAX; solve Q5 and Q6",
    "GROUP BY and HAVING; solve Q7 and Q8",
    "Multi-column grouping and aggregate subqueries; solve Q9",
    "INNER and LEFT JOIN; solve Q10",
    "SELF JOIN and manager hierarchy; solve Q11",
    "CROSS JOIN and multi-table JOIN; solve Q12",
    "Consecutive rows with self-joins; solve Q13",
    "JOIN review: one Easy, one Medium, one Hard reattempt",
    "Scalar and correlated subqueries; solve Q14 and Q15",
    "CTEs and chained CTEs; solve Q16",
    "Rewrite two prior subqueries as CTEs; timed practice",
    "Window basics and DENSE_RANK; solve Q17",
    "LAG and LEAD; solve Q18",
    "Top N per group; solve Q19",
    "Running totals and rolling averages; solve Q20",
    "Window-function review; reattempt Q17-Q20 without hints",
    "String functions; solve Q21",
    "Date functions and string aggregation; solve Q22 and Q23",
    "CASE WHEN, NULL, COALESCE; solve Q24",
    "Conditional logic and row transformation; solve Q25",
    "EXISTS vs IN and conditional aggregation; solve Q26",
    "Advanced-pattern review; one Easy, Medium, and Hard",
    "Gaps and Islands; solve Q27",
    "UNION ALL and bidirectional relationships; solve Q28",
    "Complex filtering and rates; solve Q29",
    "Recursive CTE and median patterns; write two original queries",
    "60-minute mock: 3 Easy, 2 Medium, 1 Hard",
    "Final mock plus error-log review and weak-topic reattempt"
)

$index = $dayNumber - 1
if ($Subject -eq "DSA") {
    $title = "DSA Day $dayNumber - 12:00 PM"
    $message = "$($dsaFocus[$index])`n$($dsaProblems[$index])`nStart with DP, attempt before hints, and log mistakes."
    $tags = "computer,books"
} else {
    $title = "SQL Day $dayNumber - 4:00 PM"
    $message = "$($sqlAssignments[$index])`nWrite first, check the solution second, then record one reusable pattern."
    $tags = "bar_chart,books"
}

$headers = @{
    "Title"    = $title
    "Priority" = "high"
    "Tags"     = $tags
    "Click"    = "https://ntfy.sh/$topic"
}

Invoke-RestMethod `
    -Uri "https://ntfy.sh/$topic" `
    -Method Post `
    -Headers $headers `
    -ContentType "text/plain; charset=utf-8" `
    -Body $message | Out-Null
