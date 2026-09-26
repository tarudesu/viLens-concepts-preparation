Step 09b fix: the blank Word at Excel row 21,732 is pandas' default NA parsing
("null" sits alphabetically between "nuke" and "nullification"), not a data defect.

1. Read the workbook with keep\_default\_na=False and na\_values={} (or read Word with
   dtype=str and no NA conversion), so literal strings like "null", "nan", "NA",
   "None" are kept as words.
2. Verify: 0 blank Word cells; row 21,732 reads "null"; print any Word values that
   pandas' DEFAULT NA list would have converted (so we can see all affected words).
3. Keep the loader's rule: if a Word cell is ever truly empty, STOP (don't skip silently).
4. Test: a fixture xlsx containing the words "null", "nan" and "NA" loads them as
   strings.
   Commit as "step 09b: keep literal NA-like words", then resume TASK 15 STEP B from
   make concreteness onward, as specified.
