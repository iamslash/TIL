# Abstract

How to parse the log. You need to practise it.

# Useful Commands

```bash
1. GREP

# Find all lines containing the word "example" in a file:
grep "example" file.txt

# Find a list of files that have a specific text pattern (case insensitive):
grep -i -r "search_pattern" /path/to/your/directory

# Count the number of lines with a specific pattern:
grep -c "pattern" file.txt

2. CUT

# Remove the first column from a CSV file:
cut -f 2- -d ',' file.csv

# Extract the first 10 characters of each line from a file:
cut -c 1-10 file.txt

# Get the username and user ID fields from /etc/passwd:
cut -d ':' -f 1,3 /etc/passwd

3. SED

# Replace all occurrences of the word "apple" with "orange" in a file:
sed 's/apple/orange/g' input.txt > output.txt

# Delete lines containing a specific pattern in a file:
sed '/pattern_to_delete/d' input.txt > output.txt

# Insert a line of text after a specific line number in a file:
sed '2i Inserted line of text' input.txt > output.txt

4. AWK

# Sum the values in the second column of a file:
awk '{sum += $2} END {print sum}' file.txt

# Print the lines where the value in the third column is greater than 10:
awk '$3 > 10' file.txt

# Print the first and third column separated by a comma:
awk -F '\t' '{print $1 "," $3}' file.txt

5. SORT

# Sort a file alphabetically based on the first column:
sort -k 1,1 file.txt

# Sort a file numerically based on the second column in descending order:
sort -k 2,2nr file.txt

# Sort a file alphabetically and remove duplicate lines:
sort -u file.txt

6. UNIQ

# Remove duplicate adjacent lines from a file and display result:
uniq file.txt

# Count the number of occurrences of each line in a file:
uniq -c file.txt

# Display only the unique lines in a file (not repeated):
uniq -u file.txt
```

# Useful Mixed Examples

```bash
# Find the top 10 most frequent words in a file:
grep -oE '\w+' file.txt | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -nr | head -10

# List the top 5 largest files or directories in a directory:
du -a /path/to/your/directory | sort -nr | head -5

# Show total disk usage for each user on the system:
awk -F: '{print $1}' /etc/passwd | xargs -I {} sudo -u {} du -sh /home/{} 2>/dev/null | sort -rh

# Find and replace a specific text pattern in multiple files:
grep -rl "search_pattern" /path/to/your/directory | xargs sed -i 's/search_pattern/replacement_text/g'

# Display the longest line from a file:
awk '{ if (length > maxLength) { maxLength = length; longestLine = $0 } } END { print longestLine }' file.txt

# Find the top 3 most common IP addresses in a log file:
grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' log_file.txt | sort | uniq -c | sort -nr | head -3

# List total lines of code for each file in a directory:
find /path/to/your/directory -type f -name "*.c" -or -name "*.cpp" -or -name "*.h" | xargs wc -l | sort -nr

# Calculate the total time duration of a list of video files:
find /path/to/your/video/directory -type f -name "*.mp4" -or -name "*.mkv" | xargs -I {} ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {} | awk '{sum += $1} END {printf("%02d:%02d:%02d\n", sum/3600, sum%3600/60, sum%60)}'

# Show the top 10 most frequently used commands in the shell history:
awk '{CMD[$2]++;count++;}END { for (a in CMD)print CMD[a] " " CMD[a]/count*100 "% " a; }' <(grep -i "command" .bash_history) | sort -nr | head -n 10

# Get a list of all open ports and associated processes:
sudo netstat -tuln | awk '{print $4 " " $7}' | sed -n '3,$p' | sed 's/.*://g' | sort | uniq -c | sort -nr
```
