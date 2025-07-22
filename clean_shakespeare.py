import re

# Path to your large text file
input_file = 'shakespeare.txt'
output_file = 'clean_shakespeare.txt'

# The repeated chunk pattern (adjust as needed!)
pattern = r"<<.*?>>"

# If the chunk spans multiple lines, use re.DOTALL
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Remove all occurrences
cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)

# Save to a new file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(cleaned_text)

print("Done! Cleaned file saved as:", output_file)
