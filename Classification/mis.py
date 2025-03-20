import re
import pandas as pd

# Read the content from sample.txt
with open("misclassified_samples.txt", "r", encoding="utf-8") as file:
    content = file.read()

# Remove unwanted symbols
cleaned_content = re.sub(r'[⁇]+|[?]+', '', content)

# Extract text, true label, and predicted label using regex
pattern = re.compile(
    r'Text: (.*?)\s*True Label: (\d+)\s*Predicted Label: (\d+)', re.DOTALL
)
matches = pattern.findall(cleaned_content)

# Convert to CSV format
data = []
for match in matches:
    text, true_label, predicted_label = match
    data.append([text.strip(), true_label, predicted_label])

# Create a DataFrame
df = pd.DataFrame(data, columns=["text", "true label", "predicted label"])

# Save as CSV
output_file = "mis-classified.csv"
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"CSV file saved successfully as '{output_file}'")
