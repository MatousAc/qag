import csv
import json

# Open the CSV file for reading
with open('data/nkjv.csv', newline='') as csvfile:
  # Read the CSV file using a DictReader
  reader = csv.DictReader(csvfile)
  # Create a dictionary to hold the data
  data = {}
  # Loop through each row in the CSV file
  for row in reader:
    book = row['Book']
    chapter_number = row['ChapterNumber']
    verse_number = row['VerseNumber']
    verse = row['Verse']
    
    # Create the nested structure if it doesn't exist
    if book not in data:
      data[book] = {}
    if chapter_number not in data[book]:
      data[book][chapter_number] = {}
    
    # Add the verse to the nested structure
    data[book][chapter_number][verse_number] = verse

# Open the JSON file for writing
with open('data/nkjv.json', 'w') as jsonfile:
  # Write the data to the JSON file
  json.dump(data, jsonfile, indent=2)
