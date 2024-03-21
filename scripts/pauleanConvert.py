import pandas as pd

def create_dataframe_from_file(file_path):
  data = {'refQuestion': [], 'answer': []}
  currQuest = None
  currAns = []

  with open(file_path, 'r') as file:
    for line in file:
      line = line.strip()
      if not line:
        continue
      if line.startswith('--'):
        currAns.append(line.lstrip('-').strip())
      else:
        if currQuest is not None:
          data['refQuestion'].append(currQuest)
          data['answer'].append(', '.join(currAns))
        currQuest = line
        currAns = []

    if currQuest is not None:
      data['refQuestion'].append(currQuest)
      data['answer'].append(', '.join(currAns))

  df = pd.DataFrame(data)
  return df

file_path = 'paulean.txt'
df = create_dataframe_from_file(file_path)
print(df)
df.to_csv('paulean.csv', index=False)
