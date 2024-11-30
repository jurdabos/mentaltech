# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Step 1: Data Collection
data = pd.read_csv('D:/Tanulás/iu/Subjects/INTERCULTURAL AND ETHICAL DECISION-MAKING/Visuals/company_structure.csv')

# %%
# Step 2: Data Processing
department_totals = data.groupby('Department')['Number of Employees'].sum()

# %%
# Step 3: Visualization
# A: Number of Employees per department type
plt.figure(figsize=(8, 6))
department_totals.plot(kind='bar', color='skyblue')
plt.title('Distribution of I\'MMIGRATION Employees by Department Type')
plt.xlabel('')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()

# %%
# B: The composition of the managerial circle
labels = ['Reg. Finance', 'Regional IT Managers', 'Regional HR Managers',
          'Reg. Comms', 'Regional Directors', 'CEO', 'CTO', 'COO',
          'HQ Finance Director', 'HQ IT Director', 'HQ Communications Director', 'Head of Global HR']
sizes = [6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1]
plt.figure(figsize=(10, 6))
plt.pie(sizes, labels=labels, startangle=140)
plt.title('Managerial Circle Distribution', pad=20)
plt.axis('equal')
total_people = sum(sizes)
plt.text(0, 0, f'Total: {total_people}', ha='center', va='center', fontsize=14)
plt.show()
plt.close()