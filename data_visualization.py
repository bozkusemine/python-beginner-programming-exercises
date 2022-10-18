import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
[col for col in df.columns if df[col].dtypes =="o"]



df["sex"].value_counts().plot(kind="bar")
plt.show()


plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

import seaborn as sns
df = sns.load_dataset("tips")
sns.scatterplot(x=df["tip"], y=df["total_bill"],
                hue=df["smoker"], data=df)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
titanic = sns.load_dataset("titanic")
sns.countplot(x="class", data=titanic)
plt.show()