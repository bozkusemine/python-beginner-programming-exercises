
##################################################
# List Comprehensions
##################################################

# ###############################################
# # GÖREV 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
# ###############################################
#
# # Beklenen Çıktı
#
# # ['NUM_TOTAL',
# #  'NUM_SPEEDING',
# #  'NUM_ALCOHOL',
# #  'NUM_NOT_DISTRACTED',
# #  'NUM_NO_PREVIOUS',
# #  'NUM_INS_PREMIUM',
# #  'NUM_INS_LOSSES',
# #  'ABBREV']
#
# # Notlar:
# # Numerik olmayanların da isimleri büyümeli.
# # Tek bir list comp yapısı ile yapılmalı.

import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("car_crashes")
df.columns
df.info()

[col for col in df.columns if df[col].dtype !="O"]
## harf büyütme
[col.upper() for col in df.columns if df[col].dtype !="O"]
#Hepsine Num ekleme
["NUM_"+col.upper() for col in df.columns if df[col].dtype != "O"]

["NUM_"+col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

# ###############################################
# # GÖREV 2: List Comprehension yapısı kullanarak car_crashes verisindeki isminde "no" barındırmayan değişkenlerin isimlerininin sonuna "FLAG" yazınız.
# ###############################################
#
# # Notlar:
# # Tüm değişken isimleri büyük olmalı.
# # Tek bir list comp ile yapılmalı.
#
# # Beklenen çıktı:
#
# # ['TOTAL_FLAG',
# #  'SPEEDING_FLAG',
# #  'ALCOHOL_FLAG',
# #  'NOT_DISTRACTED',
# #  'NO_PREVIOUS',
# #  'INS_PREMIUM_FLAG',
# #  'INS_LOSSES_FLAG',
# #  'ABBREV_FLAG']

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
[col.upper() +"_FLAG" if "no" not in col else col.upper() for col in df.columns]

# ###############################################
# # Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.
# ###############################################

# # Notlar:
# # Önce yukarıdaki listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
# # Sonra df[new_cols] ile bu değişkenleri seçerek yeni bir df oluşturunuz adını new_df olarak isimlendiriniz.
#
# # Beklenen çıktı:
#
# #    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# # 0 18.800     7.332    5.640          18.048      784.550     145.080
# # 1 18.100     7.421    4.525          16.290     1053.480     133.930
# # 2 18.600     6.510    5.208          15.624      899.470     110.350
# # 3 22.400     4.032    5.824          21.056      827.340     142.390
# # 4 12.000     4.200    3.360          10.920      878.410     165.630
#
import seaborn as sns

df=sns.load_dataset("car_crashes")
df.head()
df.columns
og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
df = df[new_cols]
df

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
import numpy as np
#kütüphane kullanılmadığı için sönük renkli
#örneğin np.array() eklersek kütüphane çalışır duruma gelir.

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

#veri otomatik olarak kütüphane içerisinde yok ve csv dosyası olarak eklemek istersek;

data = pd.read_csv("titanic.csv")
df = data.copy()
df.head()

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################
#Veri kümesinde veya bir sütunda oluşan benzersiz değerlerin sayısını saymak için .nunique() kullanırız.

df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################
#Ayrıca bir sütun için benzersiz kayıtları .nunique() ile sayabiliriz. Tek yapmamız gereken sütun adını eklemek.

df["pclass"].unique()

#df["sex"].unique()
#df["sex"].nunique()

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
#Birden fazla sütun için unique sayısını görmek istiyorsak, bir tane daha köşeli parantez eklememiz gerekir.

df[["pclass", "parch"]].nunique()

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################
#Titanic veri setindeki kolonların veri tiplerini kontrol ettik. Embarked değişkenin tipinin bir nesne olduğunu gördük.

df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()

#CategoricalDtype(categories=['C', 'Q', 'S'], ordered=False)

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"] == "C"].head(10)


#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"] != "S"]["embarked"].unique()
df[~(df["embarked"] == "S")]["embarked"].unique()
df[df["embarked"] != "S"]["embarked"].head()


#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df["age"] < 30) & (df["sex"] == "female")].head()
df.loc[(df["age"] < 30) & (df["sex"] == "female")].head()

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

new_df = df.loc[(df["fare"] > 500) | (df["age"] > 70)]
new_df

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################
#Veri bilimindeki en yaygın sorunlardan biri eksik değerlerdir.
# Bunları tespit etmek için .isnull() adında güzel bir yöntem var.
# Bu yöntemle bir boolean serisi (True veya False) elde edebiliriz.

df.isnull().sum()

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop("who", axis=1, inplace=True).head()
df.drop(columns="who", inplace=True).head()

#########################################
# Görev 13: deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

type(df["deck"].mode())
df["deck"].mode()[0]

df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()

#########################################
# Görev 14: age değişkenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df["age"].median()
df["age"].fillna(df["age"].median(), inplace=True)
df.isnull().sum()

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################
def age_func(age):
    if age < 30:
        return 1
    else:
        return 0
df["age_flag"] = df["age"].apply(lambda x: age_func(x))

#istenilen yol
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)
df.head()

#aply ve lambda fonk kullanılmadan da cözüm yapılır.
df['age_flag'] = df["age"].apply(age_func)
#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")
df.head()
df.describe()
df.shape
#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg(
    {"total_bill": ["sum", "min", "max", "mean"], "tip": ["sum", "min", "max", "mean"]})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.head()
new_df.shape


