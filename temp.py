import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Veriyi yükle
PATH = 'DataSet2.xlsx'  # Veri setinin dosya yolu
df = pd.read_excel(PATH)  # Veri setini Excel dosyasından oku

# Metin temizleme fonksiyonu
def remove_html(text):
    if isinstance(text, str):  # Metin verisi kontrolü
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    else:
        return str(text)

# Önişlem
max_obs = len(df)  # Tüm veri seti
observations_clean = []

for n in tqdm(range(max_obs)):
    observation = remove_html(df.iloc[n]['review'])
    clean = ' '.join(e.lower() for e in observation.split() if e.isalnum())
    observations_clean.append(clean)

set_of_words = set(' '.join(observations_clean).split())

dict_list = []

for clean in tqdm(observations_clean):
    dict_of_words = dict.fromkeys(set_of_words, 0)
    
    for word in clean.split(" "):
        if word in dict_of_words:
            dict_of_words[word] += 1
    dict_list.append(dict_of_words)

# DataFrame oluştur
df_word_counts = pd.DataFrame(dict_list)

# Etiketleri ve verileri hazırla
df['sentiment'] = df['sentiment'].str.strip()  # Boşlukları kaldır
df['sentiment'] = df['sentiment'].str.capitalize()  # İlk harfi büyük yap

y = df['sentiment'].replace({"Pozitif": 1, "Negatif": 0}).astype(int)
x = df_word_counts

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(x, y)

# Modeli oluştur
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğit
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Tahmin yap ve sonuçları yazdır
test = input("Bir cümle giriniz: ")
test_clean = ' '.join(e.lower() for e in test.split() if e.isalnum())
test_counts = {word: test_clean.split(" ").count(word) for word in set_of_words}
test_df = pd.DataFrame([test_counts])

pred = model.predict(test_df)

if pred[0][0] > 0.5:
    print("Olumlu bir cümle")
else:
    print("Olumsuz bir cümle")
