import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Gelişmiş veri setini yükle
print("📦 Gelişmiş zaman serisi veri seti yükleniyor...")
df = pd.read_csv('gelismis_zaman_serisi_verisi.csv')

# 2. Girdiler (X) ve Hedefler (y)
# Arkadaşının eklediği 5 saniyelik ortalama ve standart sapma verilerini AI'a veriyoruz!
X = df[['Senaryo_Tipi', 'Sniffer_X', 'Sniffer_Y', 'Sniffer_Z', 'RSSI_Anlik', 'RSSI_Mean_5s', 'RSSI_Std_5s']]
y = df[['Gercek_X', 'Gercek_Y', 'Gercek_Z', 'Gercek_Derinlik']]

# 3. Veriyi Eğitim (%80) ve Test (%20) olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modeli Eğit
print("🧠 ECHO-AI Modeli gelişmiş verilerle eğitiliyor (Bu işlem biraz sürebilir)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Test Et ve Hataları Göster
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions, multioutput='raw_values')

print("\n✅ Eğitim Tamamlandı! Yeni Modelin Hata Payları:")
print(f"X Koordinatı : {mae[0]:.2f} metre")
print(f"Y Koordinatı : {mae[1]:.2f} metre")
print(f"Z Koordinatı : {mae[2]:.2f} metre")
print(f"Derinlik     : {mae[3]:.2f} metre")

# 6. Eğitilen Modeli Kaydet
joblib.dump(model, 'echo_ai_model.pkl')
print("\n💾 Yeni Gelişmiş Model başarıyla 'echo_ai_model.pkl' olarak kaydedildi.")