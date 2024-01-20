import streamlit as st
import joblib
import pandas as pd
import os
# App.py dosyasının bulunduğu dizin
app_dizini = os.path.dirname(os.path.abspath(__file__))

# App.py dosyasının bulunduğu dizinden başlayarak dosya yolu oluşturma
model_yolu = os.path.join(app_dizini, '../Algoritmalar ve Preprocess/gradient_boosting_model.pkl')
veri_yolu =  os.path.join(app_dizini, '../Algoritmalar ve Preprocess/IslenmisVeri.csv')


def main():
    st.sidebar.title('Streamlit ile Ev Fiyat Tahmini')
    selected_page = st.sidebar.selectbox("Sayfa Seçiniz...", ['-','Tahmin Yap','Hakkında'])
    
    if selected_page == '-':
        st.title('Streamlit Uygulamasına Hoşgeldiniz.')
        st.write("""
        Bu uygulama, önceden eğitilmiş bir ev fiyat tahmin modeli kullanarak ev fiyatları tahmin etmenize olanak tanır. 
        Sol taraftaki menüden istediğiniz sayfayı seçebilirsiniz.
        """)
        
    elif selected_page == 'Tahmin Yap':
        predict()
    elif selected_page == 'Hakkında':
        about()

def about():
    st.title('Geliştirici Bilgileri')
    st.subheader('İsim: Alperen')
    st.subheader('Soyisim: Yılmaz')
    st.subheader('Okul: Celal Bayar Üniversitesi')
    st.subheader('Bölüm: Yazılım Mühendisliği')

def predict():
    st.title("Ev Fiyat Tahmini Yap")
    
    # Önceden eğitilmiş modeli yükle
    loaded_model = joblib.load(model_yolu)
    
    # Kullanıcıdan girişleri al
    metrekare = st.number_input("Metrekare:", value=100.0)
    oda_sayisi = st.number_input("Oda Sayısı:", value=2.0)
    kat = st.number_input("Kat:", value=1.0)
    konum = st.selectbox("Konum:", ['Balçova', 'Bayraklı', 'Bergama', 'Bornova', 'Buca','Dikili','Foça', 'Gaziemir', 'Güzelbahçe', 'Karabağlar', 'Karaburun','Balçova', 'Karşıyaka', 'Kemalpaşa', 'Konak', 'Menderes','Menemen', 'Narlıdere', 'Seferihisar', 'Selçuk', 'Tire','Torbalı', 'Urla', 'Çeşme', 'Çiğli', 'Ödemiş'])
    
    # Eğitim sırasında kullanılan sütunları belirle
    veri_seti = pd.read_csv(veri_yolu)
    egitim_sutunlari = ['Metrekare', 'Oda', 'Kat'] + [col for col in veri_seti.columns if col.startswith('Konum_')]
    
    # Kullanıcının girdilerini modele uygun formata getir
    input_data = [metrekare, oda_sayisi, kat] + [1 if col == f'Konum_{konum}' else 0 for col in egitim_sutunlari[3:]]
    
    # Tahmin yap
    tahmin = loaded_model.predict([input_data])
    
    # Tahmin sonucunu ekrana yazdır
    st.subheader("Tahmini Fiyat:")
    st.write(f"{tahmin[0]:,.2f} TL")

# Uygulamayı çalıştır
if __name__ == '__main__':
    main()