import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import joblib
import plotly.express as px

LANGUAGES = {
    "tr": {
        "page_title": "Volatilite Tahmini",
        "title": "📈 Hisse Senedi Volatilite Tahmincisi",
        "tab_predict": "🧠 Tahmin Aracı",
        "tab_details": "🎯 Proje Detayları",
        "header_predict": "Canlı Volatilite Tahmini",
        "description_predict": "Bir hisse senedinin önümüzdeki 5 gün için beklenen volatilitesini (fiyat dalgalanmasını) tahmin edin.",
        "select_stock": "Tahmin için bir Hisse Senedi Seçin",
        "button_predict": "Volatiliteyi Tahmin Et",
        "spinner_text": "için en son veriler çekiliyor...",
        "error_no_data": "için veri bulunamadı. Lütfen geçerli bir sembol girin.",
        "warning_insufficient_data": "Tahmin için yeterli geçmiş veri bulunamadı. Lütfen daha bilinen bir hisse senedi deneyin.",
        "prediction_header": "İçin Tahmin Sonucu",
        "prediction_label": "Önümüzdeki 5 Günlük Beklenen Günlük Volatilite",
        "volatility_high": "Yüksek Dalgalanma Bekleniyor: Fiyatlarda sert hareketler görülebilir.",
        "volatility_medium": "Orta Derecede Dalgalanma Bekleniyor: Fiyatlarda ılımlı hareketler görülebilir.",
        "volatility_low": "Düşük Dalgalanma Bekleniyor: Fiyatların daha stabil olması bekleniyor.",
        "chart_title": "Son 1 Aylık Kapanış Fiyatı Grafiği",
        "details_header": "Projenin Amacı ve Teknik Detaylar",
        "details_text": """
        Bu projenin amacı, finansal piyasalardaki riskin bir ölçütü olan **volatiliteyi (oynaklığı)** tahmin etmektir. Volatilite, bir hisse senedi fiyatının belirli bir süre içinde ne kadar dalgalandığını gösterir. Yüksek volatilite, yüksek risk ve potansiyel olarak yüksek getiri anlamına gelir.
        
        - **Model:** XGBoost Regressor
        - **Veri Kaynağı:** `alpha-vantage` kütüphanesi ile anlık olarak Alpha Vantage API'sinden çekilmektedir.
        - **Özellikler:** Model, bir hissenin geçmiş getirileri (lag features), hareketli ortalamaları ve geçmiş volatilitesi gibi temel zaman serisi özelliklerini kullanarak tahmin yapar.

        ---

        ### **Projenin Kapsamı ve Limitleri**

        Bu uygulama, **#30GündeYapayZeka** maratonumun 5. gününde, **sadece bir günlük bir sürede geliştirilmiş bir konsept kanıtlama (Proof of Concept) projesidir.** Temel amacı, canlı API'dan veri çekme, özellik mühendisliği, model eğitimi ve interaktif bir arayüzde sunum gibi uçtan uca adımları kapsamlı bir şekilde pratik etmektir.

        Bu bağlamda, mevcut modelin bir yatırım tavsiyesi aracı **olmadığını** ve gerçek dünya uygulamaları için aşağıdaki gibi ek metriklere ve geliştirmelere ihtiyaç duyduğunu belirtmek isterim:

        * **Kapsamlı Model Değerlendirmesi:** Model, henüz bir test verisi üzerinde R-kare ($R^2$), Ortalama Mutlak Hata (MAE) gibi standart regresyon metrikleri ile değerlendirilmemiştir.
        * **Sınırlı Özellikler:** Tahminler yalnızca geçmiş fiyat hareketlerine dayanmaktadır. İşlem hacmi, teknik göstergeler (RSI, Bollinger Bantları vb.) veya temel analiz verileri (şirket haberleri, finansal raporlar) gibi önemli metrikler henüz dahil edilmemiştir.
        * **Gelecek Geliştirme Fırsatları:** Modelin daha isabetli ve güvenilir olması için, farklı sektörlerden çok sayıda hisse senedi ile eğitilmesi, hiperparametre optimizasyonu yapılması ve tahminlerin bir belirsizlik aralığı ile sunulması gibi adımlar planlanabilir.
        """,
        "language_select_label": "Dil"
    },
    "en": {
        "page_title": "Volatility Prediction",
        "title": "📈 Stock Volatility Predictor",
        "tab_predict": "🧠 Prediction Tool",
        "tab_details": "🎯 Project Details",
        "header_predict": "Live Volatility Prediction",
        "description_predict": "Predict the expected volatility (price fluctuation) of a stock for the next 5 days.",
        "select_stock": "Select a Stock for Prediction",
        "button_predict": "Predict Volatility",
        "spinner_text": "Fetching latest data for...",
        "error_no_data": "No data found for. Please enter a valid symbol.",
        "warning_insufficient_data": "Not enough historical data to make a prediction. Please try a more well-known stock.",
        "prediction_header": "Prediction Result for",
        "prediction_label": "Expected Daily Volatility for the Next 5 Days",
        "volatility_high": "High Volatility Expected: Sharp price movements may occur.",
        "volatility_medium": "Moderate Volatility Expected: Moderate price movements may occur.",
        "volatility_low": "Low Volatility Expected: Prices are expected to be more stable.",
        "chart_title": "Last 1-Month Closing Price Chart",
        "details_header": "Project Goal and Technical Details",
        "details_text": """
        The goal of this project is to predict **volatility**, a measure of risk in financial markets. Volatility indicates how much a stock's price fluctuates over a specific period. High volatility implies high risk and potentially high returns.
        
        - **Model:** XGBoost Regressor
        - **Data Source:** Live data is fetched from the Alpha Vantage API using the `alpha-vantage` library.
        - **Features:** The model uses time-series features such as past returns (lag features), moving averages, and historical volatility to make predictions.

        ---

        ### **Project Scope & Limitations**

        This application is a **Proof of Concept (PoC) project, developed in a single day** as part of my **#30DaysOfAI** challenge (Day 5). Its primary goal was to practice the end-to-end pipeline of fetching data from a live API, feature engineering for time series, training a machine learning model, and deploying it in an interactive web interface.

        In this context, it's important to note that the current model is **not an investment advisory tool** and would require additional metrics and improvements for real-world applications, such as:

        * **Comprehensive Model Evaluation:** The model has not yet been evaluated on a test set using standard regression metrics like R-squared ($R^2$) or Mean Absolute Error (MAE).
        * **Limited Features:** Predictions are based solely on historical price movements. Important metrics like trading volume, technical indicators (e.g., RSI, Bollinger Bands), or fundamental analysis data (e.g., company news, financial reports) have not yet been included.
        * **Future Improvement Opportunities:** To make the model more accurate and reliable, future steps could include training it on a diverse range of stocks from various sectors, performing hyperparameter optimization, and presenting predictions with a confidence interval.
        """,
        "language_select_label": "Language"
    }
}

if 'language_key' not in st.session_state:
    st.session_state.language_key = "tr"

st.set_page_config(page_title=LANGUAGES[st.session_state.language_key]["page_title"], page_icon="📈", layout="wide")

col1, col2 = st.columns([4, 1])

with col1:
    st.title(LANGUAGES[st.session_state.language_key]["title"])

with col2:
    current_index = 0 if st.session_state.language_key == "tr" else 1
    
    selected_language = st.selectbox(
        label=LANGUAGES[st.session_state.language_key]["language_select_label"],
        options=["Türkçe", "English"],
        index=current_index,
        key='language_selector'
    )
    st.session_state.language_key = "tr" if selected_language == "Türkçe" else "en"


lang_display = LANGUAGES[st.session_state.language_key]


@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model.pkl')
        model_features = joblib.load('model_features.pkl')
        return model, model_features
    except FileNotFoundError:
        return None, None

model, model_features = load_resources()

if not model or not model_features:
    st.error("Gerekli model dosyaları bulunamadı. Lütfen önce `train_model.py`'yi çalıştırdığınızdan emin olun.")
    st.stop()
    
tab1, tab2 = st.tabs([f"**{lang_display['tab_predict']}**", f"**{lang_display['tab_details']}**"])


with tab1:
    st.header(lang_display["header_predict"])
    st.write(lang_display["description_predict"])

    popular_tickers = ['TSLA', 'GOOGL', 'MSFT', 'NVDA', 'AAPL', 'AMZN', 'META', 'NFLX']
    ticker = st.selectbox(lang_display["select_stock"], popular_tickers, index=0)

    if st.button(lang_display["button_predict"]):
        try:
            with st.spinner(f"{ticker} {lang_display['spinner_text']}"):
                API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"] 
                ts = TimeSeries(key=API_KEY, output_format='pandas')
                df_live, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')

                df_live.rename(columns={
                    '1. open': 'Open', '2. high': 'High',
                    '3. low': 'Low', '4. close': 'Close',
                    '5. volume': 'Volume'
                }, inplace=True)
                df_live.index = pd.to_datetime(df_live.index)
                df_live = df_live.sort_index()
                
                if df_live.empty:
                    st.error(f"'{ticker}' {lang_display['error_no_data']}")
                else:
                    df_live['Returns'] = df_live['Close'].pct_change()
                    for i in range(1, 6):
                        df_live[f'Lag_{i}'] = df_live['Returns'].shift(i)
                    df_live['MA_10'] = df_live['Returns'].rolling(window=10).mean()
                    df_live['Volatility_10'] = df_live['Returns'].rolling(window=10).std()
                    
                    latest_data = df_live[model_features].tail(1)

                    if latest_data.isnull().values.any():
                        st.warning(lang_display["warning_insufficient_data"])
                    else:
                        prediction = model.predict(latest_data)
                        daily_volatility = prediction[0] * 100

                        st.subheader(f"🔮 {ticker} {lang_display['prediction_header']}")
                        
                        col1_res, col2_res = st.columns([1, 2])
                        
                        with col1_res:
                            st.metric(label=lang_display["prediction_label"], value=f"{daily_volatility:.2f}%")
                            if daily_volatility > 2.5:
                                st.error(lang_display["volatility_high"])
                            elif daily_volatility > 1.5:
                                st.warning(lang_display["volatility_medium"])
                            else:
                                st.success(lang_display["volatility_low"])
                        
                        with col2_res:
                            fig_title = f"{ticker} - {lang_display['chart_title']}"
                            fig = px.line(df_live, x=df_live.index, y='Close', title=fig_title)
                            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

with tab2:
    st.header(lang_display["details_header"])
    st.write(lang_display["details_text"])