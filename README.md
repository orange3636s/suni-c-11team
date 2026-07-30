# 제조 공정 불량 예측 및 원인 분석 AI

## 프로젝트 목표

반도체 제조 공정 데이터를 이용하여

- 수율 예측
- 불량 원인 분석
- SHAP 기반 설명
- Streamlit Dashboard
- n8n 자동화
- Slack 알림

을 구현한다.

---

## 기술 스택

- Python
- Streamlit
- Plotly
- FastAPI
- XGBoost
- LightGBM
- CatBoost
- SHAP
- n8n
- GitHub

---

## 프로젝트 구조

app/
api/
src/
config/
data/
models/
docs/
tests/
workflows/

---

## 실행 방법

### 패키지 설치

```bash
pip install -r requirements.txt
```

### Streamlit 실행

```bash
streamlit run app/streamlit_app.py
```

### FastAPI 실행

```bash
uvicorn api.main:app --reload
```

실행 후 다음 주소에서 API를 확인할 수 있다.

- `GET http://127.0.0.1:8000/`
- `GET http://127.0.0.1:8000/health`
