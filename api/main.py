from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "제조 공정 불량 예측 및 원인 분석 AI"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
