"""SQL 데이터 소스 판단 + 증분 수집. 접속 정보(host/port/db/user)는
`RuntimeStore.get_automation_settings()`("알림·자동화 설정" 팝업이
저장하는 전용 슬롯)에서 읽는다 -- 학습 설정(`state/training`)과는 다른
슬롯이다. 자동화는 학습과 무관한 별개 개념이라 섞으면 서로의 필드를
덮어쓴다.
서버 환경변수(`AUTO_INGEST_DB_DRIVER`/`DB_PASSWORD`/`AUTO_INGEST_QUERY`)가
모두 갖춰져 있고 실제 접속·조회에 성공하면 SQL 모드다. 하나라도
빠지거나 접속/조회가 실패하면 호출부(`src/automation/refresh.py`,
`src/automation/yield_dispatch.py`)가 곧장 폴백 모드로 넘어간다 -- 이
모듈은 그 판단에 필요한 최소 기능만 제공하고, 실패를 절대 예외로 밖에
던지지 않는다(스케줄러 루프를 죽이면 안 된다).

DB 엔진을 코드에 고정하지 않는다: 팹마다 SQL 엔진이 다르므로
`AUTO_INGEST_DB_DRIVER`(SQLAlchemy dialect+driver 접두사, 예:
"postgresql+psycopg2")를 운영팀이 직접 설정하고, 그 드라이버 패키지도
운영팀이 자신의 배포 환경에 별도 설치한다.
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import pandas as pd

from api.settings import settings
from src.runtime.store import RuntimeStore

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 10
# app_state 키 -- 마지막으로 처리한 커서(시각 또는 ID) 값을 저장한다.
CURSOR_STATE_KEY = "automation:sql_cursor"


@dataclass(frozen=True)
class SqlConnectionInfo:
    host: str
    port: str
    db: str
    user: str


def _stored_connection_info(store: RuntimeStore) -> SqlConnectionInfo | None:
    payload = store.get_automation_settings()
    host = str(payload.get("sqlHost") or "").strip()
    port = str(payload.get("sqlPort") or "").strip()
    db = str(payload.get("sqlDb") or "").strip()
    user = str(payload.get("sqlUser") or "").strip()
    if not (host and port and db and user):
        return None
    return SqlConnectionInfo(host=host, port=port, db=db, user=user)


def is_sql_configured(store: RuntimeStore) -> bool:
    """드라이버·쿼리·접속 정보가 전부 있어야 "SQL 모드 시도 대상"이다 --
    하나라도 빠지면 접속 시도 자체를 하지 않고 곧장 폴백이다."""
    if not settings.db_driver or not settings.auto_ingest_query:
        return False
    return _stored_connection_info(store) is not None


def _build_engine_url(info: SqlConnectionInfo) -> str:
    password = quote_plus(settings.db_password or "")
    user = quote_plus(info.user)
    return f"{settings.db_driver}://{user}:{password}@{info.host}:{info.port}/{info.db}"


def _connect_only(url: str) -> None:
    # 지연 import -- 이유는 `_run_query`와 같다.
    from sqlalchemy import create_engine

    engine = create_engine(url)
    try:
        with engine.connect():
            pass
    finally:
        engine.dispose()


def test_connection(store: RuntimeStore) -> tuple[bool, str | None]:
    """"연결 테스트" 버튼 -- 쿼리는 실행하지 않고 접속만 확인한다
    (운영 쿼리는 부작용이 있을 수 있어 테스트에는 쓰지 않는다). 실패를
    예외로 던지지 않는다(다른 채널의 "테스트 발송"과 같은 (ok, error)
    관례)."""
    if not settings.db_driver:
        return False, "서버에 DB 드라이버(AUTO_INGEST_DB_DRIVER)가 설정되지 않았습니다."
    info = _stored_connection_info(store)
    if info is None:
        return False, "서버 주소·DB명·사용자명을 먼저 저장하세요."
    try:
        url = _build_engine_url(info)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_connect_only, url)
            future.result(timeout=CONNECT_TIMEOUT_SECONDS)
        return True, None
    except concurrent.futures.TimeoutError:
        return False, f"{CONNECT_TIMEOUT_SECONDS}초 안에 접속하지 못했습니다."
    except Exception as exc:
        return False, str(exc)[:300]


def _run_query(url: str, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    # 지연 import -- sqlalchemy는 SQL 모드를 실제로 쓰는 배포에서만
    # 필요하고, 이 함수는 스레드풀에서 타임아웃과 함께 호출되므로 무거운
    # import를 호출부(fetch_incremental)의 try 블록 밖에 둔다.
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query), params)
            return [dict(row) for row in result.mappings().all()]
    finally:
        engine.dispose()


def fetch_incremental(store: RuntimeStore) -> pd.DataFrame | None:
    """SQL 모드 시도: 성공하면 새 행(0행일 수 있다)을 담은 DataFrame을
    반환하고 커서를 갱신한다. 접속 정보 미비/접속 실패/쿼리 실패/타임아웃
    은 전부 `None`을 반환한다 -- 호출부는 `None`이면 폴백으로 넘어간다.
    이 함수는 절대 예외를 던지지 않는다.

    타임아웃은 DB 드라이버마다 커넥트 타임아웃 키워드가 달라(예:
    psycopg2는 "connect_timeout", pyodbc는 "timeout") 특정 드라이버에
    종속된 kwarg를 코드에 고정하지 않는다 -- 대신 별도 스레드에서 실행해
    벽시계 타임아웃을 건다. `concurrent.futures`는 실행 중인 스레드를
    강제 종료할 수 없으므로, 타임아웃 시 이 함수는 곧장 반환하지만
    (스케줄러가 계속 돌아가야 하므로) 원래 쿼리 스레드는 배경에서 계속
    실행되다 스스로 끝날 수 있다 -- Python 표준 라이브러리의 알려진 한계.
    """
    info = _stored_connection_info(store)
    if info is None or not settings.db_driver or not settings.auto_ingest_query:
        return None

    cursor_record = store.get_app_state(CURSOR_STATE_KEY)
    last_cursor = (cursor_record or {}).get("value")
    params: dict[str, Any] = {}
    if settings.auto_ingest_cursor_column and last_cursor is not None:
        # 쿼리 자체는 환경변수라 커서 조건을 강제로 주입하지 않는다 --
        # 팹마다 커서 컬럼·비교 연산자가 다르므로, 쿼리 작성자가 직접
        # `:cursor` 바인드 파라미터를 넣어야 한다. 쿼리에 `:cursor`가
        # 없으면 SQLAlchemy가 미사용 파라미터로 조용히 무시한다.
        params["cursor"] = last_cursor

    try:
        url = _build_engine_url(info)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_query, url, settings.auto_ingest_query, params)
            rows = future.result(timeout=CONNECT_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        logger.warning("auto_refresh: SQL 접속/쿼리가 %d초를 넘겨 폴백으로 전환합니다.", CONNECT_TIMEOUT_SECONDS)
        return None
    except Exception:
        logger.exception("auto_refresh: SQL 접속/쿼리 실패 -- 폴백으로 전환합니다.")
        return None

    dataframe = pd.DataFrame(rows)
    if (
        settings.auto_ingest_cursor_column
        and settings.auto_ingest_cursor_column in dataframe.columns
        and not dataframe.empty
    ):
        new_cursor_raw = dataframe[settings.auto_ingest_cursor_column].max()
        new_cursor = new_cursor_raw.isoformat() if hasattr(new_cursor_raw, "isoformat") else new_cursor_raw
        if hasattr(new_cursor, "item"):
            new_cursor = new_cursor.item()
        store.set_app_state(CURSOR_STATE_KEY, {"value": new_cursor})
    return dataframe
