"use client";

import { useState } from "react";

import CsvUploadPanel from "@/components/CsvUploadPanel";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { preprocessCsv, validateCsv } from "@/lib/api";
import type {
  PreprocessResponse,
  ValidationResponse,
} from "@/types/data";

const MAX_FILE_SIZE = 20 * 1024 * 1024;

function displayValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [validation, setValidation] =
    useState<ValidationResponse | null>(null);
  const [preprocessing, setPreprocessing] =
    useState<PreprocessResponse | null>(null);
  const [error, setError] = useState("");
  const [loadingAction, setLoadingAction] = useState<
    "validate" | "preprocess" | null
  >(null);

  function selectFile(selectedFile?: File) {
    setValidation(null);
    setPreprocessing(null);
    setError("");

    if (!selectedFile) {
      setFile(null);
      return;
    }
    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setFile(null);
      setError("CSV(.csv) 파일만 선택할 수 있습니다.");
      return;
    }
    if (selectedFile.size > MAX_FILE_SIZE) {
      setFile(null);
      setError("파일 크기는 20MB 이하여야 합니다.");
      return;
    }
    setFile(selectedFile);
  }

  async function handleValidate() {
    if (!file || loadingAction) return;
    setError("");
    setLoadingAction("validate");
    try {
      setValidation(await validateCsv(file));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "데이터 검증 중 오류가 발생했습니다.",
      );
    } finally {
      setLoadingAction(null);
    }
  }

  async function handlePreprocess() {
    if (!file || loadingAction) return;
    setError("");
    setLoadingAction("preprocess");
    try {
      setPreprocessing(await preprocessCsv(file));
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "전처리 중 오류가 발생했습니다.",
      );
    } finally {
      setLoadingAction(null);
    }
  }

  const previewColumns = preprocessing?.preview.length
    ? Object.keys(preprocessing.preview[0])
    : [];

  return (
    <div className="appShell">
      <Sidebar activeItem="데이터 전처리" />
      <div className="contentShell">
        <Header />
        <main className="mainContent uploadPage">
          <section className="uploadIntro">
            <span className="eyebrow">데이터 준비</span>
            <h1>CSV 데이터 전처리</h1>
            <p>
              제조 공정 CSV 파일을 업로드하고 데이터 구조 검증, 이상치 처리 및 결측치 보정을 수행합니다.
            </p>
          </section>

          <section className="uploadCard" aria-labelledby="upload-title">
            <div className="sectionHeading compact">
              <div>
                <span className="sectionLabel">파일 선택</span>
                <h2 id="upload-title">공정 CSV</h2>
              </div>
              <p>최대 20MB · utf-8-sig, utf-8, cp949</p>
            </div>

            <CsvUploadPanel
              id="upload-file"
              file={file}
              onFileSelect={selectFile}
              disabled={loadingAction !== null}
            />

            {error && (
              <div className="messageBox error" role="alert">{error}</div>
            )}

            <div className="uploadActions">
              <button
                className="button secondary"
                type="button"
                disabled={!file || loadingAction !== null}
                data-loading={loadingAction === "validate"}
                aria-busy={loadingAction === "validate"}
                onClick={handleValidate}
              >
                {loadingAction === "validate" ? "검증 중..." : "데이터 검증"}
              </button>
              <button
                className="button primary"
                type="button"
                disabled={!file || loadingAction !== null}
                data-loading={loadingAction === "preprocess"}
                aria-busy={loadingAction === "preprocess"}
                onClick={handlePreprocess}
              >
                {loadingAction === "preprocess"
                  ? "전처리 중..."
                  : "전처리 실행"}
              </button>
            </div>
          </section>

          {validation && (
            <section className="resultCard" aria-labelledby="validation-title">
              <div className="resultHeader">
                <div>
                  <span className="sectionLabel">검증 결과</span>
                  <h2 id="validation-title">데이터 품질 확인</h2>
                </div>
                <span
                  className={`resultBadge ${
                    validation.validation.is_valid ? "success" : "error"
                  }`}
                >
                  {validation.validation.is_valid ? "정상" : "오류"}
                </span>
              </div>
              <div className="metricGrid">
                <div><span>행</span><strong>{validation.row_count}</strong></div>
                <div><span>열</span><strong>{validation.column_count}</strong></div>
                <div>
                  <span>결측값</span>
                  <strong>{validation.validation.total_missing_count}</strong>
                </div>
                <div>
                  <span>중복 ID</span>
                  <strong>
                    {validation.validation.duplicate_wafer_id_count}
                  </strong>
                </div>
              </div>
              {(validation.validation.errors.length > 0 ||
                validation.validation.warnings.length > 0) && (
                <div className="resultMessages">
                  {validation.validation.errors.map((message) => (
                    <p className="resultMessage errorText" key={message}>
                      오류 · {message}
                    </p>
                  ))}
                  {validation.validation.warnings.map((message) => (
                    <p className="resultMessage warningText" key={message}>
                      경고 · {message}
                    </p>
                  ))}
                </div>
              )}
              <div className="detectedColumns">
                {Object.entries(
                  validation.validation.detected_columns,
                ).map(([group, columns]) => (
                  <div key={group}>
                    <strong>{group.toUpperCase()}</strong>
                    <span>{columns.length ? columns.join(", ") : "없음"}</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {preprocessing && (
            <section className="resultCard" aria-labelledby="preprocess-title">
              <div className="resultHeader">
                <div>
                  <span className="sectionLabel">전처리 결과</span>
                  <h2 id="preprocess-title">처리 전후 요약</h2>
                </div>
                <span className="resultBadge success">완료</span>
              </div>
              <div className="comparisonGrid">
                <div>
                  <span>처리 전 결측값</span>
                  <strong>{preprocessing.before.missing_count}</strong>
                </div>
                <div>
                  <span>처리 후 결측값</span>
                  <strong>{preprocessing.after.missing_count}</strong>
                </div>
                <div>
                  <span>대체한 결측값</span>
                  <strong>
                    {preprocessing.changes.filled_missing_values}
                  </strong>
                </div>
                <div>
                  <span>보정한 이상치</span>
                  <strong>{preprocessing.changes.clipped_outliers}</strong>
                </div>
              </div>

              <div className="previewHeader">
                <div>
                  <span className="sectionLabel">데이터 미리보기</span>
                  <h3>전처리된 데이터 상위 {preprocessing.preview.length}행</h3>
                </div>
              </div>
              <div className="tableWrap">
                <table>
                  <thead>
                    <tr>
                      {previewColumns.map((column) => (
                        <th key={column} scope="col">{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preprocessing.preview.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {previewColumns.map((column) => (
                          <td key={column}>{displayValue(row[column])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <section className="resultCard preprocessingRulesCard" aria-labelledby="preprocessing-rules-title">
            <div className="sectionHeading compact">
              <div>
                <span className="sectionLabel">Data Preparation</span>
                <h2 id="preprocessing-rules-title">데이터 전처리 규칙</h2>
              </div>
              <span className="infoCardIcon" aria-hidden="true">i</span>
            </div>
            <ol className="preprocessingRuleList">
              <li>
                <strong>이상치 처리</strong>
                <span>LOT 기준 평균 ± 3σ 범위를 벗어나는 값은 LOT 평균값으로 대체합니다.</span>
              </li>
              <li>
                <strong>결측치 처리</strong>
                <span>결측값은 해당 LOT의 평균값으로 대체합니다.</span>
              </li>
              <li>
                <strong>데이터 검증</strong>
                <span>컬럼 구조, 데이터 타입, 결측 여부, 중복 여부를 검사합니다.</span>
              </li>
              <li>
                <strong>전처리 결과</strong>
                <span>처리가 완료된 데이터를 CSV로 다시 생성합니다.</span>
              </li>
              <li>
                <strong>분석 기준</strong>
                <span>모든 분석은 전처리가 완료된 데이터를 기준으로 수행합니다.</span>
              </li>
            </ol>
          </section>
        </main>
      </div>
    </div>
  );
}
