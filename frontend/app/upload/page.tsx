"use client";

import {
  ChangeEvent,
  DragEvent,
  useRef,
  useState,
} from "react";

import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import { preprocessCsv, validateCsv } from "@/lib/api";
import type {
  PreprocessResponse,
  ValidationResponse,
} from "@/types/data";

const MAX_FILE_SIZE = 20 * 1024 * 1024;

function formatFileSize(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

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
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [validation, setValidation] =
    useState<ValidationResponse | null>(null);
  const [preprocessing, setPreprocessing] =
    useState<PreprocessResponse | null>(null);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
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

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
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
      <Sidebar activeItem="데이터 업로드" />
      <div className="contentShell">
        <Header />
        <main className="mainContent uploadPage">
          <section className="uploadIntro">
            <span className="eyebrow">데이터 준비</span>
            <h1>CSV 데이터 업로드</h1>
            <p>
              공정 데이터를 검증하고 설정 기반 전처리 결과를 미리 확인합니다.
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

            <div
              className={`dropZone ${isDragging ? "dragging" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragEnter={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  inputRef.current?.click();
                }
              }}
              role="button"
              tabIndex={0}
            >
              <input
                ref={inputRef}
                className="visuallyHidden"
                type="file"
                accept=".csv,text/csv"
                onChange={handleInputChange}
              />
              <span className="dropIcon" aria-hidden="true">CSV</span>
              <strong>CSV 파일을 드래그하거나 클릭하여 선택하세요.</strong>
              <span>업로드 파일은 서버에 영구 저장되지 않습니다.</span>
            </div>

            {file && (
              <div className="selectedFile" aria-live="polite">
                <div>
                  <span>선택된 파일</span>
                  <strong>{file.name}</strong>
                </div>
                <span>{formatFileSize(file.size)}</span>
              </div>
            )}

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
        </main>
      </div>
    </div>
  );
}
