"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import ChartExportButton from "@/components/ChartExportButton";
import ConfigTreemap from "@/components/ConfigTreemap";
import DashboardShell from "@/components/DashboardShell";
import { FavoriteStarButton } from "@/components/FavoriteStarButton";
import { PageHeader } from "@/components/PageHeader";
import SampleNotice from "@/components/SampleNotice";
import { CONFIG_SCREENING_PASS_COUNT, CONFIG_SCREENING_TEST_COUNT } from "@/lib/fmeaFormat";
import { buildTreemapCaptionText, buildTreemapExportFilename } from "@/lib/chartExport";
import { getConfigTreemap, getDatasetSchema } from "@/lib/api";
import { useFavoriteToggle } from "@/lib/useFavoriteToggle";
import type { ConfigTreemapResponse } from "@/types/data";

const TARGETS = ["Y1", "Y2", "Y3", "Y4", "Y5"] as const;

type StepCache = Record<number, Record<string, ConfigTreemapResponse | null>>;

/** Config별 트리맵 탭 -- 설비 구성 트리맵 전용 화면.
 * Y1~Y5 다섯 트리맵을 세로로 일렬 배치하고, 스텝 선택은 첫
 * 카드(Y1)의 헤더에 하나만 둔다 -- 다섯 개가 함께 바뀐다(같은 컨트롤이
 * 다섯 번 반복되지 않도록 ConfigTreemap의 `headerRight` slot으로 넘긴다).
 * 다섯 응답을 Promise.all로 한 번에 묶어 스텝별로 캐시한다 --
 * 스텝을 다시 선택해도 재조회하지 않는다. */
export default function ConfigTreemapPage() {
  return (
    <Suspense fallback={null}>
      <ConfigTreemapContent />
    </Suspense>
  );
}

function ConfigTreemapContent() {
  const { analysis, training } = useAnalysisState();
  const datasetId = analysis?.dataset ?? "train";
  const championVersion = training?.performance.model_id ?? null;
  const searchParams = useSearchParams();
  const { isFavorited, isFavoritePending, toggleFavorite } = useFavoriteToggle();

  const [stepOptions, setStepOptions] = useState<number[]>([]);
  const [optionsDataset, setOptionsDataset] = useState("");
  const [step, setStep] = useState(1);
  const [cache, setCache] = useState<StepCache>({});
  const [loadingStep, setLoadingStep] = useState<number | null>(null);
  // 데이터셋이 바뀌면(다른 원인 분석 결과) 이전 캐시는 무효하다.
  const cachedDatasetRef = useRef(datasetId);
  // 즐겨찾기에서 열었을 때(`?step=&target=`) 스텝을 그 값으로
  // 복원한다 -- 로컬 저장소보다 우선한다. 마운트 시 한 번만 적용한다.
  const stepFromUrlRef = useRef<number | null>((() => {
    const raw = Number(searchParams.get("step"));
    return Number.isFinite(raw) && raw > 0 ? raw : null;
  })());
  const treemapDeepLinkHandledRef = useRef(false);

  useEffect(() => {
    if (cachedDatasetRef.current !== datasetId) {
      cachedDatasetRef.current = datasetId;
      setCache({});
    }
    let cancelled = false;
    void getDatasetSchema(datasetId)
      .then((schema) => {
        if (cancelled) return;
        const available = schema.config_steps.length > 0 ? schema.config_steps : schema.steps_present;
        const urlStep = stepFromUrlRef.current;
        const storedStep = Number(window.localStorage.getItem(`config-treemap-step:${datasetId}`));
        const nextStep =
          urlStep != null && available.includes(urlStep)
            ? urlStep
            : available.includes(storedStep)
              ? storedStep
              : available.includes(step)
                ? step
                : (available[0] ?? step);
        setStepOptions(available);
        setStep(nextStep);
        setOptionsDataset(datasetId);
      })
      .catch(() => {
        if (cancelled) return;
        setStepOptions([step]);
        setOptionsDataset(datasetId);
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  // 스텝 데이터가 준비되면 즐겨찾기가 가리킨 타깃 카드로 스크롤한다.
  useEffect(() => {
    if (treemapDeepLinkHandledRef.current) return;
    const targetFromUrl = searchParams.get("target");
    if (!targetFromUrl || !cache[step]) return;
    treemapDeepLinkHandledRef.current = true;
    const timer = window.setTimeout(() => {
      document.getElementById(`treemap-${targetFromUrl}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 0);
    return () => window.clearTimeout(timer);
  }, [cache, step, searchParams]);

  useEffect(() => {
    if (optionsDataset !== datasetId) return;
    if (cache[step]) return; // 이미 캐시된 스텝 -- 재조회하지 않는다.
    let cancelled = false;
    setLoadingStep(step);
    window.localStorage.setItem(`config-treemap-step:${datasetId}`, String(step));
    // 다섯 타깃을 Promise.all로 한 번에 묶어 요청한다 -- 스텝
    // 변경 시 5회가 아니라 1묶음(내부적으로는 5개 fetch가 동시에
    // 나가지만, UI는 한 번의 로딩 상태로 취급한다).
    void Promise.all(
      TARGETS.map((target) =>
        getConfigTreemap(datasetId, step, target).catch(() => null),
      ),
    ).then((results) => {
      if (cancelled) return;
      const byTarget: Record<string, ConfigTreemapResponse | null> = {};
      TARGETS.forEach((target, index) => { byTarget[target] = results[index]; });
      setCache((prev) => ({ ...prev, [step]: byTarget }));
      setLoadingStep(null);
    });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, optionsDataset, step]);

  const stepData = cache[step];
  const isLoading = loadingStep === step;
  const anySignificant = stepData ? TARGETS.some((t) => stepData[t]?.significant) : false;
  // 다섯 타깃이 같은 표본을 쓰므로(같은 dataset_version) 아무거나
  // 하나에서 고지를 읽으면 충분하다.
  const sampleInfo = stepData ? (TARGETS.map((t) => stepData[t]?.sample_info).find(Boolean) ?? null) : null;
  // 이미지 저장 캡션의 wafer 수 -- 이 스텝에서 실제로 집계된 wafer 총합
  // (챔버별 n의 합). 데이터셋 행 수 자체는 이 페이지가 갖고 있지 않지만,
  // Config가 채워진 wafer 수가 곧 "이 트리맵이 담은 표본 수"다.
  const treemapWaferTotal = stepData
    ? Math.max(...TARGETS.map((t) => stepData[t]?.groups.reduce((sum, g) => sum + g.n, 0) ?? 0))
    : 0;
  const treemapExportRef = useRef<HTMLDivElement | null>(null);

  return (
    <DashboardShell activeItem="Config별 트리맵">
      <div className="rcPage">
        <PageHeader
          eyebrow="장비 구성"
          title="Config별 트리맵"
          description="스텝 하나를 고르면 Model → EQ → Chamber 조합별 Y1~Y5 평균 불량률을 한 번에 봅니다."
        >
          <SampleNotice sampleInfo={sampleInfo} />
        </PageHeader>

        {/* FDR 안내는 다섯 트리맵 바로 위, 이미지 저장 캡처 영역 안쪽에
            둔다 -- 여기 있어야 "이미지 저장" 버튼으로 내보낸 PNG에도
            함께 담겨, 이미지만 봐도 왜 타일이 전부 중립색인지 알 수 있다. */}
        <div ref={treemapExportRef}>
          {!anySignificant && (
            <p className="sectionCaption">
              FDR 보정 후 유의한 Config 조합이 없습니다 ({CONFIG_SCREENING_TEST_COUNT}건 검정, 통과 {CONFIG_SCREENING_PASS_COUNT}건).
              색 대신 수치로 비교하세요.
            </p>
          )}
          {TARGETS.map((target, index) => {
            const favoriteSnapshot = {
              dataset: datasetId,
              target,
              feature: "",
              viewType: "treemap" as const,
              step,
            };
            return (
              <div key={target} id={`treemap-${target}`}>
                <ConfigTreemap
                  target={target}
                  step={step}
                  data={stepData?.[target] ?? null}
                  loading={isLoading}
                  headerRight={
                    <div className="monitoringTreemapControls">
                      {/* 트리맵은 타깃마다(Y1~Y5 각각) 별을 단다 -- 이미지
                          저장은 다섯 개를 한 장으로 묶지만, 즐겨찾기는
                          트리맵 하나씩 개별로 저장한다. */}
                      <FavoriteStarButton
                        favorited={isFavorited(favoriteSnapshot)}
                        disabled={isFavoritePending(favoriteSnapshot)}
                        onClick={() =>
                          toggleFavorite({
                            ...favoriteSnapshot,
                            isConfig: true,
                            interpretation: `Config vs ${target} 트리맵 · Step${step}`,
                            championVersion,
                          })
                        }
                      />
                      {index === 0 && (
                        <>
                          <ChartExportButton
                            nodeRef={treemapExportRef}
                            buildOptions={() => ({
                              filename: buildTreemapExportFilename(step),
                              captionText: buildTreemapCaptionText({ step, totalWafers: treemapWaferTotal, datasetId }),
                            })}
                            title="Y1~Y5 트리맵을 한 이미지로 저장 (PNG)"
                          />
                          <label className="monitoringStepSelect">
                            스텝
                            <select value={step} onChange={(event) => setStep(Number(event.target.value))}>
                              {stepOptions.map((s) => (
                                <option key={s} value={s}>Step{s}</option>
                              ))}
                            </select>
                          </label>
                        </>
                      )}
                    </div>
                  }
                />
              </div>
            );
          })}
        </div>
      </div>
    </DashboardShell>
  );
}
