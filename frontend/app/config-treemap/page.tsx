"use client";

import { useEffect, useRef, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import ConfigTreemap from "@/components/ConfigTreemap";
import DashboardShell from "@/components/DashboardShell";
import { PageHeaderMeta } from "@/components/LastRunNote";
import SampleNotice from "@/components/SampleNotice";
import { CONFIG_SCREENING_PASS_COUNT, CONFIG_SCREENING_TEST_COUNT } from "@/lib/fmeaFormat";
import { getConfigTreemap, getDatasetSchema } from "@/lib/api";
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
  const { analysis } = useAnalysisState();
  const datasetId = analysis?.dataset ?? "train";

  const [stepOptions, setStepOptions] = useState<number[]>([]);
  const [optionsDataset, setOptionsDataset] = useState("");
  const [step, setStep] = useState(1);
  const [cache, setCache] = useState<StepCache>({});
  const [loadingStep, setLoadingStep] = useState<number | null>(null);
  // 데이터셋이 바뀌면(다른 원인 분석 결과) 이전 캐시는 무효하다.
  const cachedDatasetRef = useRef(datasetId);

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
        const storedStep = Number(window.localStorage.getItem(`config-treemap-step:${datasetId}`));
        const nextStep = available.includes(storedStep) ? storedStep : available.includes(step) ? step : (available[0] ?? step);
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

  return (
    <DashboardShell activeItem="Config별 트리맵">
      <div className="rcPage">
        <div className="pageHeading">
          <h1>Config별 트리맵</h1>
          <p>스텝 하나를 고르면 Model → EQ → Chamber 조합별 Y1~Y5 평균 불량률을 한 번에 봅니다.</p>
          <PageHeaderMeta />
          <SampleNotice sampleInfo={sampleInfo} />
          {!anySignificant && (
            <p className="sectionCaption">
              FDR 보정 후 유의한 Config 조합이 없습니다 ({CONFIG_SCREENING_TEST_COUNT}건 검정, 통과 {CONFIG_SCREENING_PASS_COUNT}건).
              색 대신 수치로 비교하세요.
            </p>
          )}
        </div>

        {TARGETS.map((target, index) => (
          <ConfigTreemap
            key={target}
            target={target}
            step={step}
            data={stepData?.[target] ?? null}
            loading={isLoading}
            headerRight={
              index === 0 ? (
                <label className="monitoringStepSelect">
                  스텝
                  <select value={step} onChange={(event) => setStep(Number(event.target.value))}>
                    {stepOptions.map((s) => (
                      <option key={s} value={s}>Step{s}</option>
                    ))}
                  </select>
                </label>
              ) : undefined
            }
          />
        ))}
      </div>
    </DashboardShell>
  );
}
