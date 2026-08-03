import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";

export default function RootCausePage() {
  return (
    <div className="appShell">
      <Sidebar activeItem="원인 분석" />
      <div className="contentShell">
        <Header />
        <main className="mainContent uploadPage">
          <section className="uploadIntro pageHeading">
            <span className="eyebrow">ROOT CAUSE</span>
            <h1>원인 분석</h1>
            <p>Spotfire식 산점도 기반 원인 분석 화면을 재구성하는 중입니다.</p>
          </section>
          <section className="resultCard">
            <p className="emptyMessage">
              이 탭은 인자 스크리닝(ε² + BH-FDR)과 정상범위 기반 산점도로 다시 구축될 예정입니다.
            </p>
          </section>
        </main>
      </div>
    </div>
  );
}
