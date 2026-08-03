import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";

export default function AlertsPage() {
  return (
    <div className="appShell">
      <Sidebar activeItem="사전 알람 로그" />
      <div className="contentShell">
        <Header />
        <main className="mainContent uploadPage">
          <section className="uploadIntro pageHeading">
            <span className="eyebrow">PRE-ALERT LOG</span>
            <h1>사전 알람 로그</h1>
            <p>정상범위 이탈 기반 알람 로그 화면을 재구성하는 중입니다.</p>
          </section>
          <section className="resultCard">
            <p className="emptyMessage">
              이 탭은 학습 데이터의 정상범위와 평가 데이터의 이탈 여부를 비교하는 방식으로 다시 구축될 예정입니다.
            </p>
          </section>
        </main>
      </div>
    </div>
  );
}
