# 해밀토니안 선택 규칙: ξ-다발 제약 조건의 엄밀한 공식화

**작성**: 2026-04-14
**상태**: 이론 프레임워크 (수치 검증 진행 중)

---

## 1. 동기

Hilbert-Pólya 추측: "리만 제타 함수의 비자명 영점은 어떤 자기수반 연산자 H의 고유값이다."

이 추측의 문제는 H에 대한 제약이 너무 약하다는 것 — 스펙트럼이 {t_n}인 자기수반 연산자는 무한히 많다. 물리적으로 의미 있는 H를 **선택**하려면 추가 구조가 필요하다.

**우리의 기여**: ξ-다발 프레임워크에서 수치적으로 확립한 28개 결과로부터 8개의 독립적인 제약 조건을 추출하여, Berry-Keating 후보 해밀토니안의 **선택 규칙(selection rule)**으로 사용한다.

이는 응집물질물리에서 Chern 수가 TKNN 해밀토니안의 위상적 분류를 제공하는 것과 구조적으로 동일하다.

---

## 2. ξ-다발 구조 요약

### 2-1. 기본 설정

- **밑공간(base)**: 임계 영역 S = {s = σ + it : 0 < σ < 1}
- **올(fiber)**: C* = C \ {0}
- **사영**: π : E → S,  단면 ξ(s)
- **접속**: L(s) = ξ'/ξ(s) = (d/ds) log ξ(s)
- **곡률**: κ(s) = |L(s)|²
- **모노드로미**: 영점 ρ_n 주위 Δarg ξ = ±π

### 2-2. 핵심 등식들

| 등식 | 수학적 내용 | 위상학적 의미 |
|------|------------|-------------|
| ∫∫_R κ dA = 2πN | Gauss-Bonnet | 곡률 적분 = 2π × 영점 수 |
| ∮_∂R Im(L) ds = 2πN | Stokes 변환 | 경계 위상 누적 = 2π × 영점 수 |
| Δarg ξ\|_{γ_n} = ±π | 모노드로미 | 단순 영점의 반-뒤틀림 |
| Re(L(1/2+it)) = 0 | 유니터리 게이지 | 임계선의 특수성 |

---

## 3. 8개 제약 조건

### C1: Gauss-Bonnet 정수성 (위상적 양자화)

**정의**:
$$\frac{1}{2\pi} \iint_R \Delta(\log|\xi|) \, d\sigma \, dt = N \in \mathbb{Z}$$

여기서 R은 임계 영역 내 충분히 큰 직사각형, N은 R 내 영점 수.

**해밀토니안 함의**: H의 스펙트럼 제타 함수 ζ_H(s) = Σ λ_n^{-s} 가 동일한 위상적 양자화를 만족해야 한다. 즉:
$$\frac{1}{2\pi} \oint_{|s-\rho|=\epsilon} \frac{\zeta_H'}{\zeta_H}(s) \, ds = 1 \quad \text{(단순 영점마다)}$$

**수치 근거**: 결과 #5 (Gauss-Bonnet 2πN 폐합), #6 (Chern 수), #11 (디리클레 보편성)

**검증 방법**: `HamiltonianConstraints.check_c1_gauss_bonnet()`

---

### C2: 유니터리 게이지 (임계선 제약)

**정의**:
$$\text{Re}\left(\frac{\xi'}{\xi}(\tfrac{1}{2} + it)\right) = 0 \quad \forall t \in \mathbb{R}$$

이는 ξ의 함수방정식 ξ(s) = ξ(1-s)에서 자동으로 도출된다.

**해밀토니안 함의**: H는 스펙트럼이 임계선에 "사영"되는 연산자여야 한다. 구체적으로:
- H의 Green 함수 G(E) = (E - H)^{-1}의 스펙트럼 함수가 Re Tr G(E+iε) = 0을 임계선에서 만족

**수치 근거**: 결과 #3 (접속 반대칭), #8 (σ-국소화)

---

### C3: Klein 4-군 대칭

**정의**: 해밀토니안은 Z₂ × Z₂ 대칭을 가져야 한다:
- S₁: s → 1-s (함수방정식) → 시간 역전 T
- S₂: s → s̄ (복소 켤레) → 입자-구멍 대칭 C
- S₃ = S₁S₂: s → 1-s̄

$$[H, T] = 0, \quad [H, C] = 0$$

**해밀토니안 함의**:
- 고유값은 실수 (T-대칭)
- 양/음 고유값이 쌍을 이룸 (C-대칭) — 단, chiral 대칭이 아닌 산술적 대칭
- 결합 TC 대칭은 스펙트럼 강성(rigidity) 유발

**수치 근거**: 결과 #1 (Klein 대칭), #3 (접속 반대칭), #4 (곡률 집중)

---

### C4: 모노드로미 양자화

**정의**: H의 각 고유값 λ_n에 대해, resolvent의 위상 변화가 ±π:
$$\oint_{|z-\lambda_n|=\epsilon} d\,\text{arg}\,\det(z - H) = \pm\pi$$

단순 고유값이면 위상 변화는 정확히 ±π (특성다항식의 단순근).

**해밀토니안 함의**:
- 모든 고유값이 단순(비퇴화) — degeneracy 금지
- ±π (2π가 아님)는 ξ의 반-정수 모노드로미 (ξ² ≠ ξ)

**수치 근거**: 결과 #2 (모노드로미 ±π), #7 (다발 적대적 검증)

---

### C5: 곡률 집중 (스펙트럼 국한)

**정의**: κ(s) = |ξ'/ξ(s)|²는 영점에서만 발산하고, 영점에서 멀어지면 빠르게 감쇠:
$$\kappa(\sigma + it) \sim \frac{1}{|s - \rho_n|^2} \quad (s \to \rho_n)$$

**해밀토니안 함의**:
- spectral density ρ(E) = Σ δ(E - λ_n)이 "국한" — 고유값 외에서 0
- 이것은 자명해 보이지만, **곡률의 감쇠율**이 추가 제약을 준다
- 구체적으로: Lorentzian 감쇠율이 고유값 간격과 관련

**수치 근거**: 결과 #4 (곡률 집중), #8 (σ-국소화)

---

### C6: 자기수반성 (스펙트럼 실수성)

**정의**: H = H† (에르미트), 따라서 스펙트럼이 실수.

**완화**: PT-대칭 — 비에르미트이지만 PT H (PT)^{-1} = H이면 실수 스펙트럼 가능.

**해밀토니안 함의**:
- 에르미트 경로: 경계 조건으로 자기수반 확장 (Berry-Keating의 미해결 문제)
- PT 경로: Bender-Brody-Müller (2017)가 시도했으나 Bellissard (2017)가 순환논증 지적

**수치 근거**: 리만 가설 ↔ 영점이 모두 임계선 위 ↔ 스펙트럼이 모두 실수

---

### C7: Euler 곱 호환성

**정의**: H의 trace formula가 소수에 대한 항을 포함해야 한다:
$$\text{Tr}\,f(H) = \int f(E)\,d\bar{N}(E) + \sum_p \sum_{k=1}^{\infty} g_k(p) \cdot (\text{oscillatory term})$$

여기서 첫째 항은 평활부(Weyl law), 둘째 항은 소수에 의한 진동부.

**해밀토니안 함의**:
- H는 소수와 관련된 주기 궤도 구조를 가져야 한다
- Berry-Keating: 주기 궤도 = 소수 멱 p^k (Selberg trace formula 유추)
- LeClair-Mussardo: Bethe Ansatz의 S-행렬이 ξ에서 유도 → 자동으로 소수 구조 포함

**수치 근거**: 결과 #14 (Weyl law), #11 (L-함수 보편성)

---

### C8: 반고전 카운팅 일치

**정의**: H의 누적 고유값 밀도가 리만 카운팅 함수와 일치:
$$N_H(T) := \#\{n : \lambda_n \leq T\} \approx \frac{T}{2\pi}\log\frac{T}{2\pi e} + \frac{7}{8}$$

**해밀토니안 함의**:
- 위상 공간 부피가 정확히 Berry-Keating 형태
- 이는 H = xp의 반고전 근사에서 자동 만족 (BK99)
- 실제 양자 수준에서도 만족하려면 정규화 필요

**수치 근거**: 결과 #14 (Weyl law), #28 (고 t 로버스트니스)

---

## 4. 후보 순위표

| 후보 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | 합계 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:----:|
| LeClair-Mussardo | ✅ | ✅ | ✅ | ✅ | ✅ | △ | ✅ | ✅ | 8/8 |
| Yakaboylu | ✅ | ✅ | △ | ✅ | ✅ | △ | △ | ✅ | 6.5/8 |
| Sierra-Townsend | ✅ | △ | ✗ | △ | ✗ | ✗ | △ | ✅ | 3/8 |
| H = xp | ✗ | ✗ | ✗ | ✗ | ✗ | △ | ✗ | ✅ | 1/8 |
| BBM (PT) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | △ | 0.5/8 |

✅ = 만족, △ = 부분/조건부, ✗ = 불만족

---

## 5. 핵심 등가성: Bethe Ansatz ↔ Gauss-Bonnet

**정리 (비형식적)**: LeClair-Mussardo의 Bethe 위상 양자화 조건

$$k_j L + \sum_{m \neq j} \delta(k_j - k_m) = 2\pi n_j$$

은 ξ-다발의 Gauss-Bonnet 조건

$$\oint_{\partial R} \text{Im}(\xi'/\xi) \, ds = 2\pi N$$

과 동일한 위상학적 제약을 표현한다.

**연결 고리**:
1. S-행렬 S(θ) = ξ(1/2+iθ)/ξ(1/2-iθ) → 위상 이동 δ = -i log S = 2 arg ξ(1/2+iθ)
2. Bethe 위상 누적 = ∮ d(arg ξ) = 모노드로미
3. 모노드로미 누적 = Gauss-Bonnet (Stokes)

이 등가성은 `quantum/hamiltonian/bethe_ansatz_gb.py`에서 수치적으로 검증한다.

---

## 6. 실험 프로토콜

### 6-1. DQPT 검증 (5큐비트)

횡자기장 Ising 모형에서 quench → Loschmidt echo 영점 = DQPT 임계점.
κ 프로파일을 coupling에 주입하여 영점 위치 제어 가능 여부 실험.

→ `quantum/dqpt/dqpt_zeta.py`

### 6-2. Floquet 재현 (1큐비트)

ξ 함수를 단일 큐비트 해밀토니안으로 인코딩, echo 극소점 = 영점.
κ와 echo 하강률의 상관 분석.

→ `quantum/dqpt/floquet_zeros.py`

### 6-3. VQE + GB 제약 (5큐비트)

5개 후보 해밀토니안의 유한 차원 트렁케이션에 대해
GB 제약을 비용 함수에 포함한 VQE 최적화.

→ `quantum/vqe/vqe_gauss_bonnet.py`

### 6-4. 위상 코드 (10큐비트)

모노드로미 ±π를 스태빌라이저로 사용하는 양자 에러 보정 코드.
곡률 가중 디코딩의 성능 이점 검증.

→ `quantum/topological/xi_bundle_code.py`

---

## 7. 예상 논문

**제목안**: "Gauss-Bonnet quantization as a selection rule for Hilbert-Pólya operators: from ξ-bundle geometry to quantum simulation"

**대상 저널**: Journal of Physics A: Mathematical and Theoretical

**핵심 메시지**: ξ-다발에서 수치적으로 확립한 위상적 양자화 조건이 Berry-Keating 후보 해밀토니안의 물리적 선택 규칙으로 작동하며, 이를 현재 양자 하드웨어에서 검증할 수 있다.
